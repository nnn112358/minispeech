#!/usr/bin/env python3
"""Compact non-AR FastSpeech-style acoustic model: OpenJTalk phoneme ids -> 80-mel
(22050/hop256). Uses precomputed VITS-MAS durations (no attention to converge, no
forced alignment at inference: a duration predictor is trained). Edge-light:
conv-based FFT blocks, depthwise-separable, no self-attention. Trains from
fs_manifest.json (phoneme_ids + durations + mel .npy)."""
import sys
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, math, argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class FSSet(Dataset):
    def __init__(self, manifest): self.items = json.load(open(manifest))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        return (torch.LongTensor(it["phoneme_ids"]),
                torch.LongTensor(it["durations"]),
                torch.from_numpy(np.load(it["mel"])).float())  # (80,T)

def collate(b):
    phs, durs, mels = zip(*b)
    B = len(b); Tp = max(p.shape[0] for p in phs); Tf = max(m.shape[1] for m in mels)
    ph = torch.zeros(B, Tp, dtype=torch.long); du = torch.zeros(B, Tp, dtype=torch.long)
    me = torch.zeros(B, 80, Tf); pl = torch.LongTensor([p.shape[0] for p in phs]); ml = torch.LongTensor([m.shape[1] for m in mels])
    for i,(p,d,m) in enumerate(zip(phs,durs,mels)):
        ph[i,:p.shape[0]]=p; du[i,:d.shape[0]]=d; me[i,:,:m.shape[1]]=m
    return ph, du, me, pl, ml

class ConvBlock(nn.Module):  # depthwise-separable conv FFT-like block
    def __init__(self, d, k=5, drop=0.1):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=k//2, groups=d)
        self.pw = nn.Conv1d(d, d, 1)
        self.norm = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d,d*2), nn.ReLU(), nn.Linear(d*2,d))
        self.norm2 = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
    def forward(self, x):  # x: (B,T,d)
        y = x.transpose(1,2); y = self.pw(F.relu(self.dw(y))).transpose(1,2)
        x = self.norm(x + self.drop(y))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x

class DurationPredictor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d,d,3,padding=1), nn.ReLU(), nn.LayerNorm(0) if False else nn.Identity(),
                                 nn.Conv1d(d,d,3,padding=1), nn.ReLU())
        self.lin = nn.Linear(d,1)
    def forward(self, x):  # (B,T,d)->(B,T) log-dur
        y = self.net(x.transpose(1,2)).transpose(1,2)
        return self.lin(y).squeeze(-1)

def length_regulate(x, durations, max_len):  # x:(B,Tp,d), durations:(B,Tp) long
    outs=[]
    for b in range(x.shape[0]):
        ex = torch.repeat_interleave(x[b], durations[b].clamp(min=0), dim=0)  # (sum_d, d)
        if ex.shape[0] < max_len: ex = F.pad(ex, (0,0,0,max_len-ex.shape[0]))
        else: ex = ex[:max_len]
        outs.append(ex)
    return torch.stack(outs,0)  # (B,max_len,d)

class FastSpeech(nn.Module):
    def __init__(self, n_sym, d=256, n_enc=4, n_dec=4, n_mel=80):
        super().__init__()
        self.emb = nn.Embedding(n_sym, d, padding_idx=0)
        self.enc = nn.ModuleList([ConvBlock(d) for _ in range(n_enc)])
        self.dp = DurationPredictor(d)
        self.dec = nn.ModuleList([ConvBlock(d) for _ in range(n_dec)])
        self.out = nn.Linear(d, n_mel); self.d=d
    def posenc(self, x):
        T=x.shape[1]; d=x.shape[2]; pe=torch.zeros(T,d,device=x.device)
        pos=torch.arange(T,device=x.device).float().unsqueeze(1)
        dv=torch.exp(torch.arange(0,d,2,device=x.device).float()*(-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*dv); pe[:,1::2]=torch.cos(pos*dv)
        return x+pe.unsqueeze(0)
    def forward(self, ph, durations=None, max_frames=None):
        x = self.posenc(self.emb(ph))
        for b in self.enc: x = b(x)
        log_dur = self.dp(x.detach())
        if durations is None:
            durations = torch.clamp(torch.round(torch.exp(log_dur)-1), min=1).long()  # >=1 frame/phoneme
            max_frames = max(int(durations.sum(1).max().item()), 1)
        y = self.posenc(length_regulate(x, durations, max_frames))
        for b in self.dec: y = b(y)
        mel = self.out(y).transpose(1,2)  # (B,80,Tf)
        return mel, log_dur

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", default="fs_data/fs_manifest.json")
    ap.add_argument("--out", default="m3_fs_out")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--cosine", action="store_true", help="cosine LR decay to 0")
    a=ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items=json.load(open(a.manifest)); n_sym=max(max(it["phoneme_ids"]) for it in items)+1
    print(f"utts={len(items)} n_sym={n_sym}", flush=True)
    dl=DataLoader(FSSet(a.manifest), batch_size=a.batch_size, shuffle=True, collate_fn=collate, num_workers=0, drop_last=True)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FastSpeech(n_sym).to(dev)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location="cpu")["model"]); print(f"resumed {a.resume}", flush=True)
    nparams=sum(p.numel() for p in model.parameters())
    print(f"params={nparams/1e6:.2f}M device={dev}", flush=True)
    opt=torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs) if a.cosine else None
    for ep in range(1,a.epochs+1):
        model.train(); rm=rd=0.0; t0=time.time()
        for ph,du,me,pl,ml in dl:
            ph,du,me=ph.to(dev),du.to(dev),me.to(dev)
            Tf=me.shape[2]
            mel,log_dur=model(ph,du,Tf)
            # mel mask
            fmask=(torch.arange(Tf,device=dev)[None,:]<ml.to(dev)[:,None]).unsqueeze(1)
            lmel=(F.l1_loss(mel,me,reduction='none')*fmask).sum()/(fmask.sum()*80)
            pmask=(torch.arange(ph.shape[1],device=dev)[None,:]<pl.to(dev)[:,None])
            ldur=(F.mse_loss(log_dur, torch.log(du.float()+1), reduction='none')*pmask).sum()/pmask.sum()
            loss=lmel+ldur
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            rm+=lmel.item(); rd+=ldur.item()
        if sched: sched.step()
        lr=opt.param_groups[0]["lr"]
        print(f"epoch {ep} mel_l1 {rm/len(dl):.4f} dur {rd/len(dl):.4f} lr {lr:.2e} {time.time()-t0:.0f}s", flush=True)
        if ep%a.save_every==0 or ep==a.epochs:
            torch.save({"model":model.state_dict(),"n_sym":n_sym,"epoch":ep}, f"{a.out}/fs_{ep}.pth")
    print("DONE", flush=True)

if __name__=="__main__": main()
