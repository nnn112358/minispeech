#!/usr/bin/env python3
"""MiniSpeech: compact non-AR acoustic model. OpenJTalk phoneme ids -> 80-mel
(22050/hop256). Conv-based blocks (depthwise-separable), no self-attention.
A duration predictor is trained so inference needs no alignment.

Two ways to get the training durations:
  --learn-alignment : SELF-ALIGNING (default-recommended). Learns the phoneme<->mel
      alignment internally (AlignmentEncoder + forward-sum/CTC + MAS), like
      FastPitch/RAD-TTS. NO external aligner (no VITS, no MFA). Manifest only needs
      {phoneme_ids, mel}.
  (otherwise)       : use precomputed {durations} in the manifest (e.g. forced-aligned)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math, argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class FSSet(Dataset):
    def __init__(self, manifest): self.items = json.load(open(manifest))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        dur = torch.LongTensor(it["durations"]) if "durations" in it else torch.zeros(len(it["phoneme_ids"]), dtype=torch.long)
        return (torch.LongTensor(it["phoneme_ids"]), dur, torch.from_numpy(np.load(it["mel"])).float())


def collate(b):
    phs, durs, mels = zip(*b)
    B = len(b); Tp = max(p.shape[0] for p in phs); Tf = max(m.shape[1] for m in mels)
    ph = torch.zeros(B, Tp, dtype=torch.long); du = torch.zeros(B, Tp, dtype=torch.long)
    me = torch.zeros(B, 80, Tf); pl = torch.LongTensor([p.shape[0] for p in phs]); ml = torch.LongTensor([m.shape[1] for m in mels])
    for i, (p, d, m) in enumerate(zip(phs, durs, mels)):
        ph[i, :p.shape[0]] = p; du[i, :d.shape[0]] = d; me[i, :, :m.shape[1]] = m
    return ph, du, me, pl, ml


class ConvBlock(nn.Module):  # depthwise-separable conv FFT-like block
    def __init__(self, d, k=5, drop=0.1):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.pw = nn.Conv1d(d, d, 1)
        self.norm = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d))
        self.norm2 = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
    def forward(self, x):  # (B,T,d)
        y = x.transpose(1, 2); y = self.pw(F.relu(self.dw(y))).transpose(1, 2)
        x = self.norm(x + self.drop(y))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DurationPredictor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.ReLU(), nn.Conv1d(d, d, 3, padding=1), nn.ReLU())
        self.lin = nn.Linear(d, 1)
    def forward(self, x):  # (B,T,d)->(B,T) log-dur
        return self.lin(self.net(x.transpose(1, 2)).transpose(1, 2)).squeeze(-1)


def length_regulate(x, durations, max_len):  # x:(B,Tp,d), durations:(B,Tp) long
    outs = []
    for b in range(x.shape[0]):
        ex = torch.repeat_interleave(x[b], durations[b].clamp(min=0), dim=0)
        if ex.shape[0] < max_len: ex = F.pad(ex, (0, 0, 0, max_len - ex.shape[0]))
        else: ex = ex[:max_len]
        outs.append(ex)
    return torch.stack(outs, 0)


class MiniSpeech(nn.Module):
    def __init__(self, n_sym, d=256, n_enc=4, n_dec=4, n_mel=80, learn_alignment=False):
        super().__init__()
        self.emb = nn.Embedding(n_sym, d, padding_idx=0)
        self.enc = nn.ModuleList([ConvBlock(d) for _ in range(n_enc)])
        self.dp = DurationPredictor(d)
        self.dec = nn.ModuleList([ConvBlock(d) for _ in range(n_dec)])
        self.out = nn.Linear(d, n_mel); self.d = d
        if learn_alignment:
            from sqzw.alignment import AlignmentEncoder
            self.aligner = AlignmentEncoder(n_mel=n_mel, n_text=d)

    def posenc(self, x):
        T, d = x.shape[1], x.shape[2]; pe = torch.zeros(T, d, device=x.device)
        pos = torch.arange(T, device=x.device).float().unsqueeze(1)
        dv = torch.exp(torch.arange(0, d, 2, device=x.device).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * dv); pe[:, 1::2] = torch.cos(pos * dv)
        return x + pe.unsqueeze(0)

    def encode(self, ph):
        x = self.posenc(self.emb(ph))
        for b in self.enc: x = b(x)
        return x

    def decode(self, y_lr):
        y = self.posenc(y_lr)
        for b in self.dec: y = b(y)
        return self.out(y).transpose(1, 2)  # (B,80,Tf)

    def forward(self, ph, durations=None, max_frames=None):  # inference / GT-duration training
        x = self.encode(ph)
        log_dur = self.dp(x.detach())
        if durations is None:
            durations = torch.clamp(torch.round(torch.exp(log_dur) - 1), min=1).long()
            max_frames = max(int(durations.sum(1).max().item()), 1)
        return self.decode(length_regulate(x, durations, max_frames)), log_dur

    def forward_align(self, ph, mel, pl, ml, attn_prior=None, prior_weight=1.0):  # self-aligning training
        from sqzw.alignment import hard_durations
        x = self.encode(ph)                                  # (B, Tp, d)
        log_dur = self.dp(x.detach())
        Tp = ph.shape[1]
        key_mask = torch.arange(Tp, device=ph.device)[None, :] >= pl.to(ph.device)[:, None]   # True=pad
        soft, logp = self.aligner(mel, x.transpose(1, 2), key_mask, attn_prior, prior_weight)  # (B,1,T_mel,Tp)
        dur, hard = hard_durations(soft, pl.tolist(), ml.tolist())         # (B, Tp)
        mel_pred = self.decode(length_regulate(x, dur.long(), mel.shape[2]))
        return mel_pred, log_dur, dur, soft, hard, logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="fs_data/tyc/fs_manifest.json")
    ap.add_argument("--out", default="fs/out")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--learn-alignment", action="store_true", help="self-align (no precomputed durations)")
    ap.add_argument("--bin-weight", type=float, default=0.0, help="binarization-loss weight (0=off; it destabilised training)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.manifest)); n_sym = max(max(it["phoneme_ids"]) for it in items) + 1
    print(f"utts={len(items)} n_sym={n_sym} learn_alignment={a.learn_alignment}", flush=True)
    dl = DataLoader(FSSet(a.manifest), batch_size=a.batch_size, shuffle=True, collate_fn=collate, num_workers=0, drop_last=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniSpeech(n_sym, learn_alignment=a.learn_alignment).to(dev)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location="cpu")["model"], strict=False); print(f"resumed {a.resume}", flush=True)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M device={dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs) if a.cosine else None
    if a.learn_alignment:
        from sqzw.alignment import ForwardSumLoss, bin_loss, prior_batch
        fsum = ForwardSumLoss().to(dev)

    for ep in range(1, a.epochs + 1):
        model.train(); rm = rd = ra = 0.0; t0 = time.time()
        # forward-sum + diagonal prior drive the alignment; MAS gives hard durations.
        # (binarization loss destabilised it here, so it is off by default.)
        bin_w = a.bin_weight if a.learn_alignment else 0.0
        prior_w = max(0.0, 1.0 - ep / (a.epochs * 0.4))   # diagonal prior: strong early, off by 40%
        for ph, du, me, pl, ml in dl:
            ph, du, me = ph.to(dev), du.to(dev), me.to(dev)
            Tf = me.shape[2]
            fmask = (torch.arange(Tf, device=dev)[None, :] < ml.to(dev)[:, None]).unsqueeze(1)
            pmask = (torch.arange(ph.shape[1], device=dev)[None, :] < pl.to(dev)[:, None])
            if a.learn_alignment:
                prior = prior_batch(pl.tolist(), ml.tolist(), ph.shape[1], Tf).to(dev)
                mel, log_dur, dur, soft, hard, logp = model.forward_align(ph, me, pl, ml, prior, prior_w)
                lmel = (F.l1_loss(mel, me, reduction='none') * fmask).sum() / (fmask.sum() * 80)
                ldur = (F.mse_loss(log_dur, torch.log(dur.float() + 1), reduction='none') * pmask).sum() / pmask.sum()
                lfs = fsum(logp, pl.tolist(), ml.tolist())
                lbin = bin_loss(soft, hard)
                loss = lmel + 0.1 * ldur + lfs + bin_w * lbin
                ra += lfs.item()
            else:
                mel, log_dur = model(ph, du, Tf)
                lmel = (F.l1_loss(mel, me, reduction='none') * fmask).sum() / (fmask.sum() * 80)
                ldur = (F.mse_loss(log_dur, torch.log(du.float() + 1), reduction='none') * pmask).sum() / pmask.sum()
                loss = lmel + ldur
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            rm += lmel.item(); rd += ldur.item()
        if sched: sched.step()
        extra = f" fsum {ra/len(dl):.3f} bin_w {bin_w:.2f}" if a.learn_alignment else ""
        print(f"epoch {ep} mel_l1 {rm/len(dl):.4f} dur {rd/len(dl):.4f}{extra} lr {opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
        if ep % a.save_every == 0 or ep == a.epochs:
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith("aligner.")}  # aligner is training-only
            torch.save({"model": sd, "n_sym": n_sym, "epoch": ep}, f"{a.out}/fs_{ep}.pth")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
