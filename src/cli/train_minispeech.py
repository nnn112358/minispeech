#!/usr/bin/env python3
"""MiniSpeechEncoder trainer. Two alignment modes:
  --learn-alignment : SELF-ALIGNING (recommended). No external aligner needed.
  (otherwise)       : use precomputed {durations} in the manifest."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, argparse, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from encoder.minispeech import MiniSpeechEncoder, FSSet, collate


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
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--n-enc", type=int, default=4)
    ap.add_argument("--n-dec", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.manifest)); n_sym = max(max(it["phoneme_ids"]) for it in items) + 1
    print(f"utts={len(items)} n_sym={n_sym} learn_alignment={a.learn_alignment}", flush=True)
    dl = DataLoader(FSSet(a.manifest), batch_size=a.batch_size, shuffle=True, collate_fn=collate, num_workers=0, drop_last=True, pin_memory=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniSpeechEncoder(n_sym, d=a.dim, n_enc=a.n_enc, n_dec=a.n_dec, learn_alignment=a.learn_alignment).to(dev)
    if a.resume:
        sd = {k.replace("dp.net.3.", "dp.net.2."): v for k, v in torch.load(a.resume, map_location="cpu")["model"].items()}
        model.load_state_dict(sd, strict=False); print(f"resumed {a.resume}", flush=True)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M device={dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs) if a.cosine else None
    if a.learn_alignment:
        from encoder.alignment import ForwardSumLoss, bin_loss, prior_batch
        fsum = ForwardSumLoss().to(dev)

    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for ep in range(1, a.epochs + 1):
        model.train(); rm = rd = ra = 0.0; t0 = time.time()
        bin_w = a.bin_weight if a.learn_alignment else 0.0
        prior_w = max(0.0, 1.0 - ep / (a.epochs * 0.4))
        for ph, du, me, pl, ml in dl:
            ph, du, me = ph.to(dev, non_blocking=True), du.to(dev, non_blocking=True), me.to(dev, non_blocking=True)
            Tf = me.shape[2]
            fmask = (torch.arange(Tf, device=dev)[None, :] < ml.to(dev)[:, None]).unsqueeze(1)
            pmask = (torch.arange(ph.shape[1], device=dev)[None, :] < pl.to(dev)[:, None])
            with torch.amp.autocast("cuda", enabled=use_amp):
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
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            rm += lmel.item(); rd += ldur.item()
        if sched: sched.step()
        extra = f" fsum {ra/len(dl):.3f} bin_w {bin_w:.2f}" if a.learn_alignment else ""
        print(f"epoch {ep} mel_l1 {rm/len(dl):.4f} dur {rd/len(dl):.4f}{extra} lr {opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
        if ep % a.save_every == 0 or ep == a.epochs:
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith("aligner.")}
            torch.save({"model": sd, "n_sym": n_sym, "epoch": ep,
                        "config": {"dim": a.dim, "n_enc": a.n_enc, "n_dec": a.n_dec}}, f"{a.out}/fs_{ep}.pth")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
