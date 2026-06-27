#!/usr/bin/env python3
"""Auxiliary-loss fine-tune for SqueezeWave: add multi-resolution STFT loss +
mel reconstruction loss on top of the flow NLL. The aux losses are PAIRED — we
generate audio conditioned on the GT mel (differentiable reverse flow) and match
it to that same GT mel / GT waveform, exactly the Vocos mel-loss principle.
This directly optimises spectral fidelity, which pure flow-NLL does not.
Init from a converged NLL checkpoint."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from decoders.squeezewave.model import SqueezeWave, SqueezeWaveLoss
from common.features import PiperMelFeatures
from common.dataset import WavSet
from decoders.squeezewave.flow import diff_infer, mrstft_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", default="_unused/m3_sqzw_ckpts/ckpts/sqzw_step200000.pth")
    ap.add_argument("--filelist", default="data/filelist_train.txt")
    ap.add_argument("--out", default="checkpoints/ckpts_aux")
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--w-nll", type=float, default=1.0)
    ap.add_argument("--w-mel", type=float, default=5.0)
    ap.add_argument("--w-stft", type=float, default=2.0)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--resume", default="")   # continue an aux run (restores model+opt+step)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat = PiperMelFeatures().to(dev)
    ck = torch.load(a.init_from, map_location=dev)
    cfg = ck["config"]
    model = SqueezeWave(**cfg).to(dev)
    model.load_state_dict(ck["model"])
    criterion = SqueezeWaveLoss(a.sigma)
    print(f"init from {a.init_from} (step {ck['step']})  cfg {cfg['n_audio_channel']}/"
          f"{cfg['WN_config']['n_channels']}  w_nll={a.w_nll} w_mel={a.w_mel} w_stft={a.w_stft}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    step0 = 0
    if a.resume and os.path.isfile(a.resume):
        rk = torch.load(a.resume, map_location=dev)
        model.load_state_dict(rk["model"]); opt.load_state_dict(rk["opt"]); step0 = rk["step"]
        print(f"resumed {a.resume} @ step {step0}", flush=True)
    dl = DataLoader(WavSet(a.filelist, a.num_samples), batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, persistent_workers=True, pin_memory=True)
    step = step0; t0 = time.time(); model.train()
    rn = rm = rs = None
    while step < a.max_steps:
        for audio in dl:
            audio = audio.to(dev, non_blocking=True)
            with torch.no_grad():
                mel = feat(audio)
            # flow NLL
            out = model((mel, audio)); loss_nll = criterion(out)
            # paired aux: generate from GT mel, match to GT mel / GT wav
            gen = diff_infer(model, mel, a.sigma)
            mel_gen = feat(gen)
            Lm = min(mel.size(2), mel_gen.size(2))
            loss_mel = (mel[:, :, :Lm] - mel_gen[:, :, :Lm]).abs().mean()
            sc, lm = mrstft_loss(audio, gen, dev)
            loss_stft = sc + lm
            loss = a.w_nll * loss_nll + a.w_mel * loss_mel + a.w_stft * loss_stft
            if not torch.isfinite(loss):
                print(f"  WARN non-finite @ {step}, skip", flush=True); opt.zero_grad(); step += 1; continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            opt.step()
            rn = loss_nll.item() if rn is None else 0.99*rn+0.01*loss_nll.item()
            rm = loss_mel.item() if rm is None else 0.99*rm+0.01*loss_mel.item()
            rs = loss_stft.item() if rs is None else 0.99*rs+0.01*loss_stft.item()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{a.max_steps} nll {rn:.3f} mel {rm:.3f} stft {rs:.3f} "
                      f"{step/(time.time()-t0):.1f}it/s {time.time()-t0:.0f}s", flush=True)
            if step % a.save_every == 0 or step == a.max_steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "config": cfg}, f"{a.out}/sqzwaux_step{step}.pth")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "config": cfg}, f"{a.out}/sqzwaux_last.pth")
                print(f"  saved {step} (mel {rm:.3f} stft {rs:.3f})", flush=True)
            if step >= a.max_steps: break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
