#!/usr/bin/env python3
"""Standalone SqueezeWave vocoder trainer for the JSUT 2-stage TTS.
Trains the (patched) SqueezeWave flow on the SAME 80-dim log-mel the MiniSpeech
acoustic model emits (PiperMelFeatures: sr22050/n_fft1024/hop256/win1024,
center=False, log-clamp 1e-5) so the FS->SqueezeWave handoff is exact.
Pure max-likelihood flow loss (no GAN). fp32 for flow stability."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from decoders.squeezewave.model import SqueezeWave, SqueezeWaveLoss
from common.features import PiperMelFeatures
from common.dataset import WavSet

# Preset (n_audio_channel, WN n_channels) shorthands. a256=L64 (l=256, ~1.7x
# faster than a128/l=512); c=WN width. Every arch param is also a CLI flag below,
# so any (nac, nc, n_flows, n_layers, n_early_every, n_early_size) combo is selectable.
CONFIGS = {
    "a128_c128": (128, 128), "a128_c256": (128, 256),
    "a256_c128": (256, 128), "a256_c256": (256, 256),
    "a256_c64": (256, 64),   "a128_c64": (128, 64),
    "a256_c96": (256, 96),   "a256_c48": (256, 48),
}


def make_config(n_audio_channel, n_channels, n_flows, n_layers, n_early_every, n_early_size,
                dilation_cycle=None):
    wn = {"n_layers": n_layers, "n_channels": n_channels, "kernel_size": 3}
    if dilation_cycle:
        wn["dilation_cycle"] = dilation_cycle   # WaveNet 2^(i%K) receptive field, same params
    return {"n_mel_channels": 80, "n_flows": n_flows, "n_audio_channel": n_audio_channel,
            "n_early_every": n_early_every, "n_early_size": n_early_size, "WN_config": wn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="data/filelist_train.txt")
    ap.add_argument("--out", default="checkpoints/ckpts")
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-samples", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=2000)
    # --- architecture: every param individually selectable; --config is an
    #     optional preset that only fills n_audio_channel & n_channels ---
    ap.add_argument("--config", default="", help="optional preset for (nac,nc): " + ",".join(CONFIGS))
    ap.add_argument("--n-audio-channel", type=int, default=256)
    ap.add_argument("--n-channels", type=int, default=128)
    ap.add_argument("--n-flows", type=int, default=12)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-early-every", type=int, default=2)
    ap.add_argument("--n-early-size", type=int, default=16)
    ap.add_argument("--dilation-cycle", type=int, default=0, help="0=off (dilation 1); K=WaveNet 2^(i%K)")
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nac, nc = a.n_audio_channel, a.n_channels
    if a.config:
        nac, nc = CONFIGS[a.config]   # preset overrides nac/nc only
    SQZW_CONFIG = make_config(nac, nc, a.n_flows, a.n_layers, a.n_early_every, a.n_early_size,
                              a.dilation_cycle or None)

    feat = PiperMelFeatures().to(dev)
    model = SqueezeWave(**SQZW_CONFIG).to(dev)
    criterion = SqueezeWaveLoss(a.sigma)
    npar = sum(p.numel() for p in model.parameters())
    print(f"SqueezeWave[nac{nac}_c{nc}_f{a.n_flows}_l{a.n_layers}] params={npar/1e6:.2f}M "
          f"device={dev} batch={a.batch_size}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    step = 0
    if a.resume and os.path.isfile(a.resume):
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print(f"resumed {a.resume} @ step {step}", flush=True)

    dl = DataLoader(WavSet(a.filelist, a.num_samples), batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, persistent_workers=True, pin_memory=True)
    t0 = time.time(); run = None; model.train()
    while step < a.max_steps:
        for audio in dl:
            audio = audio.to(dev, non_blocking=True)      # (B, T)
            with torch.no_grad():
                mel = feat(audio)                       # (B, 80, frames)
            out = model((mel, audio))
            loss = criterion(out)
            if not torch.isfinite(loss):
                print(f"  WARN non-finite loss at step {step}, skipping", flush=True)
                opt.zero_grad(); step += 1; continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
            opt.step()
            lv = loss.item(); run = lv if run is None else 0.99 * run + 0.01 * lv
            step += 1
            if step % 100 == 0:
                sps = step / (time.time() - t0)
                print(f"step {step}/{a.max_steps} loss {lv:.4f} ema {run:.4f} {sps:.1f}it/s {time.time()-t0:.0f}s", flush=True)
            if step % a.save_every == 0 or step == a.max_steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "config": SQZW_CONFIG}, f"{a.out}/sqzw_step{step}.pth")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "config": SQZW_CONFIG}, f"{a.out}/sqzw_last.pth")
                print(f"  saved step {step} (ema {run:.4f})", flush=True)
            if step >= a.max_steps: break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
