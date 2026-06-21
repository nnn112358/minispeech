#!/usr/bin/env python3
"""SqueezeWave vocoder inference / validation for the JSUT 2-stage TTS.
(a) copy-synthesis: val wav -> PiperMelFeatures mel -> SqueezeWave.infer -> wav
    (isolates vocoder fidelity), or
(b) from a precomputed FS mel .npy (true end-to-end check).
Reports synthesis RTF and a mel-L1 reconstruction error vs the input mel."""
import sys, os, time, argparse, glob
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import soundfile as sf
from sqzw.model import SqueezeWave
from sqzw.features import PiperMelFeatures

SR = 22050


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    cfg = ck["config"]
    model = SqueezeWave(**cfg).to(dev)
    model.load_state_dict(ck["model"])
    model = SqueezeWave.remove_weightnorm(model)
    model.eval()
    return model, ck["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpts_a256_c128_q_gan/sqzwgan_bnrecal.pth")
    ap.add_argument("--filelist", default="data/filelist_val.txt")
    ap.add_argument("--mel-npy", default="")           # optional: use a precomputed FS mel instead
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="sqzw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = PiperMelFeatures(sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80).to(dev)
    model, step = load_model(a.ckpt, dev)
    print(f"loaded {a.ckpt} @ step {step}  sigma={a.sigma}  device={dev}", flush=True)

    if a.mel_npy:
        mel = torch.from_numpy(np.load(a.mel_npy)).float().unsqueeze(0).to(dev)
        with torch.no_grad():
            audio = model.infer(mel, sigma=a.sigma).squeeze(0).cpu().numpy()
        op = f"{a.out}/{a.tag}_melnpy.wav"
        sf.write(op, np.clip(audio, -1, 1), SR)
        print(f"  {os.path.basename(a.mel_npy)} -> {op}  ({len(audio)/SR:.2f}s)", flush=True)
        return

    files = [l for l in open(a.filelist).read().splitlines() if l.strip()][:a.n]
    tot_syn = 0.0; tot_aud = 0.0; l1s = []
    for i, f in enumerate(files):
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1: y = y[:, 0]
        x = torch.from_numpy(y).unsqueeze(0).to(dev)
        with torch.no_grad():
            mel = feat(x)                                  # (1,80,T)
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = model.infer(mel, sigma=a.sigma)        # (1, T*256)
            if dev.type == "cuda": torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            a_np = audio.squeeze(0).cpu().numpy()
            # mel-L1 reconstruction error (re-extract mel from output)
            mel2 = feat(torch.from_numpy(a_np).unsqueeze(0).to(dev))
            L = min(mel.size(2), mel2.size(2))
            l1 = (mel[:, :, :L] - mel2[:, :, :L]).abs().mean().item()
        adur = len(a_np) / SR
        tot_syn += dt; tot_aud += adur; l1s.append(l1)
        op = f"{a.out}/{a.tag}_{i:02d}.wav"
        sf.write(op, np.clip(a_np, -1, 1), SR)
        gp = f"{a.out}/{a.tag}_{i:02d}_gt.wav"
        sf.write(gp, y, SR)
        print(f"  [{i}] {os.path.basename(f)} aud {adur:.2f}s syn {dt*1000:.1f}ms "
              f"RTF {dt/adur:.4f} melL1 {l1:.3f} -> {os.path.basename(op)}", flush=True)
    print(f"\nsummary: {len(files)} files  total syn {tot_syn*1000:.0f}ms / aud {tot_aud:.2f}s "
          f"=> RTF {tot_syn/tot_aud:.4f} ({tot_aud/tot_syn:.0f}x RT)  mean melL1 {np.mean(l1s):.3f}", flush=True)


if __name__ == "__main__":
    main()
