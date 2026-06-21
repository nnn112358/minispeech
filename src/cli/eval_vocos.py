#!/usr/bin/env python3
"""Vocos copy-synthesis baseline eval, matched to cli/infer.py so the two
are directly comparable: same val files, same mel-L1 reconstruction metric, and
mel->audio timing only (mel extraction excluded, as it comes from FastSpeech)."""
import sys, os, time, argparse
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import soundfile as sf
from sqzw.vocos_gen import Generator
from sqzw.features import PiperMelFeatures

SR = 22050


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/ckpts/vocos_last.pth")
    ap.add_argument("--filelist", default="data/filelist_val.txt")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="vocos")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = PiperMelFeatures(sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80).to(dev)
    G = Generator().to(dev)
    ck = torch.load(a.ckpt, map_location=dev)
    G.load_state_dict(ck["G"]); G.eval()
    print(f"loaded {a.ckpt} @ step {ck.get('step','?')}  device={dev}", flush=True)

    files = [l for l in open(a.filelist).read().splitlines() if l.strip()][:a.n]
    tot_syn = 0.0; tot_aud = 0.0; l1s = []
    for i, f in enumerate(files):
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1: y = y[:, 0]
        x = torch.from_numpy(y).unsqueeze(0).to(dev)
        with torch.no_grad():
            mel = feat(x)
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = G.head(G.backbone(mel))                 # mel -> audio only
            if dev.type == "cuda": torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            a_np = audio.squeeze(0).cpu().numpy()
            mel2 = feat(torch.from_numpy(a_np).unsqueeze(0).to(dev))
            L = min(mel.size(2), mel2.size(2))
            l1 = (mel[:, :, :L] - mel2[:, :, :L]).abs().mean().item()
        adur = len(a_np) / SR
        tot_syn += dt; tot_aud += adur; l1s.append(l1)
        sf.write(f"{a.out}/{a.tag}_{i:02d}.wav", np.clip(a_np, -1, 1), SR)
        print(f"  [{i}] {os.path.basename(f)} aud {adur:.2f}s syn {dt*1000:.1f}ms "
              f"RTF {dt/adur:.4f} melL1 {l1:.3f}", flush=True)
    print(f"\nsummary: {len(files)} files  total syn {tot_syn*1000:.0f}ms / aud {tot_aud:.2f}s "
          f"=> RTF {tot_syn/tot_aud:.4f} ({tot_aud/tot_syn:.0f}x RT)  mean melL1 {np.mean(l1s):.3f}", flush=True)


if __name__ == "__main__":
    main()
