#!/usr/bin/env python3
"""Pre-compute PiperMelFeatures for all files in a filelist."""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, soundfile as sf
from common.features import PiperMelFeatures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="data/filelist_train.txt")
    ap.add_argument("--out-dir", default="data/mel_cache")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    feat = PiperMelFeatures().eval()
    files = [l for l in open(a.filelist).read().splitlines() if l.strip()]
    for i, f in enumerate(files):
        basename = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(a.out_dir, basename + ".npy")
        if os.path.exists(out_path):
            continue
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1:
            y = y[:, 0]
        with torch.no_grad():
            mel = feat(torch.from_numpy(y).unsqueeze(0)).squeeze(0).numpy()
        np.save(out_path, mel)
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(files)}", flush=True)
    print(f"Done: {len(files)} files -> {a.out_dir}")

if __name__ == "__main__":
    main()
