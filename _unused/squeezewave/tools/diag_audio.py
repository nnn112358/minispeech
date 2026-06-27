#!/usr/bin/env python3
"""Diagnose audible distortion ("割れ"). Checks (1) clipping in written wavs,
(2) RAW (unclipped) model-output amplitude vs GT/Vocos: peak, clip fraction,
RMS, crest factor, and outlier spikes. Pinpoints hard-clip vs other causes."""
import sys, os, glob
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import soundfile as sf
from decoders.squeezewave.model import SqueezeWave
from common.features import PiperMelFeatures

SR = 22050
VAL = [l for l in open("data/filelist_val.txt").read().splitlines() if l.strip()]


def stats(x, name):
    x = x.astype(np.float64)
    peak = np.max(np.abs(x)); rms = np.sqrt(np.mean(x**2))
    clip = np.mean(np.abs(x) >= 0.999) * 100
    near = np.mean(np.abs(x) >= 0.99) * 100
    crest = 20*np.log10(peak/(rms+1e-9))
    # spikes: samples > 4x the 99.9th percentile (impulsive glitches)
    p999 = np.percentile(np.abs(x), 99.9)
    spikes = np.sum(np.abs(x) > max(1.0, 4*p999))
    print(f"  {name:22} peak {peak:6.3f}  clip%>={0.999}: {clip:5.2f}  >=0.99: {near:5.2f}  "
          f"rms {rms:.3f}  crest {crest:4.1f}dB  spikes>1&>4·p99.9: {spikes}")


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    m = SqueezeWave(**ck["config"]).to(dev); m.load_state_dict(ck["model"])
    m = SqueezeWave.remove_weightnorm(m); m.eval(); return m


def main():
    dev = torch.device("cpu")
    feat = PiperMelFeatures(22050, 1024, 256, 1024, 80).to(dev)

    print("=== (1) 書き出し済みwavのクリップ実測 ===")
    for f in ["samp_a256_00.wav", "samp_c64f12_00.wav", "samp_a256_00_gt.wav", "vocos_00.wav"]:
        p = f"outputs/{f}"
        if os.path.exists(p):
            y, _ = sf.read(p, dtype="float32"); stats(y, f)

    print("\n=== (2) 生モデル出力(クリップ前)の振幅 — 同一発話04950, sigma sweep ===")
    f0 = VAL[0]; y, _ = sf.read(f0, dtype="float32")
    if y.ndim > 1: y = y[:, 0]
    stats(y, "GT(04950)")
    x = torch.from_numpy(y).unsqueeze(0).to(dev)
    mel = feat(x)
    for tag, ckpt in [("a256_c128", "_unused/m3_sqzw_ckpts/ckpts_aux/sqzwaux_step40000_bnrecal.pth"),
                      ("c64_f12_l8", "_unused/m3_sqzw_ckpts/ckpts_c64_f12_l8_aux/sqzwaux_bnrecal.pth")]:
        m = load_model(ckpt, dev)
        for sigma in [0.5, 0.7, 1.0]:
            with torch.no_grad():
                out = m.infer(mel, sigma=sigma).squeeze(0).cpu().numpy()
            stats(out, f"{tag} sigma{sigma}")


if __name__ == "__main__":
    main()
