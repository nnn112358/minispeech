#!/usr/bin/env python3
"""Spectral diagnosis of the distortion: compare GT vs a256 vs c64f12.
Figure of log-magnitude spectrograms + difference, and per-band energy ratio &
spectral flatness (noisiness). Reveals whether '割れ' is HF noise, phase
roughness, or a periodic (buzz) artifact."""
import sys, os
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, librosa
import soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decoders.squeezewave.model import SqueezeWave
from common.features import PiperMelFeatures

SR = 22050
VAL = [l for l in open("data/filelist_val.txt").read().splitlines() if l.strip()]
BANDS = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11025)]


def load_model(ckpt):
    ck = torch.load(ckpt, map_location="cpu")
    m = SqueezeWave(**ck["config"]); m.load_state_dict(ck["model"])
    return SqueezeWave.remove_weightnorm(m).eval()


def band_energy(x, n_fft=1024, hop=256):
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    out = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out.append(S[m].mean())
    return np.array(out)


def flatness_hf(x, n_fft=1024, hop=256):
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    m = freqs >= 4000
    Shf = S[m] + 1e-8
    gm = np.exp(np.mean(np.log(Shf), axis=0)); am = np.mean(Shf, axis=0)
    return np.mean(gm / am)   # 1=white noise, 0=tonal


def main():
    feat = PiperMelFeatures(22050, 1024, 256, 1024, 80)
    y, _ = sf.read(VAL[0], dtype="float32")
    if y.ndim > 1: y = y[:, 0]
    mel = feat(torch.from_numpy(y).unsqueeze(0))
    sigs = {"GT": y}
    for tag, ckpt in [("a256_c128", "_unused/m3_sqzw_ckpts/ckpts_aux/sqzwaux_step40000_bnrecal.pth"),
                      ("c64_f12_l8", "_unused/m3_sqzw_ckpts/ckpts_c64_f12_l8_aux/sqzwaux_bnrecal.pth")]:
        m = load_model(ckpt)
        with torch.no_grad():
            sigs[tag] = m.infer(mel, sigma=0.7).squeeze(0).numpy()

    L = min(len(s) for s in sigs.values())
    sigs = {k: v[:L] for k, v in sigs.items()}
    egt = band_energy(sigs["GT"])
    print("=== 帯域エネルギー比 (gen/GT, dB) — +は過剰, -は不足 ===")
    print(f"{'band(Hz)':>12} " + " ".join(f"{k:>12}" for k in sigs))
    for i, (lo, hi) in enumerate(BANDS):
        row = []
        for k in sigs:
            e = band_energy(sigs[k])[i]
            row.append(f"{10*np.log10(e/egt[i]+1e-12):+6.1f}dB" if k != "GT" else f"{'(ref)':>9}")
        print(f"{lo:5}-{hi:<5} " + " ".join(f"{r:>12}" for r in row))
    print("\n=== HF(>4kHz)スペクトル平坦度 (1=雑音的, 0=調波的) ===")
    for k in sigs:
        print(f"  {k:12} {flatness_hf(sigs[k]):.3f}")

    # ---- figure: spectrograms ----
    fig, axes = plt.subplots(len(sigs), 1, figsize=(12, 9), sharex=True)
    for ax, (k, v) in zip(axes, sigs.items()):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(v, n_fft=1024, hop_length=256)), ref=np.max)
        img = librosa.display.specshow(D, sr=SR, hop_length=256, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set_title(f"{k}", fontsize=10); ax.set_ylim(0, 11025)
    fig.suptitle("Spectrogram: GT vs a256_c128 vs c64_f12_l8 (copy-synth, sigma0.7)", fontsize=12)
    plt.tight_layout()
    out = "outputs/spectral_diag.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
