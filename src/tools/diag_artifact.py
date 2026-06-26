#!/usr/bin/env python3
"""Pin down the deterministic artifact: (a) HF flatness incl. Vocos, (b) zoomed
voiced-segment waveform (group-boundary discontinuity?), (c) long-term average
spectrum comb at the 86 Hz group/frame rate."""
import sys
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, librosa, soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decoders.squeezewave.model import SqueezeWave
from common.features import PiperMelFeatures

SR = 22050; GROUP = 256; FRATE = SR / GROUP  # 86.1 Hz
VAL = [l for l in open("data/filelist_val.txt").read().splitlines() if l.strip()]


def flat(x):
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)); fr = librosa.fft_frequencies(sr=SR, n_fft=1024)
    S = S[fr >= 4000] + 1e-8
    return float(np.mean(np.exp(np.mean(np.log(S), 0)) / np.mean(S, 0)))


def load_model(ckpt):
    ck = torch.load(ckpt, map_location="cpu"); m = SqueezeWave(**ck["config"])
    m.load_state_dict(ck["model"]); return SqueezeWave.remove_weightnorm(m).eval()


feat = PiperMelFeatures(22050, 1024, 256, 1024, 80)
y, _ = sf.read(VAL[0], dtype="float32")
if y.ndim > 1: y = y[:, 0]
mel = feat(torch.from_numpy(y).unsqueeze(0))
m = load_model("_unused/m3_sqzw_ckpts/ckpts_aux/sqzwaux_step40000_bnrecal.pth")
with torch.no_grad():
    gen = m.infer(mel, sigma=0.7).squeeze(0).numpy()
vo, _ = sf.read("outputs/vocos_00.wav", dtype="float32")
L = min(len(y), len(gen), len(vo)); y, gen, vo = y[:L], gen[:L], vo[:L]

print("=== HF(>4kHz)平坦度 (GTに近いほど良い) ===")
for k, v in [("GT", y), ("Vocos", vo), ("a256_c128", gen)]:
    print(f"  {k:12} {flat(v):.3f}")

# find a strongly voiced 40ms window (high energy)
w = int(0.04 * SR)
energies = [np.sum(y[i:i+w]**2) for i in range(0, L-w, w)]
i0 = int(np.argmax(energies)) * w

fig, ax = plt.subplots(3, 1, figsize=(12, 9))
t = np.arange(w) / SR * 1000
ax[0].plot(t, y[i0:i0+w], label="GT", lw=1.0, color="k")
ax[0].plot(t, gen[i0:i0+w], label="a256_c128", lw=1.0, color="#C44E52", alpha=0.8)
for gb in range(0, w, GROUP):            # mark 256-sample group boundaries
    ax[0].axvline(gb/SR*1000, color="#4C72B0", ls=":", lw=0.6, alpha=0.6)
ax[0].set_title("voiced 40ms waveform (dotted=256-sample group boundaries / 86Hz)"); ax[0].legend(); ax[0].set_xlabel("ms")

# long-term average spectrum
for name, v, c in [("GT", y, "k"), ("a256_c128", gen, "#C44E52")]:
    S = np.abs(librosa.stft(v, n_fft=4096, hop_length=1024)).mean(1)
    fr = librosa.fft_frequencies(sr=SR, n_fft=4096)
    ax[1].plot(fr, 20*np.log10(S+1e-8), label=name, lw=0.8, color=c, alpha=0.85)
for k in range(1, 12):                   # frame-rate comb 86,172,... Hz
    ax[1].axvline(k*FRATE, color="#4C72B0", ls=":", lw=0.5, alpha=0.5)
ax[1].set_xlim(0, 2000); ax[1].set_title("long-term avg spectrum (dotted=k·86Hz frame-rate comb)"); ax[1].legend(); ax[1].set_xlabel("Hz")

# error spectrum (gen-GT not phase aligned; show magnitude diff per freq)
Sg = np.abs(librosa.stft(gen, n_fft=1024, hop_length=256)).mean(1)
Sy = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)).mean(1)
fr = librosa.fft_frequencies(sr=SR, n_fft=1024)
ax[2].plot(fr, 20*np.log10(Sg+1e-8) - 20*np.log10(Sy+1e-8), color="#C44E52", lw=0.9)
ax[2].axhline(0, color="k", lw=0.5); ax[2].set_xlim(0, 11025); ax[2].set_ylim(-12, 12)
ax[2].set_title("a256 - GT magnitude (dB) per frequency"); ax[2].set_xlabel("Hz")
plt.tight_layout(); plt.savefig("outputs/artifact_diag.png", dpi=110, bbox_inches="tight")
print("saved listen/artifact_diag.png  (voiced window at %.2fs)" % (i0/SR))
