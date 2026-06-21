#!/usr/bin/env python3
"""Plot SqueezeWave aux fine-tune loss curves (mel, multi-res STFT) vs step,
with pure-NLL start and Vocos targets as reference lines."""
import sys
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "checkpoints/train_aux.log"
PAT = re.compile(r"step (\d+)/\d+ nll ([-\d.]+) mel ([\d.]+) stft ([\d.]+)")
# dedup by step (last occurrence wins) so a resumed run with repeated steps plots cleanly
byline = {}
for log in [LOG, LOG.replace("train_aux.log", "train_aux_r2.log")]:
    try:
        for line in open(log):
            m = PAT.match(line)
            if m:
                byline[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    except FileNotFoundError:
        pass
steps = sorted(byline)
nll = [byline[s][0] for s in steps]
mel = [byline[s][1] for s in steps]
stft = [byline[s][2] for s in steps]

# reference levels (eval metrics, which match training-loss scale at init)
VOCOS_MEL, VOCOS_STFT = 0.216, 0.92      # Vocos eval: melL1, SpecConv+LogMag
NLL_MEL, NLL_STFT = 0.739, 1.51          # pure-NLL 200k eval

fig, (axм, axs) = plt.subplots(1, 2, figsize=(13, 5))

axм.plot(steps, mel, color="#C44E52", lw=1.6, label="aux fine-tune (mel loss EMA)")
axм.axhline(NLL_MEL, color="#888", ls="--", lw=1.2, label=f"pure-NLL start ({NLL_MEL})")
axм.axhline(VOCOS_MEL, color="#4C72B0", ls=":", lw=1.8, label=f"Vocos target ({VOCOS_MEL})")
axм.set_xlabel("step"); axм.set_ylabel("mel L1 loss"); axм.set_title("Mel reconstruction loss")
axм.grid(alpha=0.3); axм.legend(fontsize=9); axм.set_ylim(0, max(mel) * 1.05)
if steps: axм.annotate(f"{mel[-1]:.3f} @ {steps[-1]}", (steps[-1], mel[-1]),
                       textcoords="offset points", xytext=(-10, 8), fontsize=9, color="#C44E52", ha="right")

axs.plot(steps, stft, color="#55A868", lw=1.6, label="aux fine-tune (STFT loss EMA)")
axs.axhline(NLL_STFT, color="#888", ls="--", lw=1.2, label=f"pure-NLL start ({NLL_STFT})")
axs.axhline(VOCOS_STFT, color="#4C72B0", ls=":", lw=1.8, label=f"Vocos target ({VOCOS_STFT})")
axs.set_xlabel("step"); axs.set_ylabel("SpecConv + LogMag-L1"); axs.set_title("Multi-resolution STFT loss")
axs.grid(alpha=0.3); axs.legend(fontsize=9); axs.set_ylim(0, max(stft) * 1.05)
if steps: axs.annotate(f"{stft[-1]:.3f} @ {steps[-1]}", (steps[-1], stft[-1]),
                       textcoords="offset points", xytext=(-10, 8), fontsize=9, color="#55A868", ha="right")

fig.suptitle(f"SqueezeWave a256_c128 aux fine-tune (mel+STFT) — {steps[-1] if steps else 0} steps, "
             f"toward Vocos quality", fontsize=12)
plt.tight_layout()
out = "outputs/aux_loss_curve.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}  (last step {steps[-1] if steps else 0}: mel {mel[-1]:.3f} stft {stft[-1]:.3f})")
