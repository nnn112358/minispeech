#!/usr/bin/env python3
"""Plot a SqueezeWave NLL-pretrain loss curve from a log.
usage: plot_nll_loss.py [log] [title] [out.png]"""
import sys
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/pipeline_c64.log"
TITLE = sys.argv[2] if len(sys.argv) > 2 else "stage1 c64_f12_l8 NLL pretrain"
OUT = sys.argv[3] if len(sys.argv) > 3 else "outputs/nll_loss_c64.png"

PAT = re.compile(r"step (\d+)/(\d+) loss ([-\d.]+) ema ([-\d.]+)")
by = {}; total = 0
for line in open(LOG):
    m = PAT.match(line)
    if m:
        by[int(m.group(1))] = (float(m.group(3)), float(m.group(4))); total = int(m.group(2))
steps = sorted(by)
loss = [by[s][0] for s in steps]; ema = [by[s][1] for s in steps]

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(steps, loss, color="#55A868", lw=0.4, alpha=0.35, label="loss (raw)")
ax.plot(steps, ema, color="#1f7a3d", lw=1.8, label="loss (EMA)")
ax.axhline(-6.12, color="#4C72B0", ls=":", lw=1.4, label="a256_c128 NLL plateau (-6.12)")
ax.axhline(-5.92, color="#C44E52", ls=":", lw=1.4, label="c64_f12_l8 NLL plateau (-5.92)")
ax.set_xlabel("step"); ax.set_ylabel("flow NLL loss (lower=better)")
ax.set_title(f"{TITLE}  ({steps[-1] if steps else 0}/{total} steps)")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
if steps:
    ax.annotate(f"{ema[-1]:.3f} @ {steps[-1]}", (steps[-1], ema[-1]),
                textcoords="offset points", xytext=(-10, 10), fontsize=9, color="#1f7a3d", ha="right")
plt.tight_layout()
plt.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved {OUT}  (last {steps[-1] if steps else 0}: ema {ema[-1]:.3f})")
