#!/usr/bin/env python3
"""Overlay loss curves across all SqueezeWave trainings run so far.
Panel 1: NLL-pretrain loss (flow NLL).  Panels 2/3: aux fine-tune mel & STFT."""
import sys
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "checkpoints/"
NLL = re.compile(r"step (\d+)/\d+ loss ([-\d.]+) ema ([-\d.]+)")
AUX = re.compile(r"step (\d+)/\d+ nll ([-\d.]+) mel ([\d.]+) stft ([\d.]+)")


def parse(logs, pat, idx):
    by = {}
    for lg in logs:
        try:
            for line in open(D + lg):
                m = pat.match(line)
                if m:
                    by[int(m.group(1))] = float(m.group(idx))
        except FileNotFoundError:
            pass
    xs = sorted(by)
    return xs, [by[x] for x in xs]


# (label, logs, color)
NLL_RUNS = [
    ("a128_c128 (7.2M)",       ["train_a128c128_old.log"], "#999999"),
    ("a256_c128 (7.9M)",       ["train.log"],              "#4C72B0"),
    ("c64_f12_l8 (3.1M)",      ["pipeline_c64.log"],       "#C44E52"),
    ("c64_f8_l8+dil (2.1M)",   ["frontier.log"],           "#55A868"),
]
AUX_RUNS = [
    ("a256_c128",            ["train_aux.log", "train_aux_r2.log"], "#4C72B0"),
    ("c64_f12_l8",           ["pipeline_c64.log"],                  "#C44E52"),
    ("c64_f8_l8+dil",        ["frontier_aux.log"],                  "#55A868"),
]

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5))

# Panel 1: NLL
for label, logs, c in NLL_RUNS:
    xs, ys = parse(logs, NLL, 3)  # ema
    if xs:
        a1.plot(xs, ys, color=c, lw=1.5, label=f"{label}  →{ys[-1]:.2f}")
a1.set_title("NLL pretrain (flow NLL, lower=better)"); a1.set_xlabel("step"); a1.set_ylabel("NLL loss")
a1.grid(alpha=0.3); a1.legend(fontsize=8)

# Panel 2: aux mel
for label, logs, c in AUX_RUNS:
    xs, ys = parse(logs, AUX, 3)  # mel
    if xs:
        a2.plot(xs, ys, color=c, lw=1.5, label=f"{label}  →{ys[-1]:.3f}")
a2.axhline(0.216, color="#333", ls=":", lw=1.4, label="Vocos (0.216)")
a2.set_title("aux fine-tune: mel-L1 loss"); a2.set_xlabel("step"); a2.set_ylabel("mel-L1")
a2.grid(alpha=0.3); a2.legend(fontsize=8); a2.set_ylim(0, 0.8)

# Panel 3: aux stft
for label, logs, c in AUX_RUNS:
    xs, ys = parse(logs, AUX, 4)  # stft
    if xs:
        a3.plot(xs, ys, color=c, lw=1.5, label=f"{label}  →{ys[-1]:.3f}")
a3.axhline(0.92, color="#333", ls=":", lw=1.4, label="Vocos (0.92)")
a3.set_title("aux fine-tune: multi-res STFT loss"); a3.set_xlabel("step"); a3.set_ylabel("SC+LogMag")
a3.grid(alpha=0.3); a3.legend(fontsize=8); a3.set_ylim(0.8, 2.2)

fig.suptitle("SqueezeWave trainings overlay — NLL pretrain & aux fine-tune (mel / STFT)", fontsize=13)
plt.tight_layout()
out = "outputs/loss_overlay.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")
for label, logs, c in NLL_RUNS:
    xs, ys = parse(logs, NLL, 3)
    if xs: print(f"NLL {label:24} last step {xs[-1]:>6}  ema {ys[-1]:.3f}")
for label, logs, c in AUX_RUNS:
    xs, ym = parse(logs, AUX, 3); _, ys = parse(logs, AUX, 4)
    if xs: print(f"aux {label:24} last step {xs[-1]:>6}  mel {ym[-1]:.3f}  stft {ys[-1]:.3f}")
