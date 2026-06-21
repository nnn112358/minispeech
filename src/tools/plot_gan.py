#!/usr/bin/env python3
"""Plot the a256_c128 GAN-finish losses: corrected (w_mel=45) vs failed (w_mel=5).
Left: mel reconstruction (anchored vs drifting = the fix). Right: corrected GAN
components (d/adv/fm/mel/stft)."""
import sys
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

jp = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
PAT = re.compile(r"step (\d+)/\d+ d ([-\d.]+) adv ([-\d.]+) fm ([-\d.]+) mel ([\d.]+) stft ([\d.]+)")


def parse(path, a256_only=False):
    """Return dict step->(d,adv,fm,mel,stft). If a256_only, stop at c64 marker."""
    by = {}
    try:
        for line in open(path):
            if a256_only and "GAN FINISH c64_f8_l4_s" in line:
                break
            m = PAT.match(line)
            if m:
                by[int(m.group(1))] = tuple(float(m.group(i)) for i in range(2, 7))
    except FileNotFoundError:
        pass
    xs = sorted(by)
    return xs, {k: [by[s][i] for s in xs] for i, k in enumerate(["d", "adv", "fm", "mel", "stft"])}

xf, F = parse("checkpoints/gan_finish.log", a256_only=True)   # failed
xc, C = parse("checkpoints/gan_finish2.log")                  # corrected

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel 1: mel anchor comparison
if xf: a1.plot(xf, F["mel"], color="#C44E52", lw=2, label=f"失敗版 w_mel=5 (→{F['mel'][-1]:.3f} ドリフト)")
if xc: a1.plot(xc, C["mel"], color="#2c7", lw=2, label=f"修正版 w_mel=45 (→{C['mel'][-1]:.3f} 安定)")
a1.axhline(0.214, color="#888", ls=":", lw=1.4, label="aux+BN出発点 (0.214)")
a1.set_title("mel再構成損失: 失敗版(ドリフト) vs 修正版(アンカー)", fontproperties=jp, fontsize=11)
a1.set_xlabel("GAN step"); a1.set_ylabel("mel-L1"); a1.grid(alpha=0.3); a1.legend(prop=jp, fontsize=9)

# Panel 2: corrected GAN components
if xc:
    a2.plot(xc, C["d"],   color="#4C72B0", lw=1.5, label="d (判別器)")
    a2.plot(xc, C["adv"], color="#C44E52", lw=1.5, label="adv (敵対的)")
    a2.plot(xc, C["fm"],  color="#CCB974", lw=1.5, label="fm (特徴マッチ)")
    a2.plot(xc, C["mel"], color="#2c7",    lw=1.5, label="mel (再構成)")
    a2.plot(xc, C["stft"],color="#7030A0", lw=1.5, label="stft")
a2.set_title(f"修正版GAN 各損失 (step {xc[-1] if xc else 0}/12000)", fontproperties=jp, fontsize=11)
a2.set_xlabel("GAN step"); a2.set_ylabel("loss"); a2.grid(alpha=0.3); a2.legend(prop=jp, fontsize=9)

fig.suptitle("a256_c128 GAN仕上げ: 修正(w_mel=45)で再構成がアンカーされ健全化", fontproperties=jp, fontsize=12)
plt.tight_layout()
out = "outputs/gan_loss.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")
print(f"failed a256: {len(xf)} pts, mel {F['mel'][-1] if xf else '-'}")
print(f"corrected a256: {len(xc)} pts (step {xc[-1] if xc else 0}), mel {C['mel'][-1] if xc else '-'}")
