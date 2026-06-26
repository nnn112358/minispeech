import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_log(path):
    epochs, mel, dur = [], [], []
    with open(path) as f:
        for line in f:
            m = re.match(r"epoch (\d+) mel_l1 ([\d.]+) dur ([\d.]+)", line)
            if m:
                epochs.append(int(m.group(1)))
                mel.append(float(m.group(2)))
                dur.append(float(m.group(3)))
    return epochs, mel, dur

# d=256: chain 3 sessions
e1, m1, d1 = parse_log("/home/nnn/mini-jtts/_legacy/fs_train.log")
e2, m2, d2 = parse_log("/home/nnn/mini-jtts/_legacy/fs_ft.log")
e3, m3, d3 = parse_log("/home/nnn/mini-jtts/fs/jsut_10k.log")
d256_ep = e1 + [500 + x for x in e2] + [1000 + x for x in e3]
d256_mel = m1 + m2 + m3
d256_dur = d1 + d2 + d3

# d=192, d=128, d=96
e192, m192, dur192 = parse_log("/home/nnn/mini-jtts/fs/enc_d192.log")
e128, m128, dur128 = parse_log("/home/nnn/mini-jtts/fs/enc_d128.log")
e96, m96, dur96 = parse_log("/home/nnn/mini-jtts/fs/enc_d96.log")

# d=256 (JSUT, same conditions as sweep)
e256j, m256j, dur256j = parse_log("/home/nnn/mini-jtts/fs/enc_d256.log")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

configs = [
    (e256j, m256j, dur256j, "d=256 (3.08M)", "#1f77b4"),
    (e192, m192, dur192, "d=192 (1.75M)", "#ff7f0e"),
    (e128, m128, dur128, "d=128 (0.79M)", "#2ca02c"),
    (e96, m96, dur96, "d=96 (0.35M)", "#d62728"),
]

for ep, mel, dur, label, color in configs:
    ax1.plot(ep, mel, label=label, color=color, alpha=0.8, linewidth=1.2)
    ax2.plot(ep, dur, label=label, color=color, alpha=0.8, linewidth=1.2)

ax1.set_ylabel("mel_l1")
ax1.set_title("Encoder Dim Sweep — Loss Curves")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.45, 0.85)

ax2.set_ylabel("dur_loss")
ax2.set_xlabel("epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.04, 0.16)

plt.tight_layout()
plt.savefig("/tmp/claude-1000/-media-nnn-------260611-TTS/9e07cdd7-e658-44d0-85b8-11886ac837cc/scratchpad/enc_sweep_loss3.png", dpi=150)
print("saved")
