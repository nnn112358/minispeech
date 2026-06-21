#!/usr/bin/env python3
"""Stacked bar: 2-stage TTS total inference time = MiniSpeech encoder (fs_enc+
fs_dec, constant) + vocoder (varies per config). ONNX Runtime CPU, 1 thread,
256-frame / 2.972s. Annotates the parameter changed and the SpecConv quality."""
import sys, time, tempfile
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, onnxruntime as ort
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sqzw.model import SqueezeWave
from sqzw.onnx_export import SqueezeWaveONNX, FRAMES

jp = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
SR = 22050; AUD = 65536 / SR
D = "checkpoints/"


from sqzw.bench import bench


def export_voc(ckpt, path):
    ck = torch.load(D + ckpt, map_location="cpu"); cfg = ck["config"]
    m = SqueezeWave(**cfg); m.load_state_dict(ck["model"]); m = SqueezeWave.remove_weightnorm(m).eval()
    nac = cfg["n_audio_channel"]; l = FRAMES * (256 // nac)
    wrap = SqueezeWaveONNX(m).eval(); mel = torch.randn(1, 80, FRAMES); z = torch.randn(1, nac, l)
    with torch.no_grad(): wrap(mel, z)
    torch.onnx.export(wrap, (mel, z), path, opset_version=18, input_names=["mel", "z"], output_names=["audio"])
    return l, nac


def main():
    tmp = tempfile.mkdtemp()
    ph = np.zeros((1, 128), np.int64); feat = np.random.randn(1, 256, 256).astype(np.float32)
    mel = np.random.randn(1, 80, FRAMES).astype(np.float32)
    # (label, ckpt, param-change, SpecConv, LogMag, color)
    VOC = [
        ("a256_c128",     "ckpts_aux/sqzwaux_last.pth",                "ベース\n(c128/f12/l8)", 0.242, 0.658, "#4C72B0"),
        ("c64_f12_l8",    "ckpts_c64_f12_l8_aux/sqzwaux_bnrecal.pth",  "WN幅\nc128→64",        0.392, 0.701, "#55A868"),
        ("c64_f8_l8+dil", "ckpts_c64_f8_l8_dgan_aux/sqzwaux_last.pth", "flow\n12→8 +dil",      0.427, 0.708, "#CCB974"),
        ("c64_f8_l4",     "ckpts_c64_f8_l4_dgan_aux/sqzwaux_last.pth", "WN層\n8→4",            0.471, 0.740, "#E08214"),
    ]
    voc_paths = []
    for label, ck, change, sc, lm, col in VOC:
        p = f"{tmp}/{label}.onnx"; export_voc(ck, p)
        nac = torch.load(D + ck, map_location='cpu')["config"]["n_audio_channel"]
        voc_paths.append((p, nac))

    def measure(threads):
        enc = bench("npu/model/fs_enc.onnx", {"phonemes": ph}, threads) \
            + bench("npu/model/fs_dec.onnx", {"feat": feat}, threads)
        rows = []
        for (label, ck, change, sc, lm, col), (p, nac) in zip(VOC, voc_paths):
            l = FRAMES * (256 // nac)
            tv = bench(p, {"mel": mel, "z": np.random.randn(1, nac, l).astype(np.float32)}, threads)
            rows.append((label, tv, change, sc, lm, col))
        tv = bench("npu/model/vocoder_fix.onnx", {"mel": mel}, threads)
        rows.append(("Vocos", tv, "別アーキ\n(iSTFT)", 0.270, 0.649, "#999999"))
        return enc, rows

    enc1, rows1 = measure(1)
    enc4, rows4 = measure(4)
    print(f"encoder: 1thread {enc1:.1f}ms  4thread {enc4:.1f}ms", flush=True)

    ymax = max(max(enc1 + r[1] for r in rows1), max(enc4 + r[1] for r in rows4)) * 1.18
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
    for ax, (ENC, rows, tl) in zip(axs, [(enc1, rows1, "1スレッド (per-core)"), (enc4, rows4, "4スレッド (4-core)")]):
        xs = list(range(len(rows)))
        for i, (label, tv, change, sc, lm, col) in enumerate(rows):
            ax.bar(i, ENC, color="#3a3f4b", edgecolor="white", width=0.62)
            ax.bar(i, tv, bottom=ENC, color=col, edgecolor="white", width=0.62)
            ax.text(i, ENC/2, f"enc{ENC:.0f}", ha="center", va="center", color="white", fontsize=7, fontproperties=jp)
            ax.text(i, ENC+tv/2, f"{tv:.0f}", ha="center", va="center", color="white", fontsize=8.5, fontproperties=jp)
            ax.text(i, ENC+tv+0.6, f"計{ENC+tv:.0f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold", fontproperties=jp)
        ax.set_xticks(xs); ax.set_xticklabels([f"{r[0]}\n[{r[2]}]" for r in rows], fontsize=8, fontproperties=jp)
        ax.set_ylim(0, ymax); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        ax.set_title(tl, fontproperties=jp, fontsize=11)
        ax2 = ax.twinx()
        scs = [r[3] for r in rows]; lms = [r[4] for r in rows]
        ax2.plot(xs, scs, "o-", color="#C00000", lw=2.2, ms=7, label="SpecConv loss")
        ax2.plot(xs, lms, "s--", color="#7030A0", lw=1.5, ms=5, label="LogMag loss")
        for i, s in enumerate(scs):
            ax2.annotate(f"{s:.3f}", (i, s), textcoords="offset points", xytext=(5, 6), fontsize=8, color="#C00000")
        ax2.set_ylim(0.15, 0.85); ax2.tick_params(axis="y", colors="#C00000")
        ax2.set_ylabel("loss (線, 低い=高品質)", fontproperties=jp, fontsize=9, color="#C00000")
        ax2.legend(prop=jp, loc="upper center", fontsize=8)
    axs[0].set_ylabel("推論時間 / 発話 [ms]  (棒, ONNX CPU, 2.97s音声)", fontproperties=jp, fontsize=10)
    fig.suptitle("2段TTS: 推論時間(棒=エンコーダ共通+ボコーダ, 左軸) と 品質loss(線, 右軸)  ―  1スレッド vs 4スレッド\n"
                 "［］=Vocosから変えたパラメータ。スレッド数でVocos(高並列)とSqueezeWave(逐次flow)の優劣が変化", fontproperties=jp, fontsize=11)
    plt.tight_layout()
    out = "outputs/time_vs_loss.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    for tag, ENC, rows in [("1thr", enc1, rows1), ("4thr", enc4, rows4)]:
        for label, tv, change, sc, lm, col in rows:
            print(f"  [{tag}] {label:14} total {ENC+tv:5.1f}ms (enc{ENC:.1f}+voc{tv:.1f})")


if __name__ == "__main__":
    main()
