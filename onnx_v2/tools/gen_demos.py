#!/usr/bin/env python3
"""encoder × vocoder 10 通りで音声合成し、STFT スペクトログラム付き mp4 を生成する。

各 combo について: text → encoder.onnx → mel → vocoder.onnx → audio を合成し、
  wav/<enc>__<voc>.wav   音声
  wav/<enc>__<voc>.png   STFT スペクトログラム静止画
  wav/<enc>__<voc>.mp4   スペクトログラム + 赤い再生ヘッド + 音声
を書き出す。README の「デモ」節はこの mp4 を埋め込んでいる。

実行 (matplotlib を一時追加。ffmpeg はシステムに必要):
  uv run --with matplotlib python tools/gen_demos.py
"""
import os
import subprocess
import sys

import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # onnx_v2/
sys.path.insert(0, HERE)
from synth_onnx import OnnxTTS, text_to_phoneme_ids  # noqa: E402

TEXT = "おはようございます、こんにちは、こんばんは、おやすみなさい。"
SR = 22050
OUT = os.path.join(HERE, "wav")

ENCODERS_ALL = ["enc_d64", "enc_d96", "enc_d128", "enc_d192", "enc_d256", "enc_d384"]
VOCODERS_ALL = ["vocos_d64_n512", "vocos_d128_n512", "vocos_d256_n512",
                "vocos_d512_n1024", "hifigan_ch256", "mbistft_n16", "mbistft_n64"]

# 全組み合わせ (6 encoder × 7 vocoder = 42)
COMBOS = [(e, v) for e in ENCODERS_ALL for v in VOCODERS_ALL]


def spectrogram_png(audio, sr, title, png):
    """スペクトログラム画像を保存し、プロット領域のピクセル矩形 (left,top,w,h) を返す。
    再生ヘッドはこの矩形内だけを time 軸に揃えて動かす。"""
    dpi = 100
    fig, ax = plt.subplots(figsize=(10.0, 3.2), dpi=dpi)  # 1000x320 (even dims)
    nfft, hop = 1024, 256
    ax.specgram(audio, NFFT=nfft, Fs=sr, noverlap=nfft - hop,
                cmap="magma", scale="dB", mode="magnitude", vmin=-100, vmax=-10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("freq [Hz]")
    ax.set_ylim(0, sr / 2)
    fig.tight_layout()
    fig.canvas.draw()  # 位置を確定させてから bbox を取得
    w_px, h_px = fig.get_size_inches() * dpi
    bb = ax.get_position()  # figure 比率 (y0=下)
    left = bb.x0 * w_px
    top = (1.0 - bb.y1) * h_px           # 画像座標は上原点
    plot_w = (bb.x1 - bb.x0) * w_px
    plot_h = (bb.y1 - bb.y0) * h_px
    fig.savefig(png)
    plt.close(fig)
    return left, top, plot_w, plot_h


def make_mp4(png, wav, mp4, dur, geom):
    # 静止スペクトログラム + 時間を示す縦棒(シアン)+ 音声。
    # 縦棒は color ソースを overlay の動的 x で動かす(drawbox の式は per-frame 評価
    # されず動かないため overlay を使う)。x はプロット領域に揃え time 軸と一致させる。
    left, top, plot_w, plot_h = geom
    bw = 5  # 縦棒の幅 [px]
    x_expr = f"{left:.1f}+({plot_w:.1f}-{bw})*t/{dur:.4f}"
    bar = f"color=c=cyan:s={bw}x{int(round(plot_h))}:r=25"
    vf = (f"[0:v][1:v]overlay=x='{x_expr}':y={top:.1f}:shortest=1,"
          f"scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v]")
    cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", "25", "-i", png,
           "-f", "lavfi", "-i", bar, "-i", wav,
           "-filter_complex", vf, "-map", "[v]", "-map", "2:a",
           "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", mp4]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    ph = text_to_phoneme_ids(TEXT)
    print(f"phonemes={len(ph)}  text={TEXT}")
    for enc, voc in COMBOS:
        name = f"{enc}__{voc}"
        tts = OnnxTTS(f"{enc}_encoder_slim_fp16.onnx",
                      f"{voc}_vocoder_slim_fp16.onnx", provider="cpu")
        r = tts.synthesize(ph)
        dur = len(r.audio) / SR
        wav = os.path.join(OUT, name + ".wav")
        png = os.path.join(OUT, name + ".png")
        mp4 = os.path.join(OUT, name + ".mp4")
        sf.write(wav, r.audio, SR)
        geom = spectrogram_png(r.audio, SR, f"{enc}  x  {voc}", png)
        make_mp4(png, wav, mp4, dur, geom)
        print(f"{name:34s} dur={dur:4.2f}s enc={r.enc_ms:5.1f}ms voc={r.voc_ms:6.1f}ms "
              f"mp4={os.path.getsize(mp4) / 1e6:.2f}MB")
    print(f"\nDONE -> {OUT}")


if __name__ == "__main__":
    main()
