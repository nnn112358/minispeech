#!/usr/bin/env python3
"""wav/ の全 encoder×vocoder スペクトログラム PNG を一覧する SPECTROGRAMS.md を生成。

PNG は事前に作っておく(例: gen_demos.py / 全組み合わせ生成)。本スクリプトは
存在する PNG をマトリクス表に並べるだけ(欠落セルには ⚠️ を付ける)。

  uv run python tools/gen_gallery_md.py
"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # onnx_v2/
OUT = os.path.join(HERE, "SPECTROGRAMS.md")
ENCODERS = ["enc_d64", "enc_d96", "enc_d128", "enc_d192", "enc_d256", "enc_d384"]
VOCODERS = ["vocos_d64_n512", "vocos_d128_n512", "vocos_d256_n512",
            "vocos_d512_n1024", "hifigan_ch256", "mbistft_n16", "mbistft_n64"]
SHORT = {"vocos_d64_n512": "vocos_d64", "vocos_d128_n512": "vocos_d128",
         "vocos_d256_n512": "vocos_d256", "vocos_d512_n1024": "vocos_d512",
         "hifigan_ch256": "hifigan", "mbistft_n16": "mbistft_n16",
         "mbistft_n64": "mbistft_n64"}
W = 200  # サムネイル幅 [px]


def main():
    n = len(ENCODERS) * len(VOCODERS)
    lines = [
        f"# スペクトログラム一覧 (encoder × vocoder = {n})\n",
        "同一テキスト **「おはようございます、こんにちは、こんばんは、おやすみなさい。」** を、",
        f"全 **{len(ENCODERS)} encoder × {len(VOCODERS)} vocoder = {n} 通り**で合成した "
        "mel→audio の STFT スペクトログラム",
        "(CPU, fp16, magma, dB, 0–11kHz)。各サムネイルは**クリックで mp4"
        "(音声 + 時間を示すシアンの縦棒)を再生**(原寸 PNG は `wav/<combo>.png`)。",
        "ブラウザで開ける HTML 版は [spectrograms.html](spectrograms.html)、"
        "最小構成のデモは [README](README.md)。\n",
        "生成: [`tools/gen_demos.py`](tools/gen_demos.py) の描画関数(synth → STFT → matplotlib)。\n",
        "| encoder ＼ vocoder | " + " | ".join(SHORT[v] for v in VOCODERS) + " |",
        "|" + "---|" * (len(VOCODERS) + 1),
    ]
    for e in ENCODERS:
        cells = [f"**{e}**"]
        for v in VOCODERS:
            png = f"wav/{e}__{v}.png"
            mp4 = f"wav/{e}__{v}.mp4"
            has_png = os.path.exists(os.path.join(HERE, png))
            has_mp4 = os.path.exists(os.path.join(HERE, mp4))
            link = mp4 if has_mp4 else png  # mp4 があれば再生リンク、無ければ原寸 PNG
            mark = "" if has_png else " ⚠️欠落"
            cells.append(f'[<img src="{png}" width="{W}">]({link}){mark}')
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}  ({len(ENCODERS)}x{len(VOCODERS)} = {n} cells)")


if __name__ == "__main__":
    main()
