# onnx_v2 — ONNX 推論モデル (現行版)

MiniSpeech の 2-stage TTS を **コンポーネント単位**で ONNX 化したもの。
音響モデル(encoder)とボコーダ(vocoder)を独立したグラフとして書き出し、
任意の encoder × vocoder を組み合わせて推論できる。

```
text ──(g2p)──▶ phoneme_ids ──▶ [ *_encoder.onnx ] ──▶ mel ──▶ [ *_vocoder.onnx ] ──▶ audio (22.05kHz)
```

> 設計は [piper](https://github.com/rhasspy/piper) / [piper-plus](https://github.com/ahoennecke/piper-plus)
> (VITS / MB-iSTFT-VITS2) を参考にしている。g2p は **piper-plus-g2p** の JA-only パス
> (OpenJTalk ベース・intersperse なし) を使用。同梱モデルは encoder・vocoder とも
> **JSUT コーパス basic5000**(単一話者日本語、CC-BY-SA 4.0)で学習したもの。

## 目次

- [クイックスタート](#クイックスタート)
- [構成ファイル](#構成ファイル)
- [収録モデル](#収録モデル)
- [デモ](#デモ-代表-4-構成)
- [パラメータ数と推論時間](#パラメータ数と推論時間-onnx-実測)
- [I/O 契約](#io-契約)
- [推論](#推論-pytorch-不要)
- [別PCで動かす](#別pcで動かす-移植)
- [最適化](#最適化-onnxslim--fp16)
- [再生成](#再生成-チェックポイント--onnx)

## クイックスタート

```bash
cd onnx_v2
uv run synth_onnx.py --text "おはようございます" --out out.wav    # CPU
./synth.sh --gpu --text "おはようございます" --out out.wav        # GPU
```

uv が依存(onnxruntime / numpy / soundfile / g2p)を自動解決し、既定モデル
**enc_d192 + vocos_d256**(fp16)で text→wav を生成する(PyTorch 不要)。
詳しくは [推論](#推論-pytorch-不要) / [別PCで動かす](#別pcで動かす-移植)。

## 構成ファイル

| ファイル | 役割 |
|---|---|
| `synth_onnx.py` | 推論本体 (text→mel→wav)。PEP 723 で依存自己完結、CPU/GPU 対応 |
| `synth.sh` | `synth_onnx.py` の uv ラッパー (CPU/GPU 自動切替) |
| `pyproject.toml` | uv プロジェクト定義 (推論依存。PEP 723 と同じ内容) |
| `enc_onnx/` | デプロイ用 encoder (`*_encoder_slim_fp16.onnx`) |
| `vocoder_onnx/` | デプロイ用 vocoder (`*_vocoder_slim_fp16.onnx`) |
| `wav/` | デモ・比較の生成物 (wav / spectrogram png / mp4) |
| `SPECTROGRAMS.md`, `spectrograms.html` | 全 encoder×vocoder スペクトログラム一覧 |
| `tools/` | デモ・一覧の生成スクリプト |


## 収録モデル

デプロイ用の **fp16 量子化版 (`*_slim_fp16.onnx`)** を encoder は `enc_onnx/`、vocoder は
`vocoder_onnx/` に配置(計 69MB)。onnxslim 最適化 + fp16 量子化済みで、fp32 原本との差は
≤0.4%(relRMSE) = 可聴差なし。fp32 原本・slim 単独版は同梱せず、必要なら
[再生成](#再生成-チェックポイント--onnx)で作り直す。`synth_onnx.py` はファイル名のみの指定でも
これらのサブディレクトリを自動探索する。

> 最小ペア: `enc_d64`(0.16M) + `vocos_d64`(0.15M) = **fp16 合計 ~1.4MB**。CPU 推論 ~3ms (RTF 0.002)。
> 自前学習(encoder precomp 2000ep / vocos GAN 30k)。品質は小サイズ相応(mel_l1 0.671 / vocos mel 0.378)。

### Encoder (`phoneme_ids → mel`) — `enc_onnx/`

| ファイル | dim | サイズ |
|---|---|---|
| `enc_d64_encoder_slim_fp16.onnx` | 64 (3+3層, 自前学習) | 0.6 MB |
| `enc_d96_encoder_slim_fp16.onnx` | 96 | 1.3 MB |
| `enc_d128_encoder_slim_fp16.onnx` | 128 | 2.1 MB |
| `enc_d192_encoder_slim_fp16.onnx` | 192 | 4.3 MB |
| `enc_d256_encoder_slim_fp16.onnx` | 256 | 7.2 MB |
| `enc_d384_encoder_slim_fp16.onnx` | 384 | 15.4 MB |

### Vocoder (`mel → audio`) — `vocoder_onnx/`

| ファイル | 種別 | 構成 | サイズ |
|---|---|---|---|
| `vocos_d64_n512_vocoder_slim_fp16.onnx` | Vocos | dim=64, n_fft=512 (tiny, 自前学習) | 0.8 MB |
| `vocos_d128_n512_vocoder_slim_fp16.onnx` | Vocos | dim=128, n_fft=512 (Lite-C) | 1.6 MB |
| `vocos_d256_n512_vocoder_slim_fp16.onnx` | Vocos | dim=256, n_fft=512 (Lite-A) | 4.3 MB |
| `vocos_d512_n1024_vocoder_slim_fp16.onnx` | Vocos | dim=512, n_fft=1024 (Full) | 29.1 MB |
| `hifigan_ch256_vocoder_slim_fp16.onnx` | HiFi-GAN | init_channels=256 | 3.0 MB |
| `mbistft_n16_vocoder_slim_fp16.onnx` | MB-iSTFT | n_fft=16 | 2.9 MB |
| `mbistft_n64_vocoder_slim_fp16.onnx` | MB-iSTFT | n_fft=64 | 2.8 MB |

> SqueezeWave は `_unused/squeezewave/` にアーカイブ済み(active codebase 外)。
> 追加入力 `z` を要するボコーダにも `synth_onnx.py` は対応するが、ONNX は同梱しない。

## デモ (代表 4 構成)

同一テキスト **「おはようございます、こんにちは、こんばんは、おやすみなさい。」** を、
最小 → 既定 → Full まで **4 通り**の encoder × vocoder で合成した結果。
各 mp4 は **STFT スペクトログラム + 時間を示すシアンの縦棒(再生ヘッド)+ 音声**(CPU, fp16, 約 3.4s)。
生成は [`tools/gen_demos.py`](tools/gen_demos.py)(synth → spectrogram → ffmpeg)。
**全 42 通り(6 encoder × 7 vocoder)の一覧**は [SPECTROGRAMS.md](SPECTROGRAMS.md)(GitHub 表示)
または [spectrograms.html](spectrograms.html)(ブラウザ、クリックで再生)を参照。

> GitHub 上で `<video>` が再生されない場合は、各見出し行の **[▶ mp4]** リンクから開いてください
> (`wav/*.png` はスペクトログラム静止画)。

**1. enc_d64 × vocos_d64_n512** — 最小構成 (fp16 合計 ~1.4MB) — [▶ mp4](wav/enc_d64__vocos_d64_n512.mp4)

<video src="wav/enc_d64__vocos_d64_n512.mp4" poster="wav/enc_d64__vocos_d64_n512.png" controls width="720"></video>

**2. enc_d192 × vocos_d128_n512** — Lite-C — [▶ mp4](wav/enc_d192__vocos_d128_n512.mp4)

<video src="wav/enc_d192__vocos_d128_n512.mp4" poster="wav/enc_d192__vocos_d128_n512.png" controls width="720"></video>

**3. enc_d192 × vocos_d256_n512** — 既定構成 — [▶ mp4](wav/enc_d192__vocos_d256_n512.mp4)

<video src="wav/enc_d192__vocos_d256_n512.mp4" poster="wav/enc_d192__vocos_d256_n512.png" controls width="720"></video>

**4. enc_d192 × vocos_d512_n1024** — Full Vocos — [▶ mp4](wav/enc_d192__vocos_d512_n1024.mp4)

<video src="wav/enc_d192__vocos_d512_n1024.mp4" poster="wav/enc_d192__vocos_d512_n1024.png" controls width="720"></video>


### Encoder (`phoneme_ids → mel`)

| モデル | dim | パラメータ数 | fp16 サイズ | CPU 時間 |
|---|---|---|---|---|
| `enc_d64`  | 64  | 0.29 M | 0.6 MB  | 1.5 ms  |
| `enc_d96`  | 96  | 0.65 M | 1.3 MB  | 3.3 ms  |
| `enc_d128` | 128 | 1.05 M | 2.1 MB  | 4.7 ms  |
| `enc_d192` | 192 | 2.14 M | 4.3 MB  | 8.0 ms  |
| `enc_d256` | 256 | 3.61 M | 7.2 MB  | 13.1 ms |
| `enc_d384` | 384 | 7.67 M | 15.4 MB | 26.1 ms |

> エンコーダのパラメータ数は **位置エンコーディング定数バッファ(max_len 2048 × dim)を含む**。
> 学習重みのみの数(PE 除く)は順に 0.16 / 0.45 / 0.79 / 1.75 / 3.08 / 6.88 M で、
> 差分は 2048×dim(例 enc_d192: 学習 1.75M + PE 0.39M = 2.14M)。

### Vocoder (`mel → audio`)

| モデル | 種別 | パラメータ数 | fp16 サイズ | CPU 時間 |
|---|---|---|---|---|
| `vocos_d64_n512`   | Vocos    | 0.41 M  | 0.8 MB  | 3.7 ms   |
| `vocos_d128_n512`  | Vocos    | 0.80 M  | 1.6 MB  | 7.6 ms   |
| `vocos_d256_n512`  | Vocos    | 2.13 M  | 4.3 MB  | 15.9 ms  |
| `vocos_d512_n1024` | Vocos    | 14.51 M | 29.1 MB | 103.0 ms |
| `hifigan_ch256`    | HiFi-GAN | 1.46 M  | 3.0 MB  | 244.6 ms |
| `mbistft_n16`      | MB-iSTFT | 1.45 M  | 2.9 MB  | 45.4 ms  |
| `mbistft_n64`      | MB-iSTFT | 1.40 M  | 2.8 MB  | 28.3 ms  |



## I/O 契約

すべて opset 18、可変長(dynamic axes)。

**Encoder** — 入力 `phoneme_ids` : `int64 [1, L]`(先頭 BOS=1) / 出力 `mel` : `float32 [1, 80, T]`
**Vocoder** — 入力 `mel` : `float32 [1, 80, T]` / 出力 `audio` : Vocos は `[1, T_audio]`、他は `[1, 1, T_audio]`

mel は全ステージ共通の `PiperMelFeatures` 契約
(`sr=22050, n_fft=1024, hop=256, win=1024, center=False, log-clamp 1e-5`)。
encoder 出力と vocoder 入力は設計上 bit-identical。fp16 版は入出力のみ fp32(`keep_io_types`)。

## 推論 (PyTorch 不要)

`synth_onnx.py` が encoder→vocoder の ONNX 推論を行う。依存
(`onnxruntime / numpy / soundfile / piper-plus-g2p / pyopenjtalk-plus`)は
スクリプト先頭の **PEP 723 インラインメタデータ**に書いてあり、**uv が自動解決**する
(初回 DL、以降キャッシュ)。特定 venv への依存はなく、PyTorch も不要。

```bash
# 基本 (既定は fp16 モデル, enc_d192 + vocos_d256)
uv run synth_onnx.py --text "おはようございます" --out out.wav
./synth.sh --text "おはようございます"                       # uv run のショートカット

# GPU 実行 (synth.sh が onnxruntime-gpu を用意して起動)
./synth.sh --gpu --text "おはようございます" --out out.wav

# encoder / vocoder を明示 (ファイル名のみで enc_onnx/ vocoder_onnx/ を探索)
uv run synth_onnx.py --encoder enc_d384_encoder_slim_fp16.onnx \
    --vocoder hifigan_ch256_vocoder_slim_fp16.onnx --text "..." --out out.wav

# 音素ID直接指定 (g2p 不要) / ベンチ (10回平均, RTF)
uv run synth_onnx.py --phonemes 1,10,14,8,38,10,11,10,8,2 --out out.wav
uv run synth_onnx.py --text "..." --bench
```

主なオプション:
- `--encoder` / `--vocoder` — モデル指定。ファイル名のみなら `enc_onnx/`・`vocoder_onnx/` を
  探索(相対パスはスクリプト位置基準で解決、cwd 非依存)。
- `--provider {auto,cpu,cuda}` / `--gpu` — 実行プロバイダ(既定 auto)。GPU は onnxruntime-gpu 必須。
- `--text` / `--phonemes` — 入力。`--sigma` / `--seed` は z を要するボコーダ用。
- `--threads` — CPU スレッド数。`--bench` — 速度計測。

g2p は piper-plus-g2p の JA-only パス(intersperse なし)で先頭に BOS=1 を付与。
出力 ID は piper_train の `text_to_phoneme_ids_and_prosody` と bit-identical(検証済み)。

ライブラリとしても利用可:

```python
from synth_onnx import OnnxTTS, text_to_phoneme_ids
tts = OnnxTTS("enc_d192_encoder_slim_fp16.onnx",
              "vocos_d256_n512_vocoder_slim_fp16.onnx", provider="auto")
r = tts.synthesize(text_to_phoneme_ids("こんにちは"))
r.audio  # np.float32 [-1, 1], 22.05kHz
```

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv 未導入なら
cd onnx_v2 && uv run synth_onnx.py --text "おはよう" --out out.wav
```



**速度の実測 (参考値, median ms, mel T≈131 ≒ 音声1.5s, onnxruntime 1.22)**
CPU = threads=1 / GPU = RTX 3070 Laptop (numpy in/out の H2D/D2H 込み):

| モデル | CPU orig | CPU fp16 | GPU orig | GPU fp16 | fp16 relRMSE |
|---|---|---|---|---|---|
| enc_d192 | 2.96 | 3.64 | 0.90 | **0.64** | 0.08% |
| enc_d256 | 5.00 | 6.14 | 0.79 | **0.66** | 0.09% |
| enc_d384 | 9.99 | 12.34 | 0.90 | **0.72** | 0.11% |
| vocos_d128 | **2.52** | 3.13 | **0.33** | 0.77 | 0.39% |
| vocos_d256 | **5.82** | 7.03 | **0.37** | 0.80 | 0.27% |
| vocos_d512 | **36.8** | 43.9 | **0.99** | 2.42 | 0.15% |
| hifigan | **75.2** | 81.8 | 2.06 | **1.32** | 0.07% |
| mbistft_n16 | **17.9** | 19.6 | 0.78 | **0.60** | 0.15% |
| mbistft_n64 | **11.2** | 11.9 | **0.41** | 0.47 | 0.08% |



