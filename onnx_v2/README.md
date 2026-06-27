# onnx_v2 — ONNX 推論モデル (現行版)

MiniSpeech の 2-stage TTS を **コンポーネント単位**で ONNX 化したもの。
音響モデル(encoder)とボコーダ(vocoder)を独立したグラフとして書き出し、
任意の encoder × vocoder を組み合わせて推論できる。

```
text ──(g2p)──▶ phoneme_ids ──▶ [ *_encoder.onnx ] ──▶ mel ──▶ [ *_vocoder.onnx ] ──▶ audio (22.05kHz)
```

| ファイル | 役割 |
|---|---|
| `synth_onnx.py` | 推論本体 (text→mel→wav)。PEP 723 で依存自己完結、CPU/GPU 対応 |
| `synth.sh` | `synth_onnx.py` の uv ラッパー (CPU/GPU 自動切替) |
| `optimize_onnx.py` | onnxslim + fp16 量子化ツール |
| `*_slim_fp16.onnx` | デプロイ用モデル (slim+fp16 済み) |
| `archive/` | fp32 原本・slim 単独版の退避先 |

> 旧 `onnx/` との違い: 旧版は音響モデルを encoder/decoder の 2 グラフに分け、プリセット名で
> 3 ファイル束ねていた。本 v2 は `forward_infer`(encode→duration→length-regulate→decode)を
> **1 グラフに統合**し、構成キーで独立命名している。

## ファイル一覧

ルートには **fp16 量子化版 (`*_slim_fp16.onnx`) のみ**を配置(デプロイ用、計 68MB)。
onnxslim 最適化 + fp16 量子化済みで、fp32 原本との差は ≤0.4%(relRMSE) = 可聴差なし。
fp32 原本と slim 単独版は `archive/`(計 269MB)に退避。原本は `archive/<name>.onnx` で指定可。

### Encoder (`phoneme_ids → mel`)

| ファイル | dim | サイズ |
|---|---|---|
| `enc_d192_encoder_slim_fp16.onnx` | 192 | 4.3 MB |
| `enc_d256_encoder_slim_fp16.onnx` | 256 | 7.2 MB |
| `enc_d384_encoder_slim_fp16.onnx` | 384 | 15.4 MB |

### Vocoder (`mel → audio`)

| ファイル | 種別 | 構成 | サイズ |
|---|---|---|---|
| `vocos_d128_n512_vocoder_slim_fp16.onnx` | Vocos | dim=128, n_fft=512 (Lite-C) | 1.6 MB |
| `vocos_d256_n512_vocoder_slim_fp16.onnx` | Vocos | dim=256, n_fft=512 (Lite-A) | 4.3 MB |
| `vocos_d512_n1024_vocoder_slim_fp16.onnx` | Vocos | dim=512, n_fft=1024 (Full) | 29.1 MB |
| `hifigan_ch256_vocoder_slim_fp16.onnx` | HiFi-GAN | init_channels=256 | 3.0 MB |
| `mbistft_n16_vocoder_slim_fp16.onnx` | MB-iSTFT | n_fft=16 | 2.9 MB |
| `mbistft_n64_vocoder_slim_fp16.onnx` | MB-iSTFT | n_fft=64 | 2.8 MB |

> SqueezeWave は `_unused/squeezewave/` にアーカイブ済み(active codebase 外)。
> 追加入力 `z` を要するボコーダにも `synth_onnx.py` は対応するが、ONNX は同梱しない。

## モデル諸元 (実測)

fp32 原本での測定。CPU = threads=1、mel T≈131(≒音声1.5s)、median of 60。
**メモリはモデル単体(正味)** = プロセス全体 − ランタイム土台(Python+onnxruntime ≈ 46MB)。
GPU の時間・VRAM は [最適化](#最適化-onnxslim--fp16) のベンチ表を参照。

| モデル | 種別 | パラメータ数 | CPU RAM(正味) | CPU 時間 |
|---|---|---|---|---|
| `enc_d192` | Encoder | 1.75 M | 23 MB | 3.1 ms |
| `enc_d256` | Encoder | 3.08 M | 30 MB | 5.6 ms |
| `enc_d384` | Encoder | 6.88 M | 50 MB | 10.5 ms |
| `vocos_d128_n512` | Vocoder | 0.80 M | 18 MB | 2.7 ms |
| `vocos_d256_n512` | Vocoder | 2.13 M | 25 MB | 6.0 ms |
| `vocos_d512_n1024` | Vocoder | 14.51 M | 94 MB | 37.0 ms |
| `hifigan_ch256` | Vocoder | 1.46 M | 68 MB | 73.4 ms |
| `mbistft_n16` | Vocoder | 1.45 M | 26 MB | 17.3 ms |
| `mbistft_n64` | Vocoder | 1.40 M | 21 MB | 10.8 ms |

- **パラメータ数**: ONNX initializer の総要素数。fp16 でも同数(バイトサイズは半分)。
- **メモリ(正味)** = モデルの重み(≈ params×4B/fp32) + アクティベーションのアリーナ。
  `hifigan` は**重みが小さいのにメモリ大**(アップサンプリングで中間テンソルが巨大)。
- **実アプリの総量はこれに土台を足す**: 例 `synth_onnx.py`(enc_d192+vocos_d256, fp16, g2p 込み)の
  実測ピーク RSS = **約 93MB**(土台 onnxruntime 44 → +enc+voc 83 → +g2p 91 → 推論 93)。
  encoder と vocoder は 1 プロセスで土台を共有するので、表の正味値は単純加算してよい。
- end-to-end 時間は encoder+vocoder の合算(例: enc_d192+vocos_d256 → CPU ~9ms / GPU ~1.2ms)。
- デプロイの fp16 版は重みメモリ・サイズが約½(→ [最適化](#最適化-onnxslim--fp16))。

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

# encoder / vocoder を明示
uv run synth_onnx.py --encoder enc_d384_encoder_slim_fp16.onnx \
    --vocoder hifigan_ch256_vocoder_slim_fp16.onnx --text "..." --out out.wav

# fp32 原本を使う (archive/ 内)
uv run synth_onnx.py --vocoder archive/vocos_d256_n512_vocoder.onnx --text "..."

# 音素ID直接指定 (g2p 不要) / ベンチ (10回平均, RTF)
uv run synth_onnx.py --phonemes 1,10,14,8,38,10,11,10,8,2 --out out.wav
uv run synth_onnx.py --text "..." --bench
```

主なオプション:
- `--encoder` / `--vocoder` — モデル指定。相対パスはスクリプト位置基準で解決(cwd 非依存)。
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

## 別PCで動かす (移植)

**uv さえ入っていれば、フォルダをコピーするだけで即動く。** 依存(Python 本体含む)は
`uv run` が PEP 723 メタデータから自動ブートストラップする。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv 未導入なら
cd onnx_v2 && uv run synth_onnx.py --text "おはよう" --out out.wav
```

検証済み(クリーン環境シミュレーション): uv キャッシュ・Python 配置先を空にし、既存
venv/conda を外した状態で、uv が **管理 Python を新規DL** + PyPI から全依存を解決して
音声生成まで完走することを確認。絶対パスは無く、pyopenjtalk-plus は manylinux/aarch64
wheel 配布のため**コンパイラ不要**。

⚠️ 移植時の注意:
- **`.onnx` を必ず同梱**する(git 管理外)。デプロイはルートの fp16 一式(68MB)だけで足りる。
  fp32 原本が要るなら `archive/`(269MB)も。
- **初回実行はネットワーク必須**(PyPI 依存DL + OpenJTalk 辞書取得)。オフラインは別途対応。
- Python は `>=3.10` 開放で uv は最新を選ぶ。wheel 未提供の最新版に当たったら
  `uv run --python 3.12 synth_onnx.py ...` で固定。
- GPU は `onnxruntime-gpu==1.22.0`(CUDA 12 系)。CUDA/cuDNN はシステム or nvidia pip 必要。

## 最適化 (onnxslim / fp16)

`optimize_onnx.py` が onnxslim 最適化と fp16 量子化を行う(同梱の `*_slim_fp16.onnx` の生成元)。

```bash
# slim + fp16 を生成 (元ファイルは残る)。出力は <name>_slim.onnx / <name>_slim_fp16.onnx
uv run optimize_onnx.py archive/enc_d192_encoder.onnx archive/vocos_d256_n512_vocoder.onnx --fp16
```

- `<name>_slim.onnx` — onnxslim でグラフ最適化(冗長ノード除去)。**出力数値は不変**。
- `<name>_slim_fp16.onnx` — slim 後に fp16 量子化(`keep_io_types`)。**サイズ約1/2**、品質劣化 ≤0.4%。
  エンコーダの posenc 由来 Cast 型不整合はツールが自動修正。生成後 ORT ロード検証し、
  失敗するモデルは fp16 を自動スキップ。

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

要点:
- **GPU は桁違いに速い**(重いモデルほど顕著: hifigan ~36x, vocos_d512 ~37x, vocos_d256 ~16x)。
- **fp16 は CPU では常に遅い**(Cast 挿入 + CPU に fp16 ネイティブ演算が無い)。利点はサイズ½。
- **fp16 は GPU でもモデル次第**: エンコーダ(+29%)・HiFi-GAN(+36%)・MB-iSTFT(n16) は速いが、
  **Vocos は逆に遅い**(vocos_d512 で 2.4倍)。原因は **Vocos 固有演算(iSTFT/ConvNeXt)の fp16
  カーネルが遅い**ため — fp16 が足す Cast は入出力境界の 2 個のみで、`fp16 後の再 slim でも
  ノード数・速度は不変`(検証済み)。なので **slim→fp16 の順で既に最適**。
- onnxslim 単独はエンコーダで微増、畳み込み系は ORT 最適化と被り中立(出力不変)。

**おすすめ構成** (速度最優先なら):
- **CPU**: encoder=slim(or 原本), vocoder=**原本(fp32)**。end-to-end ~8.7ms (RTF~0.006)。
- **GPU**: encoder=**fp16**, Vocos=**原本(fp32)**, HiFi-GAN=**fp16**。end-to-end ~1.0ms (RTF~0.0007)。
- **サイズ最優先**: 同梱の fp16 一式(品質差 ≤0.4%)。

速度・品質は必ず実機・実行プロバイダで `synth_onnx.py --bench` で確認すること。

## 再生成 (チェックポイント → ONNX)

`src/cli/export_onnx_all.py` で書き出す(デフォルト出力先が `onnx_v2/`)。

```bash
# encoder + vocoder をまとめて
python src/cli/export_onnx_all.py --fs-ckpt fs/enc_d192/fs_2000.pth \
    --voc-ckpt checkpoints/vocos_lite_c/vocos_last.pth --voc-type vocos

# vocoder のみ
python src/cli/export_onnx_all.py \
    --voc-ckpt checkpoints/hifigan_jsut_v2/hifigan_last.pth --voc-type hifigan
```

- `--voc-type`: `vocos` / `hifigan` / `mbistft`。ファイル名は checkpoint の `config` から自動決定。
- encoder は `enc_d{dim}` で命名され**既存ならスキップ**。encoder × vocoder を自由に追加できる。
- 書き出した fp32 原本に `optimize_onnx.py --fp16` をかけてデプロイ用 fp16 を作る。
