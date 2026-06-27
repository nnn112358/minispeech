# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Two-stage non-autoregressive Japanese TTS (Duration 展開方式):
**OpenJTalk g2p → MiniSpeechEncoder (text→mel) → Vocoder (mel→audio)**

**デフォルト構成は MiniSpeechEncoder + Vocos**。HiFi-GAN / MB-iSTFT は評価・比較用の
ボコーダで、同一の mel 契約・同一データで学習・評価できる(default を置き換えるものではない)。
(SqueezeWave は `_unused/squeezewave/` にアーカイブ済み — active codebase からは外れている)
All stages share identical mel features (`PiperMelFeatures`: sr=22050, n_fft=1024, hop=256, win=1024, center=False, log-clamp 1e-5). MiniSpeechEncoder output and vocoder input are bit-identical by design.

## Running scripts

All scripts assume **repo root** (`/home/nnn/mini-jtts`) as cwd. CLI scripts auto-add `src/` to sys.path.
Python venv: `~/.venvs/piper_ft/bin/python`

```bash
# --- 音響モデル (MiniSpeechEncoder) ---
# 自己アライメントで学習:
python src/cli/train_minispeech.py --manifest fs_data/fs_manifest.json --learn-alignment --epochs 500

# precomputed durations で学習 (dim/層数指定可):
python src/cli/train_minispeech.py --manifest fs_data/fs_manifest.json --dim 192 --n-enc 4 --n-dec 4 --epochs 2000 --cosine

# --- ボコーダ ---
# Vocos (lite構成指定可):
python src/cli/vocos_train.py --filelist data/filelist_train.txt --dim 256 --num-layers 4 --n-fft 512

# MB-iSTFT:
python src/cli/mbistft_train.py --filelist data/filelist_train.txt --out checkpoints/mbistft_jsut

# HiFi-GAN:
python src/cli/hifigan_train.py --filelist data/filelist_train.txt --out checkpoints/hifigan_jsut

# --- 推論 ---
# テキストから音声を生成 (encoderのconfig自動検出):
python src/cli/synth.py --fs-ckpt fs/enc_d192/fs_2000.pth --voc-ckpt checkpoints/vocos_lite_a/vocos_last.pth --text "おはようございます"

# ボコーダ単体の評価 (copy-synthesis):
python src/cli/eval_vocos.py --ckpt checkpoints/vocos_lite_a/vocos_last.pth --n 4

# --- ONNX ---
# encoder + vocoder を ONNX 化 (出力先 onnx_v2/)。--slim/--fp16 で最適化も同時に:
python src/cli/export_onnx_all.py --fs-ckpt fs/enc_d192/fs_2000.pth --voc-ckpt checkpoints/vocos_lite_c/vocos_last.pth --voc-type vocos --slim --fp16
# encoder のみ / vocoder のみも可 (--voc-ckpt / --fs-ckpt を省略)
# ONNX のみで text→wav 推論:
python src/cli/synth_onnx.py --encoder onnx_v2/enc_d192_encoder.onnx --vocoder onnx_v2/vocos_d128_n512_vocoder.onnx --text "おはようございます"
python src/tools/bench_vocoders.py
```

## Code layout

- `src/common/` — shared library (features, mel, modules, dataset, GAN loss, eval loop)
- `src/encoder/` — MiniSpeechEncoder acoustic model, alignment, MAS
- `src/decoders/hifigan/` — HiFi-GAN (transposed conv)
- `src/decoders/vocos/` — Vocos (ConvNeXt + iSTFT)
- `src/decoders/mb_istft/` — MB-iSTFT (multi-band iSTFT + PQMF)
- `src/cli/` — training and inference entry points (each is a standalone script with argparse)
- `src/tools/` — dev-only analysis, plotting, sweeps, diagnostics
- `src/vocos/` — vendored Vocos (trimmed, MIT); discriminators used for GAN training only
- `scripts/` — bash pipelines that compose CLI scripts into full recipes
- `npu/` — Pulsar2 config templates (build artifacts gitignored)
- `_unused/`, `_legacy/` — archived experiments (not part of active codebase)。SqueezeWave 一式 (source / CLI / tools / scripts / artifacts) は `_unused/squeezewave/` に退避済み
- `fs/` — MiniSpeechEncoder チェックポイント (`enc_d{96,128,192,256}/`)
- `checkpoints/` — Vocoder チェックポイント (`vocos_lite_{a,b,c}/`, `vocos_jsut/`, etc.)

## Key architecture details

### MiniSpeechEncoder

Duration 展開方式の軽量音響モデル。Self-Attention を全て depthwise-separable Conv に置き換え。

- **ConvBlock**: Depthwise Conv (MobileNet) + Pointwise Conv + LayerNorm + FFN (Transformer) + 残差接続
- **Duration**: `round → long` で整数化 → `repeat_interleave` で展開。Duration は一本化されており VITS の float/ceil 二重定義バグは構造的に発生しない
- **dim 構成**: `--dim`, `--n-enc`, `--n-dec` で指定。checkpoint に `config` dict として保存され、synth.py が自動検出
- **位置エンコーディング**: `max_len`(既定 2048≈24s)の正弦波 PE を非永続バッファ `_pe` に precompute し slice。`arange().float()` の `Cast(to=FLOAT)` を排し ONNX を軽量化&fp16 変換可能に(`convert_float_to_float16` が当該 Cast を壊す問題を回避)。値は旧実装と bit-identical、`persistent=False` で旧 checkpoint も無改変ロード可。T>max_len は PyTorch のみ動的 fallback
- **Self-Alignment** (`--learn-alignment`): cosine-similarity attention + beta-binomial prior + forward-sum (CTC) loss + MAS。Aligner weights は save 時に strip
- checkpoint format: `{"model": state_dict, "n_sym": int, "epoch": int, "config": {"dim": int, "n_enc": int, "n_dec": int}}`
- DP key remap: 旧チェックポイントは `dp.net.3.*` → 現行は `dp.net.2.*`。synth.py/train で自動 remap

### JA-only g2p (intersperse なし)

piper-plus は多言語対応で音素間に blank token (id=0) を挿入する (intersperse) が、
JA-only モデルは intersperse なしで学習している。推論時に `PiperEncoder.encode()` を使うと
blank が挿入されて mel が壊れる。代わりに以下の JA-only パスを使う:

```python
from piper_plus_g2p.encode.id_maps import get_phoneme_id_map
from piper_train.infer_onnx import text_to_phoneme_ids_and_prosody
id_map = get_phoneme_id_map("ja")
ids, _ = text_to_phoneme_ids_and_prosody(text, id_map, language="ja", language_id_map=None)
ph_ids = [1] + ids  # prepend BOS
```

### Vocos vocoder (default)

- **Full**: dim=512, layers=8, n_fft=1024, 13.5M params
- **Lite-A**: dim=256, layers=4, n_fft=512, 1.91M params
- **Lite-B**: dim=192, layers=6, n_fft=512, 1.59M params
- **Lite-C**: dim=128, layers=4, n_fft=512, 0.58M params
- checkpoint format: `{"G": state_dict, "config": {"dim", "intermediate_dim", "num_layers", "n_fft", "hop_length"}}`
- synth.py の vocoder 自動判定: `"G" in ck and "dim" in ck.get("config", {})` → Vocos

### SqueezeWave vocoder (アーカイブ済み)

WaveGlow 派生の normalizing flow。`_unused/squeezewave/` に退避済みで active codebase からは外れている。
復元する場合は source (`decoders/squeezewave/`) と CLI (`train.py` / `finetune_aux.py` / `finetune_gan.py` /
`bake_bnrecal.py` / `infer.py` 等) を元の場所に戻し、synth.py / export_onnx_all.py の auto-detect 分岐を再追加する。

### MB-iSTFT vocoder (評価・比較用)

- piper-plus decoder アーキテクチャ (upsample → sub-band iSTFT → PQMF synthesis)
- PQMF: 4 bands, 62 taps, Kaiser window (beta=9), cutoff_ratio=0.142, PR-SNR=64dB
- ONNX-compatible iSTFT (DFT matrix 方式, `stft_onnx.py`)
- 学習時にサブバンド STFT loss を使用

### HiFi-GAN vocoder (評価・比較用)

- piper config (ResBlock2, 256ch, upsample 8×8×4), 1.51M params
- comparison baseline

### Shared components

- **PiperMelFeatures** (`common/features.py`): the shared mel contract. Any change here breaks compatibility between MiniSpeechEncoder and all vocoders.
- **Discriminators** (Vocos MPD/MRD): GAN 学習で全 vocoder が共有。training-only (zero inference cost)。

## Conventions

- No build system. Plain `pip install -r requirements.txt` + CUDA torch installed separately.
- 現行 vocoder (Vocos / HiFi-GAN / MB-iSTFT) は決定論的な mel→audio。ONNX export は mel 1 入力のみ
  (アーカイブ済み SqueezeWave のみ noise `z` を入力として渡す方式だった)。
- Data filelists are newline-separated paths to 22.05 kHz mono wavs.
- MiniSpeechEncoder manifest: JSON array of `{phoneme_ids, durations (optional), mel (npy path), n_frames}`.

## Gotchas

- `w_mel=45` in GAN fine-tune is a tuned anchor — changing it risks quality regression.
- Self-alignment `--bin-weight` must stay 0 (binarization loss destabilizes alignment).
- Beta-binomial prior is annealed to zero by 40% of training.
- The vendored `src/vocos/` is a trimmed fork (no HuggingFace, no EnCodec). Don't update from upstream without checking removals.
- `num_workers=0` in MiniSpeechEncoder DataLoader is intentional — >0 was slower due to fork/copy overhead for this workload.
- Bash tool の `timeout` コマンドや `env` 接頭辞越しにモデル読込すると無音死する。export + plain python + ツール側 timeout で回避。
