# MiniSpeech

エッジデバイス向けの日本語 TTSです。

1. **MiniSpeechEncoder** がテキストからメルスペクトログラム (声の特徴) を生成
2. **Vocoder** がメルスペクトログラムから音声波形を生成

デフォルトのボコーダは **Vocos** です。**HiFi-GAN**・**MB-iSTFT** も実装しており、評価用として同じ条件で学習・評価できます。

> 🔊 **デモ**: [合成音声サンプルの比較](https://nnn112358.github.io/minispeech_demo/)
> (エンコーダ × ボコーダの組み合わせを音声とメルスペクトログラムで比較)
> / **ONNX 推論**: [minispeech_onnx](https://github.com/nnn112358/minispeech_onnx) (学習済みモデル同梱)


## 全体の流れ

```
テキスト → Piper-Plus-G2p→ MiniSpeechEncoder → メルスペクトログラム → Vocoder → 音声
```

## MiniSpeechEncoder とは？

テキスト (の音素列) からメルスペクトログラムを生成する音響モデルです。

[piper-plus](https://github.com/ayutaz/piper-plus)をベースに、エンコーダの重い部分 (Self-Attention, VAE, Normalizing Flow) を削って軽量な depthwise-separable Conv ブロックに置き換えたものです。
G2P、自己アライメントのコード (`monotonic_align.py`) や mel 計算 (`mel.py`) を piper-plus から流用しています。

### 設計の背景

piper-plus(VITS) はエンドツーエンドで高品質ですが、モデル全体で約 30M パラメータあり、
構造が複雑 (Transformer encoder + VAE + Flow + HiFi-GAN decoder の一体型) です。

MiniSpeechEncoder では、piper-plus の構成要素のうち:
- **Transformer encoder** → depthwise-separable Conv に置き換え
- **VAE + Flow** → 削除 (Duration Predictor + Length Regulator で十分)
- **HiFi-GAN decoder** → 分離して独立ボコーダに (交換可能に)
- **Pitch / Energy Predictor** → 削除 (単一話者に限定)

とすることで、0.45〜3.08M パラメータ・外部ツール不要の自己完結モデルにしています。

### アーキテクチャ・設計の詳細

設計の根拠 (ConvBlock 構成、Self-Attention を Conv で代替する理由、VITS との構造的な違い、
dim 構成、関連論文)、piper / piper-plus との比較、自己アライメントの仕組みは
[docs/architecture.md](docs/architecture.md) にまとめています。

### 実装リンク
- [piper (VITS)](https://github.com/rhasspy/piper)
- [piper-plus (VITS)](https://github.com/ayutaz/piper-plus)
- [OpenJTalk](https://open-jtalk.sp.nitech.ac.jp/)

## Vocos とは？

ConvNeXt ブロックと iSTFT (逆短時間フーリエ変換) を組み合わせたボコーダです。
周波数領域で振幅と位相を予測し、iSTFT で波形に変換します。

- 高品質で、マルチスレッド CPU で特に速い
- 3種のボコーダの中では最も高品質
- 判別器 (MPD/MRD) は全ボコーダの GAN 学習で共有

関連資料:
- 論文: [Vocos (Siuzdak, 2023)](https://arxiv.org/abs/2306.00814)
- リポジトリ: [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)

## 比較用ボコーダ (HiFi-GAN / MB-iSTFT)

評価・比較用の HiFi-GAN・MB-iSTFT の詳細と、3 ボコーダの比較表は
[docs/vocoders.md](docs/vocoders.md) にまとめています (デフォルトの Vocos は上記参照)。

## インストール

```bash
# 仮想環境を作成 (uv を使用)
uv venv

# GPU 学習には CUDA 版 PyTorch を別途インストール
uv pip install torch torchaudio numpy scipy librosa soundfile numba einops pyyaml \
    onnxruntime pytorch-lightning transformers huggingface_hub matplotlib

# 日本語テキストからの g2p (synth.py / synth_onnx.py で --text を使う場合) には
# piper-plus の OpenJTalk g2p (piper_plus_g2p) が別途必要
```

## 使い方

リポジトリのルートから実行します:

```bash
# --- 音響モデル (MiniSpeechEncoder) ---
# 自己アライメントで学習 (推奨):
uv run python src/cli/train_minispeech.py --manifest fs_data/jsut_manifest.json --learn-alignment --epochs 500

# --- ボコーダ (Vocos / HiFi-GAN / MB-iSTFT) ---
uv run python src/cli/vocos_train.py --filelist data/filelist_train.txt --out checkpoints/vocos_jsut
uv run python src/cli/hifigan_train.py --filelist data/filelist_train.txt --out checkpoints/hifigan_jsut
uv run python src/cli/mbistft_train.py --filelist data/filelist_train.txt --out checkpoints/mbistft_jsut

# --- 推論・評価 ---
# テキストから音声を生成:
uv run python src/cli/synth.py --fs-ckpt fs/jsut_align/fs_500.pth --voc-ckpt checkpoints/vocos_jsut/vocos_last.pth --text "おはようございます"

# ボコーダ単体の評価 (正解メルから音声を再合成):
uv run python src/cli/eval_vocos.py --ckpt checkpoints/vocos_jsut/vocos_last.pth
uv run python src/cli/eval_hifigan.py --ckpt checkpoints/hifigan_jsut/hifigan_last.pth
uv run python src/cli/eval_mbistft.py --ckpt checkpoints/mbistft_jsut/mbistft_last.pth

# --- ONNX (encoder + vocoder を書き出し → PyTorch 不要で推論) ---
# --slim でグラフ最適化, --fp16 で float16 量子化 (サイズ約1/2) も同時に:
uv run python src/cli/export_onnx_all.py --fs-ckpt fs/enc_d192/fs_2000.pth \
    --voc-ckpt checkpoints/vocos_jsut/vocos_last.pth --voc-type vocos --slim --fp16
uv run python src/cli/synth_onnx.py --encoder onnx_v2/enc_d192_encoder.onnx \
    --vocoder onnx_v2/vocos_d256_n512_vocoder.onnx --text "おはようございます"
uv run python src/tools/bench_vocoders.py     # ボコーダの速度比較
```



## データ

学習には **JSUT** 単一話者日本語コーパス (CC-BY-SA 4.0) を使用します。
本リポジトリには含まれません。

1. 22.05 kHz の wav ファイルを `data/wavs/` に配置
2. ファイルパスを `data/filelist_{train,val}.txt` に1行1パスで列挙
3. MiniSpeechEncoder 用のマニフェスト (`fs_data/jsut_manifest.json`) を用意:
   `{phoneme_ids, mel, n_frames}` の JSON 配列。`--learn-alignment` 使用時は
   durations は不要

## 関連リポジトリ

### minispeech_onnx — ONNX 推論実装 (エッジ向け)

[**minispeech_onnx**](https://github.com/nnn112358/minispeech_onnx) は、MiniSpeechEncoder + Vocos を
使ったエッジデバイス向けの ONNX 推論実装です。JSUT basic5000 (単一話者日本語,
CC-BY-SA 4.0) で学習済みの ONNX モデル (encoder + vocoder) を同梱し、テキストを渡すだけで音声を生成できます。

### デモ

[**合成音声サンプルの比較**](https://nnn112358.github.io/minispeech_demo/) — 6 種のエンコーダ ×
4 種のボコーダ (計 24 通り) で合成したサンプル音声と、各メルスペクトログラムをブラウザ上で比較できます。

## ライセンス

MIT ([LICENSE](LICENSE))。

本リポジトリは以下のオープンソースソフトウェアのコードを含んでいます:

| ソフトウェア | ライセンス | 利用箇所 |
|---|---|---|
| [Vocos](https://github.com/gemelo-ai/vocos) | MIT | 比較ボコーダ・判別器 (`src/vocos/`) |
| [HiFi-GAN](https://github.com/jik876/hifi-gan) | MIT | 比較ボコーダ (`src/decoders/hifigan/generator.py`) |
| [piper-plus (MB-iSTFT-VITS2)](https://github.com/ayutaz/piper-plus) | MIT | 比較ボコーダ (`src/decoders/mb_istft/`) |

