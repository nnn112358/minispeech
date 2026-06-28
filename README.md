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

[piper](https://github.com/rhasspy/piper) (VITS ベースの TTS) を分解し、重い部分 (Self-Attention, VAE, Normalizing Flow) を削って軽量な depthwise-separable Conv ブロックに置き換えたものです。
自己アライメントのコード (`monotonic_align.py`) や mel 計算 (`mel.py`) も piper から流用しています。

### 設計の背景

piper (VITS) はエンドツーエンドで高品質ですが、モデル全体で約 30M パラメータあり、
構造が複雑 (Transformer encoder + VAE + Flow + HiFi-GAN decoder の一体型) です。

MiniSpeechEncoder では、piper の構成要素のうち:
- **Transformer encoder** → depthwise-separable Conv に置き換え
- **VAE + Flow** → 削除 (Duration Predictor + Length Regulator で十分)
- **HiFi-GAN decoder** → 分離して独立ボコーダに (交換可能に)
- **Pitch / Energy Predictor** → 削除 (単一話者に限定)

とすることで、0.45〜3.08M パラメータ・外部ツール不要の自己完結モデルにしています。

### アーキテクチャの系譜
MiniSpeechEncoder は Duration 展開による非自己回帰 (Non-AR) 音響モデルで、
ブロック構造を MobileNet / ConvNeXt 由来の軽量 Conv で構成しています。

#### ConvBlock の設計

MiniSpeechEncoder の ConvBlock は複数の先行研究のアイディアを組み合わせています:

```
入力 ─→ [Depthwise Conv] → [Pointwise Conv] → +残差 → LayerNorm
                                                          ↓
                                         [FFN (d→2d→d)] → +残差 → LayerNorm → 出力
```

| 要素 | 出典 | 役割 |
|---|---|---|
| Depthwise-separable Conv | MobileNet (Google, 2017) | チャネル間結合をPointwiseに分離し計算量を削減 |
| LayerNorm + 残差接続 | Transformer (Vaswani et al., 2017) | 勾配の安定化、深いネットワークの学習を可能に |
| FFN (expand→ReLU→contract) | Transformer の Feed-Forward 層 | チャネル方向の非線形変換 |
| Conv で Attention を代替 | ConvNeXt (Meta, 2022) | Transformer 的な設計を Conv だけで再現しても同等性能 |


#### Self-Attention を Conv で代替できる理由

Self-Attention は系列の任意の位置間の依存関係を捉えますが、
TTS の Duration ベースモデルでは以下の理由で Conv (局所的な操作) で十分です:

1. **Encoder 側**: 音素列は言語的に局所的な依存が支配的
   (隣接する音素同士の影響が最も大きい)。kernel=5 の Conv で前後 2 音素をカバーし、
   4 層積めば実効受容野は系列全体に及ぶ
2. **Decoder 側**: Duration Predictor が「どの音素が何フレーム続くか」を決定した後は、
   `repeat_interleave` で展開された隣接フレームはほぼ同じ音素に対応するため、
   局所的な畳み込みだけで十分
3. **計算量**: Self-Attention は系列長 N に対して O(N²)、Conv は O(N)。
   短い音素列でも長い音素列でも一定コストで動く

#### VITS (piper-plus) との構造的な違い

MiniSpeechEncoder と VITS (piper-plus) は Duration の扱いが根本的に異なります:

```
MiniSpeechEncoder (Duration 整数展開):
  音素 → Encoder → Duration Predictor → round → long (整数)
                                                    ↓
                                          repeat_interleave → Decoder → mel
  
  Duration は整数で一本化。mel 長 = sum(durations) で厳密に一致。

piper/piper-plus：
  音素 → TextEncoder → StochasticDP → w (float)
                                         └─→ ceil(w) → Attention展開 → Flow → HiFi-GAN → 波形
```



#### エンコーダの dim 構成

dim (隠れ次元) を変えることで精度とサイズのトレードオフを調整できます。パラメータ数は dim の二乗にほぼ比例します (FFN の d→2d→d が支配的)。


| dim | enc | dec | パラメータ数 |
|-----|-----|-----|------------|
| 256 | 4 | 4 | 3.08M |
| 192 | 4 | 4 | 1.75M |
| 128 | 4 | 4 | 0.79M |
| 96 | 4 | 4 | 0.45M |



#### 関連論文
- SpeedySpeech: [Vainer & Dredze, 2020](https://arxiv.org/abs/2008.03802) — Depthwise-separable conv を TTS に導入
- MobileNet: [Howard et al., 2017](https://arxiv.org/abs/1704.04861) — Depthwise-separable conv の提案
- ConvNeXt: [Liu et al., 2022](https://arxiv.org/abs/2201.03545) — Conv だけで Transformer に匹敵
- One TTS Alignment: [Badlani et al., 2021](https://arxiv.org/abs/2108.10447) — 自己アライメント手法

### piper / piper-plus との比較

| | piper (VITS) | piper-plus (VITS2) | MiniSpeech |
|---|---|---|---|
| 構成 | End2End | End2End | 2段階 (音響モデル + ボコーダ) |
| 生成方式 | Non-AR (Flow + GAN) | Non-AR (Flow + GAN) | Non-AR (Duration展開) |
| エンコーダ | Transformer | Transformer | Depthwise-separable Conv |
| デコーダ | HiFi-GAN (内蔵) | MB-iSTFT (内蔵) | 独立ボコーダ (交換可能) |
| 出力 | 波形を直接出力 | 波形を直接出力 | mel → 別途 vocoder で波形化 |
| 継続長 | VAE + Flow + MAS | VAE + Flow + MAS | Duration Predictor (整数, 一本化) |
| Duration の型 | float (raw w) + ceil の二重定義 | float + ceil の二重定義 | **long (整数, 一本化)** |
| G2P | espeak-ng | OpenJTalk | OpenJTalk |
| パラメータ数 | ~15.7M | ~19.6M | 0.93〜4.43M (enc+voc) |
| ライセンス | MIT | MIT | MIT |
| ボコーダ交換 | 不可 (一体型) | 不可 (一体型) | 可能 (Vocos / HiFi-GAN / MB-iSTFT) |

### 自己アライメント

テキストと音声の時間的な対応 (継続長) を、学習中に自動で獲得します (`--learn-alignment`)。


手法は [One TTS Alignment (Badlani et al., 2021)](https://arxiv.org/abs/2108.10447) に基づきます:
AlignmentEncoder (cosine 類似度) で soft alignment を求め、
CTC 損失で単調性を誘導し、MAS で hard な継続長を抽出します。
アライメント用の重みは学習時のみ使用し、推論時は Duration Predictor だけで動きます。

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

## MB-iSTFT とは？

多帯域逆短時間フーリエ変換 (Multi-Band inverse STFT) を用いたボコーダです。
piper-plus のデコーダと同じ構成で、アップサンプリング → サブバンド iSTFT → PQMF 合成
の3段階で波形を生成します。

- piper-plus (MB-iSTFT-VITS2) のデコーダ部分を独立ボコーダとして切り出したもの
- HiFi-GAN と同等のパラメータ数 (1.45M) で、サブバンド STFT 損失による高品質学習
- 学習時にサブバンド STFT 損失を追加 (GAN + mel + sub-band STFT)

関連資料:
- 論文: [MB-iSTFT-VITS2 (Kawamura et al., ICASSP 2023)](https://arxiv.org/abs/2210.15975)
- 実装元: [piper-plus](https://github.com/ahoennecke/piper-plus)

## HiFi-GAN とは？

転置畳み込みによるアップサンプリングと残差ブロックで波形を生成するボコーダです。
VITS など多くの TTS システムの標準ボコーダとして広く使われています。

- 本リポでは piper (HiFi-GAN ) の構成 (ResBlock2, 256ch) を採用
- Vocos・MB-iSTFT との比較ベースラインとして使用

関連資料:
- 論文: [HiFi-GAN (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)
- リポジトリ: [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- piper (VITS TTS): [rhasspy/piper](https://github.com/rhasspy/piper)

## ボコーダ比較

| ボコーダ | 位置づけ | 方式 | パラメータ数 | 特徴 |
|---|---|---|---|---|
| **Vocos** | **デフォルト** | GAN (ConvNeXt + iSTFT) | 13.5M | 高品質、マルチスレッドに強い |
| HiFi-GAN | 評価・比較用 | GAN (転置畳み込み) | 1.46M | 軽量、VITS TTS 標準構成 |
| MB-iSTFT | 評価・比較用 | GAN (多帯域 iSTFT + PQMF) | 1.45M | piper-plus 方式、軽量かつ高品質 |

デフォルト構成は **MiniSpeechEncoder + Vocos**。HiFi-GAN / MB-iSTFT は比較用で、
3種とも同じデータ・同じ判別器 (MPD/MRD) で学習し、GAN の損失計算 (`common/gan_train.py`) と
評価ループ (`common/eval_vocoder.py`) を共有しています。


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
| [piper-plus (MB-iSTFT-VITS2)](https://github.com/ahoennecke/piper-plus) | MIT | 比較ボコーダ (`src/decoders/mb_istft/`) |

