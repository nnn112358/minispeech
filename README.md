# MiniSpeech

日本語テキストから音声を生成する **軽量 TTS** (Text-to-Speech) です。

処理は2段階で行います:

1. **MiniSpeechEncoder** がテキストからメルスペクトログラム (声の特徴) を生成
2. **ボコーダ** がメルスペクトログラムから音声波形を生成

ボコーダは **Vocos**・**HiFi-GAN**・**MB-iSTFT** から選択でき、同じ条件で学習・評価できます。
(初期実験で使った **SqueezeWave** は `_unused/squeezewave/` にアーカイブ済み)

## 機能追加

- MiniSpeechEncoder に**自己アライメント**を追加 (外部ツール VITS・MFA が不要に)
- 3種のボコーダ (Vocos / HiFi-GAN / MB-iSTFT) を同じデータ・同じ条件で比較
- 共有判別器 (MPD/MRD) による GAN 学習で全ボコーダの音質を底上げ

> 学習データ・チェックポイントは含みません ([データ](#データ) 参照)。
> ソースコード・実行スクリプトのみ。

## 全体の流れ

```
テキスト → OpenJTalk (読み変換) → MiniSpeechEncoder → メル → ボコーダ → 音声
```

音響モデルとボコーダの間で受け渡すメルスペクトログラムの形式は完全に一致させています。

## MiniSpeechEncoder とは？

テキスト (の音素列) からメルスペクトログラムを生成する音響モデルです。

[piper](https://github.com/rhasspy/piper) (VITS ベースの TTS) を分解し、
重い部分 (Self-Attention, VAE, Normalizing Flow) を削って
軽量な depthwise-separable Conv ブロックに置き換えたものです。
自己アライメントのコード (`monotonic_align.py`) や mel 計算 (`mel.py`) も piper から持ってきています。

### 設計の背景

piper (VITS) はエンドツーエンドで高品質ですが、モデル全体で約 30M パラメータあり、
構造が複雑 (Transformer encoder + VAE + Flow + HiFi-GAN decoder の一体型) です。

MiniSpeechEncoder では、piper の構成要素のうち:
- **Transformer encoder** → depthwise-separable Conv に置き換え
- **VAE + Flow** → 削除 (Duration Predictor + Length Regulator で十分)
- **HiFi-GAN decoder** → 分離して独立ボコーダに (交換可能に)
- **Pitch / Energy Predictor** → 省略 (単一話者なら Duration Predictor のみで十分)

とすることで、0.35〜3.08M パラメータ・外部ツール不要の自己完結モデルにしています。

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

TTS 分野では **SpeedySpeech** (Vainer & Dredze, 2020) が depthwise-separable conv を
音響モデルに導入した先行例です。

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

VITS系 (piper-plus):
  音素 → TextEncoder → StochasticDP → w (float)
                                         ├─→ ceil(w) → Attention展開 → Flow → HiFi-GAN → 波形
                                         └─→ w (raw) → ONNX出力 → ランタイムの trim に使用

  Duration が float (raw w) と int (ceil w) の二重定義になり、
  末尾に余分な音声が残るバグの温床になる (piper-plus issue #268)。
```

MiniSpeechEncoder では `round → long` で整数化した duration をそのまま `repeat_interleave` に
渡すため、生成される mel 長と出力長が構造的に一致し、この種のバグは発生しません。

#### エンコーダの dim 構成

dim (隠れ次元) を変えることで精度とサイズのトレードオフを調整できます:

| dim | enc | dec | パラメータ数 | mel_l1 @2000ep |
|-----|-----|-----|------------|----------------|
| 256 | 4 | 4 | 3.08M | (学習中) |
| 192 | 4 | 4 | 1.75M | 0.496 |
| 128 | 4 | 4 | 0.79M | 0.565 |
| 96 | 4 | 4 | 0.35M | (学習中) |

パラメータ数は dim の二乗にほぼ比例します (FFN の d→2d→d が支配的)。

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
外部ツール (VITS, MFA, 教師モデル) が不要になります。

手法は [One TTS Alignment (Badlani et al., 2021)](https://arxiv.org/abs/2108.10447) に基づきます:
AlignmentEncoder (cosine 類似度) で soft alignment を求め、
CTC 損失で単調性を誘導し、MAS で hard な継続長を抽出します。
アライメント用の重みは学習時のみ使用し、推論時は Duration Predictor だけで動きます。

### 実装リンク

- 元にした TTS: [piper (VITS)](https://github.com/rhasspy/piper)
- 読み変換: [OpenJTalk](https://open-jtalk.sp.nitech.ac.jp/)

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

- 本リポでは piper (VITS ベースの TTS) の構成 (ResBlock2, 256ch) を採用
- Vocos・MB-iSTFT との比較ベースラインとして使用

関連資料:
- 論文: [HiFi-GAN (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)
- リポジトリ: [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- piper (VITS TTS): [rhasspy/piper](https://github.com/rhasspy/piper)

## ボコーダ比較

| ボコーダ | 方式 | パラメータ数 | 特徴 |
|---|---|---|---|
| HiFi-GAN | GAN (転置畳み込み) | 1.46M | 軽量、VITS TTS 標準構成 |
| MB-iSTFT | GAN (多帯域 iSTFT + PQMF) | 1.45M | piper-plus 方式、軽量かつ高品質 |
| Vocos | GAN (ConvNeXt + iSTFT) | 13.5M | 高品質、マルチスレッドに強い |

3種のボコーダは同じデータ・同じ判別器 (MPD/MRD) で学習し、GAN の損失計算
(`common/gan_train.py`) と評価ループ (`common/eval_vocoder.py`) を共有しています。

## SqueezeWave (アーカイブ済み)

初期実験で使った WaveGlow 派生の正規化フローボコーダ。チャネル数・フロー段数・レイヤ数を
自由に変えられ、`c64` 構成は CPU で 400 倍超のリアルタイム比を出すなど小型・高速でしたが、
学習に4段の手順 (NLL 事前学習 → aux fine-tune → GAN fine-tune → BN 再キャリブ) を要し、
通常の単段 GAN 学習で済む上記3種に主役を譲りました。

ソース・学習スクリプト・チェックポイント一式は **`_unused/squeezewave/`** に退避済みです
(active codebase 外)。復元手順は [CLAUDE.md](CLAUDE.md) を参照。

- 論文: [SqueezeWave (Zhai et al., 2020)](https://arxiv.org/abs/2001.05685) /
  原型 [WaveGlow (Prenger et al., 2018)](https://arxiv.org/abs/1811.00002)
- 参考実装: [tianrengao/SqueezeWave](https://github.com/tianrengao/SqueezeWave)、[NVIDIA/waveglow](https://github.com/NVIDIA/waveglow)

## リポジトリ構成

```
src/
  common/          共通ライブラリ
    features.py        メルスペクトログラム (全モデル共通)
    mel.py             スペクトログラム計算
    modules.py         共通ビルディングブロック (ResBlock 等)
    dataset.py         音声セグメントデータセット
    gan_train.py       GAN 学習の共有ロジック
    eval_vocoder.py    評価ループの共有ロジック
  encoder/         エンコーダ (音響モデル)
    minispeech.py      MiniSpeechEncoder 本体
    alignment.py       自己アライメント (CTC + MAS)
    monotonic_align.py MAS 動的計画法
  decoders/        デコーダ (ボコーダ)
    hifigan/           HiFi-GAN (転置畳み込み)
    vocos/             Vocos (ConvNeXt + iSTFT)
    mb_istft/          MB-iSTFT (多帯域 iSTFT + PQMF)
  vocos/           Vocos の依存コード (MIT ライセンス)
  cli/             学習・推論スクリプト (それぞれ独立して実行可)
  tools/           ベンチマーク・分析ツール
scripts/           一括実行スクリプト
_unused/squeezewave/  アーカイブ済み SqueezeWave 一式 (source / CLI / tools / scripts / artifacts)
```

## インストール

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # GPU 学習には CUDA 版 PyTorch を別途インストール
```

## 使い方

リポジトリのルートから実行します:

```bash
# --- 音響モデル (MiniSpeechEncoder) ---
# 自己アライメントで学習 (推奨):
python src/cli/train_minispeech.py --manifest fs_data/jsut_manifest.json --learn-alignment --epochs 500

# --- ボコーダ (Vocos / HiFi-GAN / MB-iSTFT) ---
python src/cli/vocos_train.py --filelist data/filelist_train.txt --out checkpoints/vocos_jsut
python src/cli/hifigan_train.py --filelist data/filelist_train.txt --out checkpoints/hifigan_jsut
python src/cli/mbistft_train.py --filelist data/filelist_train.txt --out checkpoints/mbistft_jsut

# --- 推論・評価 ---
# テキストから音声を生成:
python src/cli/synth.py --fs-ckpt fs/jsut_align/fs_500.pth --voc-ckpt checkpoints/vocos_jsut/vocos_last.pth --text "おはようございます"

# ボコーダ単体の評価 (正解メルから音声を再合成):
python src/cli/eval_vocos.py --ckpt checkpoints/vocos_jsut/vocos_last.pth
python src/cli/eval_hifigan.py --ckpt checkpoints/hifigan_jsut/hifigan_last.pth
python src/cli/eval_mbistft.py --ckpt checkpoints/mbistft_jsut/mbistft_last.pth

# --- ONNX (encoder + vocoder を書き出し → PyTorch 不要で推論) ---
# --slim でグラフ最適化, --fp16 で float16 量子化 (サイズ約1/2) も同時に:
python src/cli/export_onnx_all.py --fs-ckpt fs/enc_d192/fs_2000.pth \
    --voc-ckpt checkpoints/vocos_jsut/vocos_last.pth --voc-type vocos --slim --fp16
python src/cli/synth_onnx.py --encoder onnx_v2/enc_d192_encoder.onnx \
    --vocoder onnx_v2/vocos_d256_n512_vocoder.onnx --text "おはようございます"
python src/tools/bench_vocoders.py     # ボコーダの速度比較
```

## 新規話者への適応

声の特徴は音響モデル (MiniSpeechEncoder) が決めます。ボコーダは話者に依存しません。

```bash
# 対象話者のデータで音響モデルを追加学習:
python src/cli/train_minispeech.py --resume fs/jsut_align/fs_500.pth \
    --manifest fs_data/<spk>/fs_manifest.json --epochs 300 --cosine --lr 2e-4

# (任意) 録音の質感に合わせてボコーダも追加学習 (vocos_train.py --resume など)
```

## 速度計測結果

### ONNX Runtime CPU (2.97 秒の音声を生成する時間)

| モデル | パラメータ数 | 1スレッド | 全スレッド |
|---|---|---|---|
| HiFi-GAN (piper) | 1.46M | 147.5ms (20倍RT) | 68.2ms (44倍RT) |
| Vocos | 13.50M | 74.2ms (40倍RT) | 21.8ms (136倍RT) |
| SqueezeWave a256 *(アーカイブ)* | 7.87M | 44.0ms (67倍RT) | 27.5ms (108倍RT) |
| SqueezeWave c64 *(アーカイブ)* | 1.35M | **7.2ms (411倍RT)** | **5.4ms (553倍RT)** |

> SqueezeWave 行は `_unused/squeezewave/` 退避前に測定した参考値です。

## データ

学習には **JSUT** 単一話者日本語コーパス (CC-BY-SA 4.0) を使用します。
本リポジトリには含まれません。

1. 22.05 kHz の wav ファイルを `data/wavs/` に配置
2. ファイルパスを `data/filelist_{train,val}.txt` に1行1パスで列挙
3. MiniSpeechEncoder 用のマニフェスト (`fs_data/jsut_manifest.json`) を用意:
   `{phoneme_ids, mel, n_frames}` の JSON 配列。`--learn-alignment` 使用時は
   durations は不要

## ライセンス

MIT ([LICENSE](LICENSE))。

本リポジトリは以下のオープンソースソフトウェアのコードを含んでいます:

| ソフトウェア | ライセンス | 利用箇所 |
|---|---|---|
| [Vocos](https://github.com/charactr-platform/vocos) | MIT | 比較ボコーダ・判別器 (`src/vocos/`) |
| [WaveGlow](https://github.com/NVIDIA/waveglow) / [SqueezeWave](https://github.com/tianrengao/SqueezeWave) | BSD-3 | SqueezeWave 本体 (アーカイブ済み、`_unused/squeezewave/decoders/squeezewave/model.py`) |
| [HiFi-GAN](https://github.com/jik876/hifi-gan) | MIT | 比較ボコーダ (`src/decoders/hifigan/generator.py`) |
| [piper-plus (MB-iSTFT-VITS2)](https://github.com/ahoennecke/piper-plus) | MIT | 比較ボコーダ (`src/decoders/mb_istft/`) |

詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 参照。
設計ノートは [PLAN.md](PLAN.md) 参照。
