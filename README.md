# mini-jtts

日本語テキストから音声を生成する **軽量 TTS** (Text-to-Speech) です。

処理は2段階で行います:

1. **MiniSpeech** がテキストからメルスペクトログラム (声の特徴) を生成
2. **SqueezeWave** がメルスペクトログラムから音声波形を生成

比較のため **Vocos**・**HiFi-GAN**・**MB-iSTFT** でも同じ条件で学習・評価できます。

## 機能追加

- MiniSpeech に**自己アライメント**を追加 (外部ツール VITS・MFA が不要に)
- SqueezeWave に**4段学習** (NLL → aux → GAN → BN再キャリブ) を追加
- 4種のボコーダを同じデータ・同じ条件で比較

> 学習データ・チェックポイントは含みません ([データ](#データ) 参照)。
> ソースコード・実行スクリプトのみ。

## 全体の流れ

```
テキスト → OpenJTalk (読み変換) → MiniSpeech → メル → SqueezeWave → 音声
```

音響モデルとボコーダの間で受け渡すメルスペクトログラムの形式は完全に一致させています。

## MiniSpeech とは？

テキスト (の音素列) からメルスペクトログラムを生成する音響モデルです。

[piper](https://github.com/rhasspy/piper) (VITS ベースの TTS) を分解し、
重い部分 (Self-Attention, VAE, Normalizing Flow) を削って
軽量な depthwise-separable Conv ブロックに置き換えたものです。
自己アライメントのコード (`monotonic_align.py`) や mel 計算 (`mel.py`) も piper から持ってきています。

### 設計の背景

piper (VITS) はエンドツーエンドで高品質ですが、モデル全体で約 30M パラメータあり、
構造が複雑 (Transformer encoder + VAE + Flow + HiFi-GAN decoder の一体型) です。

MiniSpeech では、piper の構成要素のうち:
- **Transformer encoder** → depthwise-separable Conv に置き換え
- **VAE + Flow** → 削除 (Duration Predictor + Length Regulator で十分)
- **HiFi-GAN decoder** → 分離して独立ボコーダに (交換可能に)
- **Pitch / Energy Predictor** → 省略 (単一話者なら Duration Predictor のみで十分)

とすることで、0.35〜3.08M パラメータ・外部ツール不要の自己完結モデルにしています。

### アーキテクチャの系譜

MiniSpeech は FastSpeech (Microsoft, 2019) の設計思想を踏襲し、
ブロック構造を MobileNet / ConvNeXt 由来の軽量 Conv に置き換えたものです。

#### FastSpeech との関係

FastSpeech は「テキスト → mel」を Duration 展開で非自己回帰 (Non-AR) に一発生成する
アーキテクチャの元祖です。MiniSpeech は以下の点で FastSpeech を簡略化しています:

| | FastSpeech (原論文) | MiniSpeech |
|---|---|---|
| Encoder / Decoder | Transformer (Self-Attention + FFN) | Conv のみ (depthwise-separable + FFN) |
| Duration の教師 | Teacher-Student 蒸留 (別途 Autoregressive モデルが必要) | 学習データから直接取得、または自己アライメント |
| 位置符号化 | sinusoidal | sinusoidal (同じ) |
| パラメータ数 | ~25M | 0.35〜3.08M |

#### ConvBlock の設計

MiniSpeech の ConvBlock は複数の先行研究のアイディアを組み合わせています:

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

MiniSpeech (FastSpeech 系) と VITS (piper-plus) は Duration の扱いが根本的に異なります:

```
FastSpeech系 (MiniSpeech):
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

FastSpeech 系では `round → long` で整数化した duration をそのまま `repeat_interleave` に
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

- FastSpeech: [Ren et al., 2019](https://arxiv.org/abs/1905.09263) — Duration ベース Non-AR TTS の元祖
- FastSpeech 2: [Ren et al., 2020](https://arxiv.org/abs/2006.04558) — Pitch/Energy Predictor を追加
- SpeedySpeech: [Vainer & Dredze, 2020](https://arxiv.org/abs/2008.03802) — Depthwise-separable conv を TTS に導入
- MobileNet: [Howard et al., 2017](https://arxiv.org/abs/1704.04861) — Depthwise-separable conv の提案
- ConvNeXt: [Liu et al., 2022](https://arxiv.org/abs/2201.03545) — Conv だけで Transformer に匹敵
- One TTS Alignment: [Badlani et al., 2021](https://arxiv.org/abs/2108.10447) — 自己アライメント手法

### piper / piper-plus との比較

| | piper (VITS) | piper-plus (VITS2) | mini-jtts (FastSpeech系) |
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
| ボコーダ交換 | 不可 (一体型) | 不可 (一体型) | 可能 (Vocos / SqueezeWave / HiFi-GAN / MB-iSTFT) |

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

## SqueezeWave とは？

メルスペクトログラムから音声波形を生成するボコーダです。
WaveGlow (正規化フロー) を軽量化したモデルで、全サンプルを並列に生成します。

- チャネル数・フロー段数・レイヤ数を CLI フラグで自由に変更可能
- 通常の学習だけでは品質が不足するため、本リポの4段学習で改善する

関連資料:
- 論文: [SqueezeWave (Zhai et al., 2020)](https://arxiv.org/abs/2001.05685)
- 原型: [WaveGlow (Prenger et al., 2018)](https://arxiv.org/abs/1811.00002)
- 参考実装: [tianrengao/SqueezeWave](https://github.com/tianrengao/SqueezeWave)、[NVIDIA/waveglow](https://github.com/NVIDIA/waveglow)

## Vocos とは？

ConvNeXt ブロックと iSTFT (逆短時間フーリエ変換) を組み合わせたボコーダです。
周波数領域で振幅と位相を予測し、iSTFT で波形に変換します。

- 高品質で、マルチスレッド CPU で特に速い
- 本リポでは SqueezeWave との比較ベースラインとして使用
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
- SqueezeWave・Vocos との比較ベースラインとして使用

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
| **SqueezeWave** | 正規化フロー | 1.35M〜7.9M | 小型・高速・構成の自由度が高い |

## 4段学習の手順

SqueezeWave の品質を段階的に改善します (`scripts/run_gan_finish.sh` で一括実行可):

1. **NLL 事前学習** (`cli/train.py`) — 通常の最尤推定で基本的な音声生成を学習。
2. **Aux fine-tune** (`cli/finetune_aux.py`) — 生成音声と正解音声のメル・波形の
   差を直接小さくする損失関数で追加学習。
3. **GAN fine-tune** (`cli/finetune_gan.py`) — 判別器 (本物か生成かを見分けるモデル)
   を使って、数値指標では捉えにくい音質の粗さを除去。
   判別器は学習時のみ使用し、推論時のコストは増えない。
4. **BN 再キャリブ** (`cli/bake_bnrecal.py`) — モデル内部の統計値を推論時の
   分布に合わせて再計算し、学習時と推論時のずれを解消。

4種のボコーダは同じデータ・同じ判別器で学習し、GAN の損失計算 (`common/gan_train.py`) と
評価ループ (`common/eval_vocoder.py`) を共有しています。

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
    minispeech.py      MiniSpeech 本体
    alignment.py       自己アライメント (CTC + MAS)
    monotonic_align.py MAS 動的計画法
  decoders/        デコーダ (ボコーダ)
    squeezewave/       SqueezeWave (正規化フロー)
    hifigan/           HiFi-GAN (転置畳み込み)
    vocos/             Vocos (ConvNeXt + iSTFT)
    mb_istft/          MB-iSTFT (多帯域 iSTFT + PQMF)
  vocos/           Vocos の依存コード (MIT ライセンス)
  cli/             学習・推論スクリプト (それぞれ独立して実行可)
  tools/           ベンチマーク・分析ツール
scripts/           一括実行スクリプト
```

## インストール

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # GPU 学習には CUDA 版 PyTorch を別途インストール
```

## 使い方

リポジトリのルートから実行します:

```bash
# --- 音響モデル (MiniSpeech) ---
# 自己アライメントで学習 (推奨):
python src/cli/train_minispeech.py --manifest fs_data/jsut_manifest.json --learn-alignment --epochs 500

# --- ボコーダ ---
# SqueezeWave 4段学習を一括実行:
bash scripts/run_gan_finish.sh

# または個別に実行:
python src/cli/train.py --filelist data/filelist_train.txt --out checkpoints/run1
python src/cli/finetune_aux.py --init-from checkpoints/run1/sqzw_last.pth
python src/cli/finetune_gan.py --init-from checkpoints/run1/sqzwaux_last.pth --w-mel 45
python src/cli/bake_bnrecal.py --ckpt checkpoints/run1/sqzwgan_last.pth

# 比較ボコーダの学習:
python src/cli/mbistft_train.py --filelist data/filelist_train.txt --out checkpoints/mbistft_jsut
python src/cli/hifigan_train.py --filelist data/filelist_train.txt --out checkpoints/hifigan_jsut
python src/cli/vocos_train.py --filelist data/filelist_train.txt --out checkpoints/vocos_jsut

# --- 推論・評価 ---
# テキストから音声を生成:
python src/cli/synth.py --fs-ckpt fs/jsut_align/fs_500.pth --voc-ckpt checkpoints/<run>/sqzwgan_bnrecal.pth

# ボコーダ単体の評価 (正解メルから音声を再合成):
python src/cli/infer.py --ckpt checkpoints/<run>/sqzwgan_bnrecal.pth --n 4 --sigma 0.7
python src/cli/eval_mbistft.py --ckpt checkpoints/mbistft_jsut/mbistft_last.pth
python src/cli/eval_vocos.py --ckpt data/ckpts/vocos_last.pth
python src/cli/eval_hifigan.py --ckpt checkpoints/hifigan_jsut/hifigan_last.pth

# --- ONNX エクスポート ---
python src/cli/export_onnx.py
python src/tools/bench_vocoders.py     # ボコーダの速度比較
```

## 新規話者への適応

声の特徴は音響モデル (MiniSpeech) が決めます。ボコーダは話者に依存しません。

```bash
# 対象話者のデータで音響モデルを追加学習:
python src/cli/train_minispeech.py --resume fs/jsut_align/fs_500.pth \
    --manifest fs_data/<spk>/fs_manifest.json --epochs 300 --cosine --lr 2e-4

# (任意) 録音の質感に合わせてボコーダも追加学習:
bash scripts/run_tyc_finetune.sh   # つくよみちゃんコーパスでの実例
```

## 速度計測結果

### ONNX Runtime CPU (2.97 秒の音声を生成する時間)

| モデル | パラメータ数 | 1スレッド | 全スレッド |
|---|---|---|---|
| HiFi-GAN (piper) | 1.46M | 147.5ms (20倍RT) | 68.2ms (44倍RT) |
| Vocos | 13.50M | 74.2ms (40倍RT) | 21.8ms (136倍RT) |
| SqueezeWave a256 | 7.87M | 44.0ms (67倍RT) | 27.5ms (108倍RT) |
| SqueezeWave c64 | 1.35M | **7.2ms (411倍RT)** | **5.4ms (553倍RT)** |

## データ

学習には **JSUT** 単一話者日本語コーパス (CC-BY-SA 4.0) を使用します。
本リポジトリには含まれません。

1. 22.05 kHz の wav ファイルを `data/wavs/` に配置
2. ファイルパスを `data/filelist_{train,val}.txt` に1行1パスで列挙
3. MiniSpeech 用のマニフェスト (`fs_data/jsut_manifest.json`) を用意:
   `{phoneme_ids, mel, n_frames}` の JSON 配列。`--learn-alignment` 使用時は
   durations は不要

## ライセンス

MIT ([LICENSE](LICENSE))。

本リポジトリは以下のオープンソースソフトウェアのコードを含んでいます:

| ソフトウェア | ライセンス | 利用箇所 |
|---|---|---|
| [Vocos](https://github.com/charactr-platform/vocos) | MIT | 比較ボコーダ・判別器 (`src/vocos/`) |
| [WaveGlow](https://github.com/NVIDIA/waveglow) / [SqueezeWave](https://github.com/tianrengao/SqueezeWave) | BSD-3 | SqueezeWave 本体 (`src/decoders/squeezewave/model.py`) |
| [HiFi-GAN](https://github.com/jik876/hifi-gan) | MIT | 比較ボコーダ (`src/decoders/hifigan/generator.py`) |
| [piper-plus (MB-iSTFT-VITS2)](https://github.com/ahoennecke/piper-plus) | MIT | 比較ボコーダ (`src/decoders/mb_istft/`) |

詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 参照。
設計ノートは [PLAN.md](PLAN.md) 参照。
