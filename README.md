# fast-jtts

日本語テキストから音声を生成する **軽量 TTS** (Text-to-Speech) です。

処理は2段階で行います:

1. **FastSpeech lightning** がテキストからメルスペクトログラム (声の特徴) を生成
2. **SqueezeWave** がメルスペクトログラムから音声波形を生成

比較のため **Vocos** と **HiFi-GAN** でも同じ条件で学習・評価できます。

## 機能追加

- FastSpeech lightning に**自己アライメント**を追加 (外部ツール VITS・MFA が不要に)
- SqueezeWave に**4段学習** (NLL → aux → GAN → BN再キャリブ) を追加
- 3種のボコーダを同じデータ・同じ条件で比較

> 学習データ・チェックポイントは含みません ([データ](#データ) 参照)。
> ソースコード・実行スクリプトのみ。

## 全体の流れ

```
テキスト → OpenJTalk (読み変換) → FastSpeech lightning → メル → SqueezeWave → 音声
```

音響モデルとボコーダの間で受け渡すメルスペクトログラムの形式は完全に一致させています。

## FastSpeech lightning とは？

テキスト (の音素列) からメルスペクトログラムを生成する音響モデルです。
FastSpeech をベースに、軽量化と自己アライメント機能の追加を行っています。

| | FastSpeech (原論文) | FastSpeech lightning (本リポ) |
|---|---|---|
| エンコーダ/デコーダ | Self-Attention (FFT Block) | depthwise-separable Conv のみ |
| Duration Predictor | あり (教師 AR モデルから蒸留) | あり (**自己アライメント**で自動取得) |
| Pitch/Energy Predictor | なし | なし |
| 外部依存 | 教師モデル (Transformer TTS) が必要 | 不要 (自己完結で学習可能) |

- **自己アライメント** (`--learn-alignment`): テキストと音声の対応を学習中に
  自動で獲得。外部ツール (VITS, MFA, 教師モデル) が不要になる

関連資料:
- 論文: [FastSpeech (Ren et al., 2019)](https://arxiv.org/abs/1905.09263)
- 自己アライメント手法: [One TTS Alignment (Badlani et al., 2021)](https://arxiv.org/abs/2108.10447)
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

3種のボコーダは同じデータ・同じ判別器で学習し、GAN の損失計算 (`sqzw/gan_train.py`) と
評価ループ (`sqzw/eval_vocoder.py`) を共有しています。

## リポジトリ構成

```
src/
  sqzw/        コアライブラリ
    model.py        SqueezeWave 本体
    features.py     メルスペクトログラムの設定 (全モデル共通)
    mel.py          スペクトログラム計算
    flow.py         学習用ユーティリティ (微分可能逆フロー、損失関数)
    gan_train.py    GAN 学習の共有ロジック
    eval_vocoder.py 評価ループの共有ロジック
    alignment.py    自己アライメント (CTC + MAS)
    vocos_gen.py    Vocos モデル
    hifigan_gen.py  HiFi-GAN モデル
  vocos/       Vocos の依存コード (MIT ライセンス)
  cli/         学習・推論スクリプト (それぞれ独立して実行可)
  tools/       ベンチマーク・分析ツール
scripts/       一括実行スクリプト
```

## インストール

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # GPU 学習には CUDA 版 PyTorch を別途インストール
```

## 使い方

リポジトリのルートから実行します:

```bash
# --- 音響モデル (FastSpeech lightning) ---
# 自己アライメントで学習 (推奨):
python src/cli/train_fastspeech.py --manifest fs_data/jsut_manifest.json --learn-alignment --epochs 500

# --- ボコーダ ---
# SqueezeWave 4段学習を一括実行:
bash scripts/run_gan_finish.sh

# または個別に実行:
python src/cli/train.py --filelist data/filelist_train.txt --out checkpoints/run1
python src/cli/finetune_aux.py --init-from checkpoints/run1/sqzw_last.pth
python src/cli/finetune_gan.py --init-from checkpoints/run1/sqzwaux_last.pth --w-mel 45
python src/cli/bake_bnrecal.py --ckpt checkpoints/run1/sqzwgan_last.pth

# 比較ボコーダの学習:
python src/cli/hifigan_train.py --filelist data/filelist_train.txt --out checkpoints/hifigan_jsut
python src/cli/vocos_train.py --filelist data/filelist_train.txt --out checkpoints/vocos_jsut

# --- 推論・評価 ---
# テキストから音声を生成:
python src/cli/synth.py --fs-ckpt fs/jsut_align/fs_500.pth --voc-ckpt checkpoints/<run>/sqzwgan_bnrecal.pth

# ボコーダ単体の評価 (正解メルから音声を再合成):
python src/cli/infer.py --ckpt checkpoints/<run>/sqzwgan_bnrecal.pth --n 4 --sigma 0.7
python src/cli/eval_vocos.py --ckpt data/ckpts/vocos_last.pth
python src/cli/eval_hifigan.py --ckpt checkpoints/hifigan_jsut/hifigan_last.pth

# --- ONNX エクスポート ---
python src/cli/export_onnx.py
python src/tools/bench_vocoders.py     # ボコーダの速度比較
```

## 新規話者への適応

声の特徴は音響モデル (FastSpeech lightning) が決めます。ボコーダは話者に依存しません。

```bash
# 対象話者のデータで音響モデルを追加学習:
python src/cli/train_fastspeech.py --resume fs/jsut_align/fs_500.pth \
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
3. FastSpeech lightning 用のマニフェスト (`fs_data/jsut_manifest.json`) を用意:
   `{phoneme_ids, mel, n_frames}` の JSON 配列。`--learn-alignment` 使用時は
   durations は不要

## ライセンス

MIT ([LICENSE](LICENSE))。

本リポジトリは以下のオープンソースソフトウェアのコードを含んでいます:

| ソフトウェア | ライセンス | 利用箇所 |
|---|---|---|
| [Vocos](https://github.com/charactr-platform/vocos) | MIT | 比較ボコーダ・判別器 (`src/vocos/`) |
| [WaveGlow](https://github.com/NVIDIA/waveglow) / [SqueezeWave](https://github.com/tianrengao/SqueezeWave) | BSD-3 | SqueezeWave 本体 (`src/sqzw/model.py`) |
| [HiFi-GAN](https://github.com/jik876/hifi-gan) | MIT | 比較ボコーダ (`src/sqzw/hifigan_gen.py`) |

詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 参照。
設計ノートは [PLAN.md](PLAN.md) 参照。
