# ボコーダ詳細

> [README](../README.md) のボコーダ説明の詳細。MiniSpeech は 3 種のボコーダを同じデータ・
> 同じ判別器 (MPD/MRD) で学習・比較できます。デフォルトの **Vocos** は README を参照してください。

## ボコーダ比較

| ボコーダ | 位置づけ | 方式 | パラメータ数 | 特徴 |
|---|---|---|---|---|
| **Vocos** | **デフォルト** | GAN (ConvNeXt + iSTFT) | 13.5M | 高品質、マルチスレッドに強い |
| HiFi-GAN | 評価・比較用 | GAN (転置畳み込み) | 1.46M | 軽量、VITS TTS 標準構成 |
| MB-iSTFT | 評価・比較用 | GAN (多帯域 iSTFT + PQMF) | 1.45M | piper-plus 方式、軽量かつ高品質 |

デフォルト構成は **MiniSpeechEncoder + Vocos**。HiFi-GAN / MB-iSTFT は比較用で、
3種とも同じデータ・同じ判別器 (MPD/MRD) で学習し、GAN の損失計算 (`common/gan_train.py`) と
評価ループ (`common/eval_vocoder.py`) を共有しています。

## MB-iSTFT とは？

多帯域逆短時間フーリエ変換 (Multi-Band inverse STFT) を用いたボコーダです。
piper-plus のデコーダと同じ構成で、アップサンプリング → サブバンド iSTFT → PQMF 合成
の3段階で波形を生成します。

- piper-plus (MB-iSTFT-VITS2) のデコーダ部分を独立ボコーダとして切り出したもの
- HiFi-GAN と同等のパラメータ数 (1.45M) で、サブバンド STFT 損失による高品質学習
- 学習時にサブバンド STFT 損失を追加 (GAN + mel + sub-band STFT)

関連資料:
- 論文: [MB-iSTFT-VITS2 (Kawamura et al., ICASSP 2023)](https://arxiv.org/abs/2210.15975)
- 実装元: [piper-plus](https://github.com/ayutaz/piper-plus)

## HiFi-GAN とは？

転置畳み込みによるアップサンプリングと残差ブロックで波形を生成するボコーダです。
VITS など多くの TTS システムの標準ボコーダとして広く使われています。

- 本リポでは piper (HiFi-GAN) の構成 (ResBlock2, 256ch) を採用
- Vocos・MB-iSTFT との比較ベースラインとして使用

関連資料:
- 論文: [HiFi-GAN (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)
- リポジトリ: [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- piper (VITS TTS): [rhasspy/piper](https://github.com/rhasspy/piper)
