# MiniSpeechEncoder アーキテクチャの系譜

> [README](../README.md) の「MiniSpeechEncoder とは？」の詳細。設計の根拠と先行研究との関係をまとめます。

MiniSpeechEncoder は Duration 展開による非自己回帰 (Non-AR) 音響モデルで、
ブロック構造を MobileNet / ConvNeXt 由来の軽量 Conv で構成しています。

## ConvBlock の設計

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

## Self-Attention を Conv で代替できる理由

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

## VITS (piper-plus) との構造的な違い

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

## エンコーダの dim 構成

dim (隠れ次元) を変えることで精度とサイズのトレードオフを調整できます。パラメータ数は dim の二乗にほぼ比例します (FFN の d→2d→d が支配的)。

| dim | enc | dec | パラメータ数 |
|-----|-----|-----|------------|
| 256 | 4 | 4 | 3.08M |
| 192 | 4 | 4 | 1.75M |
| 128 | 4 | 4 | 0.79M |
| 96 | 4 | 4 | 0.45M |

## 関連論文

- SpeedySpeech: [Vainer & Dredze, 2020](https://arxiv.org/abs/2008.03802) — Depthwise-separable conv を TTS に導入
- MobileNet: [Howard et al., 2017](https://arxiv.org/abs/1704.04861) — Depthwise-separable conv の提案
- ConvNeXt: [Liu et al., 2022](https://arxiv.org/abs/2201.03545) — Conv だけで Transformer に匹敵
- One TTS Alignment: [Badlani et al., 2021](https://arxiv.org/abs/2108.10447) — 自己アライメント手法

## piper / piper-plus との比較

| | piper (VITS) | piper-plus (MB-ISTFT-VITS) | MiniSpeech |
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

## 自己アライメント

テキストと音声の時間的な対応 (継続長) を、学習中に自動で獲得します (`--learn-alignment`)。

手法は [One TTS Alignment (Badlani et al., 2021)](https://arxiv.org/abs/2108.10447) に基づきます:
AlignmentEncoder (cosine 類似度) で soft alignment を求め、
CTC 損失で単調性を誘導し、MAS で hard な継続長を抽出します。
アライメント用の重みは学習時のみ使用し、推論時は Duration Predictor だけで動きます。
