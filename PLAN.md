# MiniSpeech — 設計ノート・進捗

対象: JSUT 単一話者 22050Hz。構成・使い方は [README.md](README.md)、
実装ガイドは [CLAUDE.md](CLAUDE.md) を参照。本書は**設計判断・進捗・残課題**を記録する。

## アーキテクチャ

```
テキスト → OpenJTalk g2p → [MiniSpeechEncoder] → 80-mel → [Vocoder] → 22050Hz 波形
```

- 共通 mel: `PiperMelFeatures` (sr22050 / n_fft1024 / hop256 / win1024, center=False, log-clamp 1e-5)。
  音響モデルの出力とボコーダの入力は設計上 bit-identical (`src/common/features.py`)。
- 音響モデル: MiniSpeechEncoder (非AR, Duration 整数展開, pitch/energy なし)。`src/encoder/minispeech.py`。
- ボコーダ (交換可能): **Vocos (デフォルト)** / HiFi-GAN / MB-iSTFT。同一 mel 契約・同一判別器で学習。

## 設計判断

### なぜ 2 段階 (音響モデル + 独立ボコーダ) か
- piper (VITS) はエンドツーエンドで高品質だが ~30M params と重く、構造が複雑
  (Transformer encoder + VAE + Flow + HiFi-GAN decoder の一体型)。
- 「テキスト→mel」と「mel→波形」に分離し、各段を小型化。ボコーダを差し替え可能にして
  Vocos / HiFi-GAN / MB-iSTFT を**同一データ・同一判別器**で比較できるようにした。

### なぜ Self-Attention を Conv に置換するか
- Duration ベースの音響モデルでは依存が局所的 (隣接音素 / 展開後の隣接フレーム) なので、
  depthwise-separable Conv (MobileNet/ConvNeXt 系) で十分。
- 計算量が Self-Attention の O(N²) → O(N) になり、系列長によらず一定コストで動く。
- 4 層積めば実効受容野が系列全体に及ぶ。

### Duration の一本化 ★VITS のバグを構造的に回避
- `round → long` で整数化した duration をそのまま `repeat_interleave` に渡す。
  mel 長 = `sum(durations)` で厳密に一致。
- VITS 系の float(raw w)/ceil(w) 二重定義 (末尾に余分な音声が残る piper-plus issue #268) が
  構造的に発生しない。

## 自己アライメント ★VITS 依存除去

MiniSpeechEncoder の継続長取得に外部 VITS MAS が必要だった依存を完全除去 (`src/encoder/alignment.py`)。

- **AlignmentEncoder** (cosine 類似度 + 学習可能 scale) + **ForwardSumLoss** (CTC) +
  **beta-binomial 事前分布** (対角線誘導、学習の 40% で消す) + **MAS** (hard 継続長)。
- `--learn-alignment` で学習。aligner 層は保存時に除外 (推論は DurationPredictor のみで動く)。
- **注意点**: (1) bin 損失はアライメントを破壊する (`--bin-weight 0` 必須)、
  (2) beta-binomial 事前分布なしだと収束しない、(3) cosine + learnable scale が注意先鋭化に必須。
- JSUT 5000 発話: mel_l1 **0.564** (500ep)。100 発話の話者適応: mel_l1 0.495。

## 比較ボコーダ

Vocos (ConvNeXt + iSTFT, デフォルト) / HiFi-GAN (ResBlock2, 256ch) /
MB-iSTFT (多帯域 iSTFT + PQMF, piper-plus 方式) を同一データ・同一判別器で訓練:

- 共有モジュール: `src/common/gan_train.py` (D/G ロス計算)、`src/common/eval_vocoder.py` (評価ループ)。
- 判別器 (Vocos MPD/MRD) は全ボコーダで共有 (学習時のみ使用、推論コストゼロ)。
- 学習: `src/cli/{vocos,hifigan,mbistft}_train.py`。評価: `src/cli/eval_{vocos,hifigan,mbistft}.py`。

## エンコーダの dim スイープ

| dim | enc/dec | パラメータ数 | mel_l1 @2000ep |
|-----|---------|------------|----------------|
| 256 | 4/4 | 3.08M | (学習中) |
| 192 | 4/4 | 1.75M | 0.496 |
| 128 | 4/4 | 0.79M | 0.565 |
| 96  | 4/4 | 0.45M | (学習中) |

パラメータ数は dim の二乗にほぼ比例 (FFN の d→2d→d が支配的)。学習重みのみの値で、
位置エンコーディングバッファ込みの ONNX サイズ・d64/d384 を含む全構成は
[onnx_v2/README.md](onnx_v2/README.md) を参照 (本表が mel_l1 実測の唯一の基準)。

## 速度計測結果

### ONNX Runtime CPU (2.97 秒の音声を生成する時間)

| モデル | パラメータ数 | 1スレッド | 全スレッド |
|---|---|---|---|
| HiFi-GAN (piper) | 1.46M | 147.5ms (20倍RT) | 68.2ms (44倍RT) |
| Vocos | 13.50M | 74.2ms (40倍RT) | 21.8ms (136倍RT) |
| SqueezeWave a256 *(アーカイブ)* | 7.87M | 44.0ms (67倍RT) | 27.5ms (108倍RT) |
| SqueezeWave c64 *(アーカイブ)* | 1.35M | **7.2ms (411倍RT)** | **5.4ms (553倍RT)** |

### AX650N NPU 実機計測 (参考, pyaxengine/AXCLRT)

| 構成 | NPU3 pipeline | 速度 |
|---|---|---|
| SqueezeWave c64 | 10.7 ms | 277倍RT |
| SqueezeWave c64 (NPU1) | 21.1 ms | 141倍RT |

> SqueezeWave 行は `_unused/squeezewave/` 退避前に測定した参考値。

## 進捗

### 完了
- [x] MiniSpeechEncoder (Duration 整数展開, depthwise-separable Conv)
- [x] MiniSpeechEncoder 自己アライメント (`--learn-alignment`, VITS 依存除去)
- [x] dim スイープ (enc_d{96,128,192,256}) と checkpoint への config 埋め込み
- [x] 比較ボコーダ 3 種 (Vocos / HiFi-GAN / MB-iSTFT) を同一データ・同一判別器で学習
- [x] GAN D/G step・評価ループの共通化 (`common/gan_train.py`, `common/eval_vocoder.py`)
- [x] フル TTS 推論パイプライン (`src/cli/synth.py`, encoder config 自動検出)
- [x] ONNX export (encoder + vocoder, `--slim`/`--fp16`) と CPU ベンチマーク
- [x] つくよみちゃん話者適応 (encoder fine-tune + vocoder fine-tune)

### 残課題
- [ ] JSUT 全ボコーダの長期学習 (HiFi-GAN / Vocos の resume 継続)
- [ ] 音響モデル高速化: ボコーダ縮小により MiniSpeechEncoder が律速になりつつある
- [ ] dim=256 / dim=96 の 2000ep 完走と mel_l1 確定
- 中間・探索チェックポイントは `_unused/` に隔離 (復元可)

## SqueezeWave (アーカイブ済み) ★設計判断

WaveGlow 派生の正規化フローボコーダ。小型・高速 (c64 で CPU 400 倍超 RT) だったが、
学習に 4 段 (NLL 事前学習 → aux FT → GAN FT → BN 再キャリブ) を要し、**単段 GAN で済む
3 種に主役を譲ってアーカイブ**した、という判断。退避先 `_unused/squeezewave/`・復元手順・
論文リンクは [README](README.md) / [CLAUDE.md](CLAUDE.md)、速度値は上記計測表 (参考値)。
