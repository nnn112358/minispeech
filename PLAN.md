# fast-jtts — 設計ノート・進捗

対象: JSUT 単一話者 22050Hz / 比較ボコーダ: Vocos・HiFi-GAN
構成・実行方法は [README.md](README.md) 参照。本書は設計判断・進捗・残課題。

## アーキテクチャ

```
テキスト → OpenJTalk g2p → [FastSpeech lightning] → 80-mel → [SqueezeWave] → 22050Hz 波形
```

- 共通 mel: `PiperMelFeatures` (sr22050 / n_fft1024 / hop256 / win1024, center=False, log-clamp 1e-5)。音響モデルの出力とボコーダの入力が完全一致。
- 音響モデル: FastSpeech lightning (非AR, DurationPredictor のみ, pitch/energy なし)。`src/cli/train_fastspeech.py`。
- ボコーダ: SqueezeWave (WaveGlow 系の正規化フロー, 非AR・並列)。`src/sqzw/model.py`。

## 設計判断 (なぜ SqueezeWave)

- 自己回帰型ボコーダ (LPCNet 系) は逐次生成で遅い → 並列フローの SqueezeWave を採用
- SqueezeWave はチャネル数・フロー段数・レイヤ数を CLI フラグで自由に変更でき、速度と品質のトレードオフを探りやすい
- 通常の最尤学習 (NLL) だけでは品質が不足するため、4段学習で改善する

## 4段学習 ★本プロジェクトの主成果

通常の NLL 学習だけでは Vocos / HiFi-GAN に品質で劣る → 4段階で改善:

1. **NLL 事前学習** (`cli/train.py`) — アーキテクチャパラメータはすべて CLI フラグ
2. **Aux fine-tune** (`cli/finetune_aux.py`) — 微分可能逆フロー `diff_infer` で GT mel から音声を生成し、mel-L1 + マルチ解像度 STFT 損失で追加学習
3. **GAN fine-tune** (`cli/finetune_gan.py`, `w_mel=45`) — Vocos 判別器 (MPD/MRD) で、スペクトル指標に出ない音質の粗さ ("割れ") を除去。判別器は学習時のみ使用
4. **BN 再キャリブ** (`cli/bake_bnrecal.py`) — 推論時の分布で BN 統計を再計算し、学習/推論のずれを解消

## 自己アライメント ★VITS 依存除去

FastSpeech lightning の継続長取得に VITS MAS が必要だった依存を完全除去。

- **AlignmentEncoder** (cosine 類似度 + 学習可能 scale) + **ForwardSumLoss** (CTC) + **beta-binomial 事前分布** (対角線誘導、40% で消す) + **MAS** (hard 継続長)
- `--learn-alignment` で学習。aligner 層は保存時に除外 (推論は DurationPredictor のみ)
- **注意点**: (1) bin 損失はアライメントを破壊する (`--bin-weight 0` 必須) (2) beta-binomial 事前分布なしだと収束しない (3) cosine + learnable scale が注意先鋭化に必須
- JSUT 5000 発話: mel_l1 **0.564** (500ep)。tyc 100 発話: mel_l1 0.495

## 比較ボコーダ

HiFi-GAN (VITS TTS 標準構成: ResBlock2, 256ch) と Vocos (ConvNeXt + iSTFT) を同一データ・同一判別器で訓練:

- 共有モジュール: `sqzw/gan_train.py` (D/G ロス計算)、`sqzw/eval_vocoder.py` (評価ループ)
- 学習: `cli/hifigan_train.py`、`cli/vocos_train.py`
- 評価: `cli/eval_hifigan.py`、`cli/eval_vocos.py`

## 速度計測結果

### ONNX Runtime CPU (2.97 秒の音声を生成する時間)

| モデル | パラメータ数 | 1スレッド | 全スレッド |
|---|---|---|---|
| HiFi-GAN (piper) | 1.46M | 147.5ms (20倍RT) | 68.2ms (44倍RT) |
| Vocos | 13.50M | 74.2ms (40倍RT) | 21.8ms (136倍RT) |
| SqueezeWave a256 | 7.87M | 44.0ms (67倍RT) | 27.5ms (108倍RT) |
| SqueezeWave c64 | 1.35M | **7.2ms (411倍RT)** | **5.4ms (553倍RT)** |

### AX650N NPU 実機計測 (参考, pyaxengine/AXCLRT)

| 構成 | NPU3 pipeline | 速度 |
|---|---|---|
| SqueezeWave c64 | 10.7 ms | 277倍RT |
| SqueezeWave c64 (NPU1) | 21.1 ms | 141倍RT |

## 成果物

- チェックポイント: `checkpoints/ckpts_{a256_c128_q,c64_f8_l4_s}_gan/sqzwgan_bnrecal.pth`
- ONNX: `src/cli/export_onnx.py` で出力
- 一括レシピ: `scripts/run_gan_finish.sh`

## 完了項目

- [x] SqueezeWave 4段学習 (NLL → aux → GAN → BN再キャリブ) 確立
- [x] SqueezeWave a256_c128 / c64_f8_l4 の品質・速度両立
- [x] 比較ボコーダ: Vocos standalone trainer (`cli/vocos_train.py`)
- [x] 比較ボコーダ: HiFi-GAN VITS TTS 標準構成 (`cli/hifigan_train.py`)
- [x] FastSpeech lightning 自己アライメント (`--learn-alignment`, VITS 依存除去)
- [x] つくよみちゃん話者適応 (encoder fine-tune + vocoder fine-tune)
- [x] ONNX CPU ベンチマーク (4ボコーダ比較)
- [x] コードリファクタリング: GAN D/G step 共通化、eval 共通化、import 整理
- [x] フル TTS 推論パイプライン (`cli/synth.py`)

## 残課題

- [ ] JSUT 全ボコーダ学習 (HiFi-GAN 50k / Vocos resume 40k+10k) — 実行中
- [ ] 音響モデル高速化: ボコーダ縮小で FastSpeech lightning (11ms) が律速に
- [ ] (任意) c64_f8_l8 等の中間構成を GAN 仕上げ / より攻めた構成
- 中間・探索チェックポイントは `_unused/` に隔離 (復元可)
