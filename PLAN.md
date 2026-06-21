# fast-jtts — エッジ向け日本語2段TTS 計画・状況

対象: JSUT 単一話者 22050Hz / 比較基準: Vocos・HiFi-GAN / NPU: AX650N (Pulsar2 U16)
構成・実行方法は [README.md](README.md) 参照。本書は設計判断・進捗・残課題。

## アーキテクチャ
```
テキスト → OpenJTalk g2p → 【音響モデル: FastSpeech】 text→80-mel
                          → 【ボコーダ: SqueezeWave】  mel→22050Hz 波形
```
- 共通mel: `PiperMelFeatures`(sr22050/n_fft1024/hop256/win1024, center=False, log-clamp 1e-5)。前段出力と後段入力が完全一致。
- 音響モデル: 自作の最小FastSpeech(非AR, DurationPredictorのみ。FS2のpitch/energy adaptorは無し)。`src/cli/train_fastspeech.py`。
- ボコーダ: **SqueezeWave**(WaveGlow系の正規化フロー, 非AR・並列)。`src/sqzw/model.py`。

## 設計判断の核 (なぜSqueezeWave)
- **自己回帰型ボコーダ(LPCNet系)はNPU不向き**(逐次生成で並列演算と相性が悪い)→ 並列フローのSqueezeWaveを採用。
- 土俵はエッジNPU(単一アクセラレータ)。CPUマルチスレッドではVocosの並列性が効くが、NPU/単スレッドでは小型SqueezeWaveが圧勝。

## 品質回復レシピ ★本プロジェクトの主成果
純flow-NLLはVocosに品質で大差→4段で回復:
1. **NLL事前学習** (`cli/train.py`, 全アーキparam選択可)
2. **aux損失fine-tune** (`cli/finetune_aux.py`): 微分可能逆flow`diff_infer`でGT mel条件付け生成→mel-L1+マルチ解像度STFT損失
3. **GAN仕上げ** (`cli/finetune_gan.py`, **w_mel=45**でアンカー): Vocos判別器で割れ(=フローraw波形のHFジッタ)を除去。**割れは磁度指標に出ず耳で判定**(知覚-歪みギャップ)
4. **BN再キャリブ** (`cli/bake_bnrecal.py`): 推論分布でBN統計を取り直し(train/eval不整合の天井を解消)

## 自己アライメント ★VITS依存除去
FastSpeechの継続長取得にVITS MASが必要だった依存を完全除去。
- **AlignmentEncoder**(cosine類似度+学習可能scale) + **ForwardSumLoss**(CTC) + **beta-binomial事前分布**(対角線誘導、40%で消す) + **MAS**(hard継続長)
- `--learn-alignment` で学習。aligner層は保存時に除外(推論はDurationPredictorのみ)。
- **罠**: ①bin損失はアライメント破壊(`--bin-weight 0`必須) ②beta-binomial事前分布なしだと収束しない ③cosine+learnable scaleが注意先鋭化に必須
- JSUT 5000発話: mel_l1 **0.569** (500ep)。tyc 100発話: mel_l1 0.495。公開可能な自己完結学習。

## 比較ボコーダ
HiFi-GAN(piper config: ResBlock2, 256ch, upsample 8×8×4)とVocos(ConvNeXt+iSTFT)を同一データ・同一判別器で訓練し公正比較:
- 共有: `sqzw/gan_train.py`(D/Gロス計算)、`sqzw/eval_vocoder.py`(copy-synthesis評価ループ)
- `cli/hifigan_train.py`、`cli/vocos_train.py`、`cli/eval_hifigan.py`、`cli/eval_vocos.py`

## 結果

### ONNX Runtime CPU (2.97s音声)
| 構成 | params | 1スレ | 全スレ |
|---|---|---|---|
| HiFi-GAN (piper) | 1.46M | 147.5ms (20×RT) | 68.2ms (44×RT) |
| Vocos | 13.50M | 74.2ms (40×RT) | 21.8ms (136×RT) |
| SqueezeWave a256 | 7.87M | 44.0ms (67×RT) | 27.5ms (108×RT) |
| SqueezeWave c64 | 1.35M | **7.2ms (411×RT)** | **5.4ms (553×RT)** |

### AX650N NPU 実機ベンチ (pyaxengine/AXCLRT)
| 構成 | NPU3 pipeline | 速度 |
|---|---|---|
| SqueezeWave c64 | 10.7 ms | 277×RT |
| SqueezeWave c64 (NPU1) | 21.1 ms | 141×RT |

推定4.9msは純計算のみ、実測10.7msはPCIe DMA込み。

## デプロイ用成果物
- ckpt: `checkpoints/ckpts_{a256_c128_q,c64_f8_l4_s}_gan/sqzwgan_bnrecal.pth`
- ONNX: `npu/model/sqzw_*.onnx` / axmodel: `npu/output_sqzw_{a256,c64}/*.axmodel`(NPU3/AX650/U16)
- 一括レシピ: `scripts/run_gan_finish.sh`

## 完了項目
- [x] SqueezeWave NLL→aux→GAN→BN recal 品質回復レシピ確立
- [x] SqueezeWave a256_c128 / c64_f8_l4 の品質/速度両立
- [x] 比較ボコーダ: Vocos standalone trainer (`cli/vocos_train.py`)
- [x] 比較ボコーダ: HiFi-GAN piper config (`cli/hifigan_train.py`)
- [x] FastSpeech 自己アライメント (`--learn-alignment`, VITS依存除去)
- [x] つくよみちゃん話者適応 (encoder fine-tune + vocoder fine-tune)
- [x] ONNX CPU ベンチマーク (4ボコーダ比較)
- [x] AX650N NPU 実機ベンチマーク (pyaxengine)
- [x] コードリファクタリング: GAN D/G step共通化、eval共通化、import整理
- [x] フルTTS推論パイプライン (`cli/synth.py`)

## 残課題
- [ ] JSUT全ボコーダ学習 (HiFi-GAN 50k / Vocos resume 40k+10k) — 実行中
- [ ] エンコーダ高速化: ボコーダ縮小でFastSpeech(11ms)が律速に。次の速度ゲインは前段
- [ ] (任意) c64_f8_l8等の中間構成をGAN仕上げ / より攻めた構成
- 中間/探索ckptは `_unused/` に隔離(復元可)。再開時はそこから。
