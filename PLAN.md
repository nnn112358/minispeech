# fast-jtts — エッジ向け日本語2段TTS 計画・状況

対象: JSUT 単一話者 22050Hz / 比較基準: Vocos / NPU: AX650N (Pulsar2 U16)
構成・実行方法は [README.md](README.md) 参照。本書は設計判断・進捗・残課題。

## アーキテクチャ
```
テキスト → OpenJTalk g2p → 【音響モデル: FastSpeech】 text→80-mel
                          → 【ボコーダ: SqueezeWave】  mel→22050Hz 波形
```
- 共通mel: `PiperMelFeatures`(sr22050/n_fft1024/hop256/win1024, center=False, log-clamp)。前段出力と後段入力が完全一致。
- 音響モデル: 自作の最小FastSpeech(非AR, DurationPredictorのみ。FS2のpitch/energy adaptorは無し)。`src/cli/train_fastspeech.py`。
- ボコーダ: **SqueezeWave**(WaveGlow系の正規化フロー, 非AR・並列)。`src/sqzw/model.py`。

## 設計判断の核 (なぜSqueezeWave)
- **自己回帰型ボコーダ(LPCNet系)はNPU不向き**(逐次生成で並列演算と相性が悪い)→ 並列フローのSqueezeWaveを採用。
- 土俵はエッジNPU(単一アクセラレータ)。CPUマルチスレッドではVocosの並列性が効くが、NPU/単スレッドでは小型SqueezeWaveが圧勝。

## 品質回復レシピ ★本プロジェクトの主成果
純flow-NLLはVocosに品質で大差→3段で回復:
1. **NLL事前学習** (`cli/train.py`, 全アーキparam選択可)
2. **aux損失fine-tune** (`cli/finetune_aux.py`): 微分可能逆flow`diff_infer`でGT mel条件付け生成→mel-L1+マルチ解像度STFT損失
3. **GAN仕上げ** (`cli/finetune_gan.py`, **w_mel=45**でアンカー): Vocos判別器で割れ(=フローraw波形のHFジッタ)を除去。**割れは磁度指標に出ず耳で判定**(知覚-歪みギャップ)
4. **BN再キャリブ** (`cli/bake_bnrecal.py`): 推論分布でBN統計を取り直し(train/eval不整合の天井を解消)

## 結果
| 構成 | params | NPU3推定 | Vocos比 | 用途 |
|---|---|---|---|---|
| a256_c128 | 7.9M | 3.87ms | 1.18×速 | 品質優先 |
| c64_f8_l4 | 1.35M | 0.83ms | **5.5×速** | 速度優先 |
| Vocos(基準) | 13.5M | 4.56ms | 1.0× | — |
- 品質: aux+BN+GANで a256はVocos同等以上、c64も耳でクリーン化を確認。
- CPU ONNX(2.97s): 1スレでSqueezeWave有利、4スレでVocosが追いつく(エンコーダ床11ms→4.7ms)。

## デプロイ用成果物
- ckpt: `checkpoints/ckpts_{a256_c128_q,c64_f8_l4_s}_gan/sqzwgan_bnrecal.pth`
- ONNX: `npu/model/sqzw_*.onnx` / axmodel: `npu/output_sqzw_{a256,c64}/*.axmodel`(NPU3/AX650/U16)
- 一括レシピ: `scripts/run_gan_finish.sh`

## 残課題
- [ ] NPU実機ベンチ確定 (`scripts/run_npu_bench.sh`。axcl_run_modelがc64でハングする件の解決 → pyaxengine併用も)
- [ ] エンコーダ高速化: ボコーダ縮小でFastSpeech(11ms)が律速に。次の速度ゲインは前段
- [ ] (任意) c64_f8_l8等の中間構成をGAN仕上げ / より攻めた構成 / 正式FastSpeech2化
- 中間/探索ckptは `_unused/` に隔離(復元可)。再開時はそこから。
