# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Two-stage non-autoregressive Japanese TTS:
**OpenJTalk g2p → MiniSpeech (text→mel) → SqueezeWave vocoder (mel→audio)**

All stages share identical mel features (`PiperMelFeatures`: sr=22050, n_fft=1024, hop=256, win=1024, center=False, log-clamp 1e-5). MiniSpeech output and vocoder input are bit-identical by design.

## Running scripts

All scripts assume **repo root** as cwd. CLI scripts auto-add `src/` to sys.path.

```bash
# Training (full quality recipe)
bash scripts/run_gan_finish.sh        # NLL → aux → GAN → BN-recal → eval

# Individual stages
python src/cli/train.py --filelist data/filelist_train.txt --out checkpoints/run1
python src/cli/finetune_aux.py --ckpt checkpoints/run1/last.pth
python src/cli/finetune_gan.py --ckpt checkpoints/run1/aux_last.pth --w-mel 45
python src/cli/bake_bnrecal.py --ckpt checkpoints/run1/gan_last.pth

# MiniSpeech acoustic model
python src/cli/train_minispeech.py --manifest fs_data/tyc/fs_manifest.json --learn-alignment
python src/cli/train_minispeech.py --manifest fs_data/tyc/fs_manifest.json  # with precomputed durations

# Inference & evaluation
python src/cli/infer.py --ckpt checkpoints/<run>/sqzwgan_bnrecal.pth --n 4 --sigma 0.7
python src/cli/eval_stft.py --ckpt checkpoints/<run>/sqzwgan_bnrecal.pth
python src/cli/synth.py --fs-ckpt fs/tyc_fs/fs_300.pth --voc-ckpt checkpoints/<run>/sqzwgan_bnrecal.pth

# ONNX export & NPU
python src/cli/export_onnx.py
python src/cli/export_npu.py
bash scripts/run_npu_bench.sh

# Benchmarks & tools
python src/tools/bench_vocoders.py     # compare vocoder ONNX speed
python src/tools/bench_onnx.py         # single-model ONNX latency
```

## Code layout

- `src/sqzw/` — core library (model, features, flow ops, ONNX export, alignment, benchmarking, `gan_train.py` shared GAN D/G loss, `eval_vocoder.py` shared eval loop)
- `src/cli/` — training and inference entry points (each is a standalone script with argparse)
- `src/tools/` — dev-only analysis, plotting, sweeps, diagnostics
- `src/vocos/` — vendored Vocos (trimmed, MIT); discriminators used for GAN training only
- `scripts/` — bash pipelines that compose CLI scripts into full recipes
- `npu/` — Pulsar2 config templates (build artifacts gitignored)
- `_unused/`, `_legacy/` — archived experiments (not part of active codebase)

## Key architecture details

- **SqueezeWave** (`sqzw/model.py`): WaveGlow-derived normalizing flow. All architecture params (n_audio_channel, n_flows, n_layers, etc.) are CLI flags, not hardcoded.
- **Quality recipe**: NLL pre-train → aux fine-tune (differentiable reverse flow + mel-L1 + multi-res STFT) → GAN fine-tune (Vocos MPD/MRD, `w_mel=45` anchor) → BN recalibration. Discriminators are training-only (zero inference cost).
- **MiniSpeech** (`cli/train_minispeech.py`): minimal non-AR acoustic model (depthwise-separable conv blocks, duration predictor only — no pitch/energy). Supports two alignment modes: `--learn-alignment` (self-aligning via AlignmentEncoder + CTC + MAS, no external dependency) or precomputed durations in manifest.
- **Self-alignment** (`sqzw/alignment.py`, `sqzw/monotonic_align.py`): cosine-similarity attention with learnable scale + beta-binomial diagonal prior + forward-sum (CTC) loss. MAS extracts hard durations. Aligner weights are stripped at save time (inference uses only the duration predictor).
- **HiFi-GAN** (`sqzw/hifigan_gen.py`): defaults to piper config (ResBlock2, 256ch, upsample 8×8×4). Used as a comparison baseline, not the primary vocoder.
- **PiperMelFeatures** (`sqzw/features.py`): the shared mel contract. Any change here breaks compatibility between MiniSpeech and all vocoders.

## Conventions

- No build system (no setup.py/pyproject.toml). Plain `pip install -r requirements.txt` + CUDA torch installed separately.
- ONNX export wraps noise `z` as an input (not sampled internally) for NPU determinism.
- Checkpoint format: `{"model": state_dict, "config": arch_params, ...}` for SqueezeWave; `{"model": state_dict, "n_sym": int, "epoch": int}` for MiniSpeech.
- Data filelists are newline-separated absolute or repo-relative paths to 22.05 kHz mono wavs.
- MiniSpeech manifest: JSON array of `{phoneme_ids, durations (optional), mel (npy path), n_frames}`.
- BN recalibration (`bake_bnrecal.py`) is mandatory after GAN fine-tune — without it, train/eval distribution mismatch causes audible artifacts.

## Gotchas

- `w_mel=45` in GAN fine-tune is a tuned anchor — changing it risks quality regression.
- Self-alignment `--bin-weight` must stay 0 (binarization loss destabilizes alignment; MAS already gives hard durations).
- Beta-binomial prior is annealed to zero by 40% of training (`prior_w = max(0, 1 - ep/(epochs*0.4))`).
- The vendored `src/vocos/` is a trimmed fork (no HuggingFace, no EnCodec). Don't update from upstream without checking removals.
