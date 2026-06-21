# fast-jtts

A lightweight, **NPU-friendly two-stage Japanese TTS** targeting edge deployment
(AX650N). Non-autoregressive throughout: **FastSpeech** acoustic model →
**SqueezeWave** vocoder, with **Vocos** and **HiFi-GAN** as comparison baselines.

On the Axera **AX650N NPU** (Pulsar2, U16), the SqueezeWave vocoder runs at
**10.7 ms per utterance on NPU3** (277× real-time) — matching Vocos while using
10× fewer parameters.

> Trained data and checkpoints are **not** included (see [Data](#data)); the repo
> contains source, run scripts, and Pulsar2 config templates.

## Why two-stage + SqueezeWave

| Vocoder | Type | NPU fit |
|---|---|---|
| LPCNet / WaveRNN | autoregressive | ✗ sequential, poor NPU fit |
| Vocos / HiFi-GAN | parallel (iSTFT/GAN) | ✓ but heavier |
| **SqueezeWave** (this repo) | **parallel normalizing flow** | ✓ small & fully parallel |

The target is the **edge NPU** (a single accelerator). There, a small parallel
flow beats the larger Vocos; on multi-core CPU the gap narrows (Vocos parallelizes
better), so per-config trade-offs are documented below.

## Architecture

```
text → OpenJTalk g2p → [FastSpeech]  text → 80-dim log-mel
                     → [SqueezeWave] mel  → 22.05 kHz waveform
```

Both stages share the exact same mel (`PiperMelFeatures`: 22050/n_fft1024/hop256/
win1024, center=False, log-clamp 1e-5), so the FastSpeech output and vocoder input
are bit-identical.

**FastSpeech** is a minimal non-AR acoustic model (depthwise-separable conv blocks,
duration predictor only — no pitch/energy). It supports **self-alignment**
(`--learn-alignment`): AlignmentEncoder + CTC forward-sum loss + MAS, so training
needs no external aligner (no VITS, no MFA). Aligner weights are stripped at save
time; inference uses only the learned duration predictor.

## Repository layout

```
src/
  sqzw/        core library  (import: `from sqzw.X import Y`)
    model.py        SqueezeWave model (WaveGlow-derived; interpolate-conditioning, dilation)
    features.py     PiperMelFeatures (shared mel contract)
    mel.py          spectrogram_torch (vendored from piper)
    flow.py         WavSet, diff_infer (differentiable reverse flow), mrstft_loss
    gan_train.py    shared GAN discriminator/generator loss computation
    eval_vocoder.py shared copy-synthesis evaluation loop
    alignment.py    AlignmentEncoder, ForwardSumLoss, beta-binomial prior, MAS
    monotonic_align.py  numba-accelerated Monotonic Alignment Search
    onnx_export.py  ONNX export wrapper (noise as input z, for NPU)
    bench.py        ONNX Runtime CPU latency helper
    vocos_gen.py    Vocos Generator (comparison baseline)
    hifigan_gen.py  HiFi-GAN generator (piper config default: 256ch, ResBlock2)
  vocos/       vendored Vocos (trimmed; MIT — see THIRD_PARTY_NOTICES.md)
  cli/         training and inference entry points (standalone argparse scripts)
  tools/       dev-only analysis: bench_*, sweep_*, plot_*, diag_*
scripts/       run_*.sh pipelines (run from repo root)
npu/           Pulsar2 config templates (*.json); build artifacts are gitignored
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # install a CUDA torch build separately for GPU training
```

NPU conversion (optional) uses the `pulsar2:6.0` Docker image and an Axera AX650 host.

## Usage

Run from the repo root (CLI scripts add `src/` to the path automatically):

```bash
# --- Encoder (FastSpeech) ---
# Self-aligning (recommended — no external aligner needed):
python src/cli/train_fastspeech.py --manifest fs_data/jsut_manifest.json --learn-alignment --epochs 500
# With precomputed durations:
python src/cli/train_fastspeech.py --manifest fs_data/fs_manifest.json --epochs 500

# --- Vocoders ---
# SqueezeWave: full quality recipe (NLL → aux → GAN → BN-recal → eval)
bash scripts/run_gan_finish.sh

# Individual stages:
python src/cli/train.py --filelist data/filelist_train.txt --out checkpoints/run1
python src/cli/finetune_aux.py --init-from checkpoints/run1/sqzw_last.pth
python src/cli/finetune_gan.py --init-from checkpoints/run1/sqzwaux_last.pth --w-mel 45
python src/cli/bake_bnrecal.py --ckpt checkpoints/run1/sqzwgan_last.pth

# HiFi-GAN (piper config, GAN from scratch):
python src/cli/hifigan_train.py --filelist data/filelist_train.txt --out checkpoints/hifigan_jsut
# Vocos (GAN from scratch):
python src/cli/vocos_train.py --filelist data/filelist_train.txt --out checkpoints/vocos_jsut

# --- Inference & evaluation ---
python src/cli/infer.py --ckpt checkpoints/<run>/sqzwgan_bnrecal.pth --n 4 --sigma 0.7
python src/cli/eval_stft.py --ckpt checkpoints/<run>/sqzwgan_bnrecal.pth
python src/cli/eval_vocos.py --ckpt data/ckpts/vocos_last.pth
python src/cli/eval_hifigan.py --ckpt checkpoints/hifigan_jsut/hifigan_last.pth

# Full TTS (phoneme_ids → FastSpeech mel → vocoder audio):
python src/cli/synth.py --fs-ckpt fs/jsut_align/fs_500.pth --voc-ckpt checkpoints/<run>/sqzwgan_bnrecal.pth

# --- ONNX export & NPU ---
python src/cli/export_onnx.py
python src/cli/export_npu.py
bash scripts/run_npu_bench.sh

# --- Benchmarks ---
python src/tools/bench_vocoders.py     # compare vocoder ONNX speed
```

## Fine-tuning to a new speaker

The **voice is determined by the encoder** (FastSpeech), not the vocoder.
Speaker adaptation = encoder fine-tune + optional vocoder fine-tune for recording
texture.

```bash
# 1. Extract durations from a teacher model (or use --learn-alignment)
# 2. Fine-tune FastSpeech on target speaker:
python src/cli/train_fastspeech.py --resume fs/jsut_align/fs_500.pth \
    --manifest fs_data/<spk>/fs_manifest.json --epochs 300 --cosine --lr 2e-4

# 3. (Optional) Fine-tune vocoder for recording texture:
bash scripts/run_tyc_finetune.sh   # worked example on Tsukuyomi corpus
```

## The quality recipe

Pure flow maximum-likelihood (NLL) plateaus well below Vocos. Quality is restored
in stages (`scripts/run_gan_finish.sh`):

1. **NLL pre-training** (`cli/train.py`) — every architecture param is a CLI flag.
2. **Aux fine-tune** (`cli/finetune_aux.py`) — a *differentiable* reverse flow
   (`diff_infer`) generates audio from the GT mel; mel-L1 + multi-resolution STFT
   losses pull it toward the target (the Vocos mel-loss principle).
3. **GAN fine-tune** (`cli/finetune_gan.py`, `w_mel=45` anchor) — Vocos MPD/MRD
   discriminators remove the *raw-waveform HF jitter* ("割れ") that magnitude
   metrics don't capture (judge by ear). Discriminators are **training-only** —
   zero inference cost.
4. **BatchNorm recalibration** (`cli/bake_bnrecal.py`) — re-estimate BN running
   stats on the inference (reverse-flow) distribution, removing a train/eval gap.

## Results

### ONNX Runtime CPU (2.97 s audio)

| Model | Params | 1-thread | All-thread |
|---|---|---|---|
| HiFi-GAN (piper) | 1.46M | 147.5ms (20×RT) | 68.2ms (44×RT) |
| Vocos | 13.50M | 74.2ms (40×RT) | 21.8ms (136×RT) |
| SqueezeWave a256 | 7.87M | 44.0ms (67×RT) | 27.5ms (108×RT) |
| SqueezeWave c64 | 1.35M | **7.2ms (411×RT)** | **5.4ms (553×RT)** |

### AX650N NPU (real measurements, pyaxengine/AXCLRT)

| Config | NPU3 pipeline | Speed |
|---|---|---|
| SqueezeWave c64 | 10.7 ms | 277×RT |
| SqueezeWave c64 (NPU1) | 21.1 ms | 141×RT |

## Data

Training uses the **JSUT** single-speaker Japanese corpus (CC-BY-SA 4.0), not
included here. Place 22.05 kHz wavs under `data/wavs/` and list them in
`data/filelist_{train,val}.txt` (absolute or repo-relative paths).

For FastSpeech, prepare a JSON manifest (`fs_data/jsut_manifest.json`): array of
`{phoneme_ids, mel (npy path), n_frames}`. Durations are optional when using
`--learn-alignment`.

## License & attribution

MIT (see [LICENSE](LICENSE)). This project vendors / derives from Vocos, SqueezeWave
(NVIDIA WaveGlow, BSD-3), HiFi-GAN, and piper — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
See [PLAN.md](PLAN.md) for design notes and open tasks.
