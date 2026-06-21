# fast-jtts

A lightweight, **NPU-friendly neural vocoder** for a two-stage Japanese TTS
(FastSpeech acoustic model → vocoder), built around **SqueezeWave** — a
non-autoregressive normalizing-flow vocoder — as a drop-in alternative to Vocos.

On the Axera **AX650N NPU** (Pulsar2, U16), the small SqueezeWave vocoder runs at
an estimated **0.83 ms / utterance — ~5.5× faster than Vocos** at comparable
intelligibility, with a quality-restoration recipe that closes the gap to GAN
vocoders.

> Trained data and checkpoints are **not** included (see [Data](#data)); the repo
> contains source, run scripts, and Pulsar2 config templates.

## Why SqueezeWave

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
win1024, center=False, log-clamp), so the FastSpeech output and vocoder input are
bit-identical.

## Repository layout

```
src/
  sqzw/        core library  (import: `from sqzw.X import Y`)
    model.py        SqueezeWave model (WaveGlow-derived; interpolate-conditioning, dilation)
    mel.py          spectrogram_torch (vendored from piper)
    features.py     PiperMelFeatures
    flow.py         WavSet, diff_infer (differentiable reverse flow), mrstft_loss
    onnx_export.py  ONNX export wrapper (noise as input z, for NPU)
    bench.py        ONNX Runtime CPU latency helper
    vocos_gen.py    Vocos Generator (comparison baseline)
  vocos/       vendored Vocos (trimmed; MIT — see THIRD_PARTY_NOTICES.md)
  cli/         core pipeline scripts: train, finetune_aux, finetune_gan,
               bake_bnrecal, infer, eval_stft, eval_vocos, export_onnx,
               export_npu, train_fastspeech, vocos_train
  tools/       dev / analysis utilities: bench_*, sweep_*, plot_*, diag_*,
               bn_recalib_eval, watch_saturate
scripts/       run_*.sh pipelines (run from repo root)
npu/           Pulsar2 config templates (*.json); build artifacts are gitignored
checkpoints/ data/ fs/ fs_data/ outputs/   # gitignored (your trained models & data)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # install a CUDA torch build separately for GPU training
```

NPU conversion (optional) uses the `pulsar2:6.0` Docker image and an Axera AX650 host.

## Usage

Run from the repo root (scripts resolve their own location; cli scripts add `src/`
to the path automatically):

```bash
# Inference (provide your own trained checkpoint)
python src/cli/infer.py --ckpt checkpoints/<run>/sqzwgan_bnrecal.pth --n 4 --sigma 0.7

# Multi-resolution STFT evaluation vs Vocos
python src/cli/eval_stft.py --ckpt checkpoints/<run>/sqzwgan_bnrecal.pth

# Full training recipe (NLL → aux → GAN → BN-recalibration → eval)
bash scripts/run_gan_finish.sh

# Export ONNX (noise-as-input, NPU-ready) and benchmark on AX650
python src/cli/export_npu.py
bash scripts/run_npu_bench.sh
```

## Fine-tuning to a new speaker

A JSUT-trained vocoder adapts to a new single-speaker corpus with a short GAN
fine-tune. `scripts/run_tyc_finetune.sh` is a worked example on the Tsukuyomi
corpus (つくよみちゃん, VOICEACTRESS100, ~11 min), fine-tuning all three vocoders
sequentially on one GPU:

```bash
# 1. Resample your wavs to 22050 Hz mono and list them:
#      data/<spk>_wavs/*.wav  ->  data/filelist_<spk>_{train,val}.txt
# 2. Fine-tune Vocos + SqueezeWave a128_c128 + c64_f8_l4 (~2000 steps each):
bash scripts/run_tyc_finetune.sh
```

- **SqueezeWave** — `cli/finetune_gan.py --init-from <jsut_ckpt> --filelist <spk>`:
  GAN fine-tune (`w_mel=45`) → BN recalibration → sample.
- **Vocos** — `cli/vocos_train.py --resume <jsut_ckpt> --filelist <spk>`: load the
  pretrained generator and GAN fine-tune.

~2000 steps adapts the voice; results land in `checkpoints/<spk>_*` and sample
wavs in `outputs/<spk>_*.wav`.

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

## Results (estimates)

NPU (AX650N, NPU3, U16; compiler-estimated cycles / 1 GHz):

| Config | params | vocoder | vs Vocos |
|---|---|---|---|
| `a256_c128` (quality) | 7.9M | 3.87 ms | 1.18× |
| `c64_f8_l4` (speed)   | 1.35M | **0.83 ms** | **5.5×** |
| Vocos (baseline)      | 13.5M | 4.56 ms | 1.0× |

Speedup comes from shrinking the architecture; the aux+GAN+BN recipe restores
quality. The FastSpeech encoder (~11 ms @ 1 thread CPU) becomes the bottleneck
once the vocoder is small.

## Data

Training uses the **JSUT** single-speaker Japanese corpus (CC-BY-SA 4.0), not
included here. Place 22.05 kHz wavs under `data/wavs/` and list them in
`data/filelist_{train,val}.txt` (absolute or repo-relative paths).

## License & attribution

MIT (see [LICENSE](LICENSE)). This project vendors / derives from Vocos, SqueezeWave
(NVIDIA WaveGlow, BSD-3), and piper — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
See [PLAN.md](PLAN.md) for design notes and open tasks.
