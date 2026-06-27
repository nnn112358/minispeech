# Third-Party Notices

This project vendors / derives from the following open-source software.

## Vocos — `src/vocos/`
- Source: https://github.com/charactr-platform/vocos (MIT License)
- License text: `src/vocos/LICENSE`
- Used as the comparison-baseline vocoder (VocosBackbone + ISTFTHead) and for its
  discriminators (MPD/MRD) and losses during GAN fine-tuning.
- Modified: trimmed `__init__.py` (removed pretrained loader) and
  `feature_extractors.py` (removed `EncodecFeatures`) to drop the `encodec` /
  `transformers` / `huggingface_hub` dependencies, which this project does not use.

## SqueezeWave / WaveGlow — `src/decoders/squeezewave/model.py`
- SqueezeWave: https://github.com/tianrengao/SqueezeWave
- Derived from NVIDIA WaveGlow (https://github.com/NVIDIA/waveglow), BSD-3-Clause.
  The original NVIDIA copyright notice is retained in the file header.
- Modified for this project: `torch.qr` -> `torch.linalg.qr` (torch 2.x), WN mel
  conditioning via `F.interpolate` (any mel-frame:audio-group ratio), and optional
  WaveNet-style dilation.

## HiFi-GAN — `src/decoders/hifigan/generator.py`
- Architecture from HiFi-GAN (Kong et al., NeurIPS 2020); reference implementation
  https://github.com/jik876/hifi-gan (MIT License). This file is an independent
  re-implementation of the V1/V2 generator for comparison; the MPD/MRD
  discriminators and losses used to train it are reused from the vendored Vocos.

## Piper — `src/common/mel.py`, `src/encoder/monotonic_align.py`
- `spectrogram_torch` vendored from https://github.com/rhasspy/piper (MIT License),
  so the MiniSpeechEncoder-output mel and the vocoder-input mel are bit-identical.
- `monotonic_align.py` — Monotonic Alignment Search (MAS) の動的計画法。
  Glow-TTS / VITS の monotonic_align (MIT License) を piper 経由で vendored。
  自己アライメント (`encoder/alignment.py`) から呼び出し、hard な継続長を抽出する。

## piper-plus (MB-iSTFT-VITS2) — `src/decoders/mb_istft/`
- Source: https://github.com/ahoennecke/piper-plus (MIT License)
- MB-iSTFT decoder architecture from Kawamura et al. (ICASSP 2023),
  vendored from piper-plus's `piper_train/vits/mb_istft.py`, `stft_onnx.py`, `stft_loss.py`.
- PQMF (Pseudo Quadrature Mirror Filterbank) and OnnxISTFT (conv_transpose1d-based iSTFT).
- Modified: extracted as standalone mel-conditioned vocoder (not VITS end-to-end),
  removed speaker conditioning / multistream / cartesian variants, reused ResBlocks
  from common/modules.py.

## Starting point
- The SqueezeWave integration was bootstrapped from
  https://github.com/alokprasad/fastspeech_squeezewave (only SqueezeWave was used;
  the acoustic model MiniSpeechEncoder is an independent implementation).

## Data
- Training uses the JSUT corpus (https://sites.google.com/site/shinnosuketakamichi/publication/jsut),
  CC-BY-SA 4.0. Data and trained checkpoints are NOT included in this repository.
