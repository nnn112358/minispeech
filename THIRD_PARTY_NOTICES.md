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

## SqueezeWave / WaveGlow — `src/sqzw/model.py`
- SqueezeWave: https://github.com/tianrengao/SqueezeWave
- Derived from NVIDIA WaveGlow (https://github.com/NVIDIA/waveglow), BSD-3-Clause.
  The original NVIDIA copyright notice is retained in the file header.
- Modified for this project: `torch.qr` -> `torch.linalg.qr` (torch 2.x), WN mel
  conditioning via `F.interpolate` (any mel-frame:audio-group ratio), and optional
  WaveNet-style dilation.

## HiFi-GAN — `src/sqzw/hifigan_gen.py`
- Architecture from HiFi-GAN (Kong et al., NeurIPS 2020); reference implementation
  https://github.com/jik876/hifi-gan (MIT License). This file is an independent
  re-implementation of the V1/V2 generator for comparison; the MPD/MRD
  discriminators and losses used to train it are reused from the vendored Vocos.

## Piper — `src/sqzw/mel.py`
- `spectrogram_torch` vendored from https://github.com/rhasspy/piper (MIT License),
  so the MiniSpeech-output mel and the vocoder-input mel are bit-identical.

## Starting point
- The SqueezeWave integration was bootstrapped from
  https://github.com/alokprasad/fastspeech_squeezewave (only SqueezeWave was used;
  the acoustic model MiniSpeech is an independent implementation).

## Data
- Training uses the JSUT corpus (https://sites.google.com/site/shinnosuketakamichi/publication/jsut),
  CC-BY-SA 4.0. Data and trained checkpoints are NOT included in this repository.
