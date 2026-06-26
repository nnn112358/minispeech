#!/bin/bash
set -euo pipefail
PY=~/.venvs/piper_ft/bin/python
FL=data/filelist_train.txt
STEPS=10000
SAVE=2000
BS=16
cd /home/nnn/mini-jtts

echo "=== [1/3] HiFi-GAN (resume from step 5000) ==="
stdbuf -oL $PY -u src/cli/hifigan_train.py \
  --filelist $FL --out checkpoints/hifigan_jsut \
  --max-steps $STEPS --batch-size $BS --save-every $SAVE \
  --resume checkpoints/hifigan_jsut/hifigan_last.pth

echo "=== [2/3] MB-iSTFT (resume from step 500) ==="
stdbuf -oL $PY -u src/cli/mbistft_train.py \
  --filelist $FL --out checkpoints/mbistft_jsut \
  --max-steps $STEPS --batch-size $BS --save-every $SAVE \
  --resume checkpoints/mbistft_jsut/mbistft_last.pth

echo "=== [3/3] Vocos (resume from step 2000) ==="
stdbuf -oL $PY -u src/cli/vocos_train.py \
  --filelist $FL --out checkpoints/vocos_jsut \
  --max-steps $STEPS --batch-size $BS --save-every $SAVE \
  --resume checkpoints/tyc_vocos/vocos_last.pth

echo "=== ALL DONE ==="
