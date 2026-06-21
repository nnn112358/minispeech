#!/bin/bash
export PYTHONPATH=""
export CUDA_VISIBLE_DEVICES=0
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec ${PYTHON:-python} -u src/cli/train_minispeech.py \
  --epochs 500 --batch-size 32 --lr 5e-4 --save-every 25 \
  --out fs
