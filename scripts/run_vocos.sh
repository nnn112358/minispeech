#!/bin/bash
export PYTHONPATH=""
export CUDA_VISIBLE_DEVICES=0
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec ${PYTHON:-python} -u src/cli/vocos_train.py \
  --max-steps 300000 --batch-size 16 --num-samples 16384 --lr 5e-4 --save-every 5000 \
  --out data/ckpts
