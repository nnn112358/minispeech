#!/bin/bash
# Encoder dim sweep: d=192, d=128, d=96 (all enc=4 dec=4, 2000 epochs)
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
MANIFEST=fs_data/fs_manifest.json

for DIM in 192 128 96; do
    OUT="fs/enc_d${DIM}"
    LOG="fs/enc_d${DIM}.log"
    echo "=== dim=${DIM} ==="
    mkdir -p "$OUT"
    $PY src/cli/train_minispeech.py \
        --manifest "$MANIFEST" \
        --out "$OUT" \
        --epochs 2000 \
        --batch-size 16 \
        --lr 3e-4 \
        --cosine \
        --save-every 500 \
        --dim "$DIM" \
        2>&1 | tee "$LOG"
    echo "=== dim=${DIM} DONE ==="
done
echo "ALL DONE"
