#!/bin/bash
# d=256 training on JSUT (same conditions as sweep: 2000ep, bs16, cosine)
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
MANIFEST=fs_data/fs_manifest.json
DIM=256
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
