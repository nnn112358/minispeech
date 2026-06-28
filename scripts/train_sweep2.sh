#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
MANIFEST=fs_data/fs_manifest.json

echo "=== Waiting for d=256 (PID $1) to finish ==="
if [ -n "$1" ]; then
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
    echo "d=256 finished"
fi

echo "=== [1/2] d=384 enc=4 dec=4 (6.88M) ==="
$PY src/cli/train_minispeech.py \
    --manifest "$MANIFEST" --out fs/enc_d384 \
    --epochs 2000 --batch-size 16 --lr 3e-4 --cosine \
    --save-every 500 --dim 384 --n-enc 4 --n-dec 4 \
    2>&1 | tee fs/enc_d384.log

echo "=== [2/2] d=256 enc=6 dec=6 (4.40M) ==="
$PY src/cli/train_minispeech.py \
    --manifest "$MANIFEST" --out fs/enc_e6d6 \
    --epochs 2000 --batch-size 16 --lr 3e-4 --cosine \
    --save-every 500 --dim 256 --n-enc 6 --n-dec 6 \
    2>&1 | tee fs/enc_e6d6.log

echo "=== All sweep2 done ==="
