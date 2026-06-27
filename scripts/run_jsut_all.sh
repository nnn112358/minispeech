#!/bin/bash
# Full JSUT training pipeline: encoder → HiFi-GAN → Vocos
set -euo pipefail
cd /home/nnn/mini-jtts
PY=~/.venvs/piper_ft/bin/python
LOG=logs/jsut_all_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs fs/jsut_align checkpoints/hifigan_jsut checkpoints/vocos_jsut

exec > >(tee -a "$LOG") 2>&1
echo "=== JSUT full pipeline started $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 1. MiniSpeech encoder (self-aligning) ----
echo ""
echo "=== [1/3] MiniSpeech encoder (self-aligning, 500 epochs) ==="
echo "start: $(date)"
$PY src/cli/train_minispeech.py \
    --manifest fs_data/jsut_manifest.json \
    --learn-alignment \
    --out fs/jsut_align \
    --epochs 500 \
    --batch-size 16 \
    --lr 3e-4 \
    --cosine \
    --save-every 100
echo "MiniSpeech done: $(date)"

# ---- 2. HiFi-GAN (piper config: 256ch, ResBlock2) ----
echo ""
echo "=== [2/3] HiFi-GAN vocoder (piper 256ch, 50k steps) ==="
echo "start: $(date)"
$PY src/cli/hifigan_train.py \
    --filelist data/filelist_train.txt \
    --out checkpoints/hifigan_jsut \
    --init-channels 256 \
    --max-steps 50000 \
    --batch-size 16 \
    --lr 2e-4 \
    --mel-coeff 45.0 \
    --save-every 5000
echo "HiFi-GAN done: $(date)"

# ---- 3. Vocos (resume from 40k) ----
echo ""
echo "=== [3/3] Vocos vocoder (resume 40k, +10k steps) ==="
echo "start: $(date)"
$PY src/cli/vocos_train.py \
    --filelist data/filelist_train.txt \
    --out checkpoints/vocos_jsut \
    --resume data/ckpts/vocos_last.pth \
    --max-steps 10000 \
    --batch-size 16 \
    --lr 5e-4 \
    --mel-coeff 45.0 \
    --save-every 5000
echo "Vocos done: $(date)"

echo ""
echo "=== ALL DONE $(date) ==="
