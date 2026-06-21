#!/bin/bash
# Full JSUT training pipeline: encoder → HiFi-GAN → Vocos → SqueezeWave
# SqueezeWave is already trained (NLL→aux→GAN→BN recal done), so only eval.
set -euo pipefail
cd /home/nnn/fast-jtts
PY=~/.venvs/piper_ft/bin/python
LOG=logs/jsut_all_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs fs/jsut_align checkpoints/hifigan_jsut checkpoints/vocos_jsut

exec > >(tee -a "$LOG") 2>&1
echo "=== JSUT full pipeline started $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 1. FastSpeech encoder (self-aligning) ----
echo ""
echo "=== [1/4] FastSpeech encoder (self-aligning, 500 epochs) ==="
echo "start: $(date)"
$PY src/cli/train_fastspeech.py \
    --manifest fs_data/jsut_manifest.json \
    --learn-alignment \
    --out fs/jsut_align \
    --epochs 500 \
    --batch-size 16 \
    --lr 3e-4 \
    --cosine \
    --save-every 100
echo "FastSpeech done: $(date)"

# ---- 2. HiFi-GAN (piper config: 256ch, ResBlock2) ----
echo ""
echo "=== [2/4] HiFi-GAN vocoder (piper 256ch, 50k steps) ==="
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
echo "=== [3/4] Vocos vocoder (resume 40k, +10k steps) ==="
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

# ---- 4. SqueezeWave (already trained — just confirm) ----
echo ""
echo "=== [4/4] SqueezeWave — already trained ==="
echo "Existing checkpoints:"
ls -lh checkpoints/ckpts_c64_f8_l4_s_gan/sqzwgan_bnrecal.pth 2>/dev/null || echo "  (not found)"
ls -lh checkpoints/ckpts_a256_c128_q_gan/sqzwgan_bnrecal.pth 2>/dev/null || echo "  (not found)"

echo ""
echo "=== ALL DONE $(date) ==="
