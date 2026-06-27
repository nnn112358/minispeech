#!/bin/bash
# GAN-finish 2 configs (quality + speed) to remove the flow 割れ (HF jitter):
#   GAN fine-tune (adv+fm+mel+stft) -> BN recalib -> eval -> sample audio.
# Discriminators (Vocos MPD/MRD) are training-only (zero inference cost).
export PYTHONPATH=""
export CUDA_VISIBLE_DEVICES=0
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PYTHON:-python}
MAXTRY=15

run_retry () {
  local label="$1"; local donef="$2"; shift 2; local t=0
  while [ ! -f "$donef" ] && [ $t -lt $MAXTRY ]; do
    t=$((t+1)); echo ">>> $label attempt $t/$MAXTRY ($(date +%m-%d_%H:%M:%S))"
    "$@" || echo ">>> $label crashed attempt $t (auto-resume)"; sleep 8
  done
  [ -f "$donef" ] || { echo ">>> $label FAILED after $MAXTRY"; return 1; }
  echo ">>> $label COMPLETE"
}

run_gan () {                       # tag init_ckpt gan_steps sigma
  local TAG=$1 INIT=$2 GANS=$3 SIG=$4
  local CKG="checkpoints/ckpts_${TAG}_gan"
  echo "############ GAN FINISH $TAG (init $INIT, $GANS steps) $(date +%m-%d_%H:%M:%S) ############"
  # HiFiGAN/Vocos recipe: strong mel anchor (45) so adversarial only fixes the
  # fine structure (割れ) instead of overpowering & drifting the generator.
  run_retry "[$TAG] GAN" "$CKG/sqzwgan_step${GANS}.pth" \
    $PY -u src/cli/finetune_gan.py --init-from $INIT --resume $CKG/sqzwgan_last.pth \
    --max-steps $GANS --batch-size 8 --lr 2e-4 --w-nll 0.0 --w-mel 45.0 --w-stft 1.0 --w-adv 1.0 --w-fm 2.0 \
    --save-every 2000 --out $CKG || return 1
  echo "=== [$TAG] BN recalib ==="
  $PY -u src/cli/bake_bnrecal.py --ckpt $CKG/sqzwgan_last.pth --out $CKG/sqzwgan_bnrecal.pth --batches 60 || return 1
  echo "=== [$TAG] EVAL (vs Vocos) ==="
  $PY -u src/cli/eval_stft.py --ckpt $CKG/sqzwgan_bnrecal.pth --n 10 || return 1
  echo "=== [$TAG] SAMPLE AUDIO ==="
  $PY -u src/cli/infer.py --ckpt $CKG/sqzwgan_bnrecal.pth --n 3 --sigma $SIG --tag ${TAG}_gan || return 1
  echo "############ GAN FINISH $TAG DONE $(date +%m-%d_%H:%M:%S) ############"
}

run_gan a256_c128_q  checkpoints/ckpts_aux/sqzwaux_last.pth                 12000 0.7
run_gan c64_f8_l4_s  checkpoints/ckpts_c64_f8_l4_dgan_aux/sqzwaux_last.pth  12000 0.7

echo "############ ALL GAN FINISH DONE $(date +%m-%d_%H:%M:%S) ############"
