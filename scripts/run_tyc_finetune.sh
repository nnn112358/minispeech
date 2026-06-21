#!/bin/bash
# Fine-tune 3 vocoders on the Tsukuyomi corpus (VOICEACTRESS100, 22050Hz).
#   - SqueezeWave c64_f8_l4  : GAN-continue from JSUT GAN ckpt -> BN recalib -> sample
#   - SqueezeWave a128_c128  : GAN from JSUT NLL ckpt          -> BN recalib -> sample
#   - Vocos                  : resume G from JSUT ckpt, GAN fine-tune          -> sample
# Sequential (single GPU). Each ~2000 steps. Logs to outputs/tyc_finetune.log.
export PYTHONPATH=""
export CUDA_VISIBLE_DEVICES=0
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PYTHON:-/home/nnn/piper-wavehax-venv/bin/python}
TYC=data/filelist_tyc_train.txt
VAL=data/filelist_tyc_val.txt
STEPS=2000
G="--w-nll 0 --w-mel 45 --w-stft 1 --w-adv 1 --w-fm 2 --max-steps $STEPS --batch-size 8 --save-every 1000"

run() { echo; echo "######## $1 ########"; shift; "$@"; echo "  -> exit $?"; }

# [1/3] c64_f8_l4 : GAN-continue from the JSUT GAN-finished model
O=checkpoints/tyc_c64_gan
run "[1/3] c64 GAN fine-tune" $PY -u src/cli/finetune_gan.py \
  --init-from checkpoints/ckpts_c64_f8_l4_s_gan/sqzwgan_bnrecal.pth --filelist $TYC --out $O $G
run "[1/3] c64 BN recalib" $PY -u src/cli/bake_bnrecal.py --ckpt $O/sqzwgan_last.pth --out $O/sqzwgan_bnrecal.pth --trainlist $TYC
run "[1/3] c64 sample"     $PY -u src/cli/infer.py --ckpt $O/sqzwgan_bnrecal.pth --filelist $VAL --n 3 --sigma 0.7 --tag tyc_c64

# [2/3] a128_c128 : GAN from the JSUT NLL model
O=checkpoints/tyc_a128_gan
run "[2/3] a128 GAN fine-tune" $PY -u src/cli/finetune_gan.py \
  --init-from _unused/m3_sqzw_ckpts/ckpts_a128c128_old/sqzw_last.pth --filelist $TYC --out $O $G
run "[2/3] a128 BN recalib" $PY -u src/cli/bake_bnrecal.py --ckpt $O/sqzwgan_last.pth --out $O/sqzwgan_bnrecal.pth --trainlist $TYC
run "[2/3] a128 sample"     $PY -u src/cli/infer.py --ckpt $O/sqzwgan_bnrecal.pth --filelist $VAL --n 3 --sigma 0.7 --tag tyc_a128

# [3/3] Vocos : resume G and GAN fine-tune
O=checkpoints/tyc_vocos
run "[3/3] Vocos fine-tune" $PY -u src/cli/vocos_train.py \
  --resume data/ckpts/vocos_last.pth --filelist $TYC --out $O --max-steps $STEPS --batch-size 8 --save-every 1000 --lr 2e-4
run "[3/3] Vocos sample"    $PY -u src/cli/eval_vocos.py --ckpt $O/vocos_last.pth --filelist $VAL --n 3 --tag tyc_vocos

echo; echo "######## TYC FINE-TUNE DONE ########"
