"""Shared GAN training utilities: discriminator/generator loss computation.
Used by cli/vocos_train.py, cli/hifigan_train.py, cli/finetune_gan.py."""


def disc_loss(mpd, mrd, dloss, y, y_hat, mrd_coeff):
    """Discriminator loss given real (y) and generated (y_hat) audio."""
    rmp, gmp, _, _ = mpd(y=y, y_hat=y_hat)
    rmr, gmr, _, _ = mrd(y=y, y_hat=y_hat)
    lmp, lmp_r, _ = dloss(disc_real_outputs=rmp, disc_generated_outputs=gmp)
    lmr, lmr_r, _ = dloss(disc_real_outputs=rmr, disc_generated_outputs=gmr)
    return lmp / len(lmp_r) + mrd_coeff * (lmr / len(lmr_r))


def gen_adv_loss(mpd, mrd, gloss, fmloss, y, y_hat, mrd_coeff):
    """Generator adversarial + feature-matching losses. Returns (loss_adv, loss_fm)."""
    _, gmp, frmp, fgmp = mpd(y=y, y_hat=y_hat)
    _, gmr, frmr, fgmr = mrd(y=y, y_hat=y_hat)
    lg_mp, l_mp = gloss(disc_outputs=gmp)
    lg_mr, l_mr = gloss(disc_outputs=gmr)
    loss_adv = lg_mp / len(l_mp) + mrd_coeff * (lg_mr / len(l_mr))
    loss_fm = (fmloss(fmap_r=frmp, fmap_g=fgmp) / len(frmp)
               + mrd_coeff * (fmloss(fmap_r=frmr, fmap_g=fgmr) / len(frmr)))
    return loss_adv, loss_fm
