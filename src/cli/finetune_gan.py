#!/usr/bin/env python3
"""GAN fine-tune for SqueezeWave: aux losses (NLL + mel + multi-res STFT) PLUS
adversarial + feature-matching from Vocos's MPD/MRD discriminators. The flow's
differentiable reverse pass (diff_infer) is the generator; discriminators are
TRAINING-ONLY (zero inference params/cost). This adds the perceptual sharpness
that reconstruction losses alone miss — the main lever for small configs.
Init from an NLL (or aux) checkpoint; BN-recalibrate afterwards."""
import sys, os, time, random, argparse
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import soundfile as sf
from torch.utils.data import DataLoader
from sqzw.model import SqueezeWave, SqueezeWaveLoss
from sqzw.features import PiperMelFeatures
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss
from sqzw.flow import diff_infer, mrstft_loss, WavSet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", default="_unused/m3_sqzw_ckpts/ckpts_c64_f12_l8/sqzw_last.pth")
    ap.add_argument("--filelist", default="data/filelist_train.txt")
    ap.add_argument("--out", default="checkpoints/ckpts_gan")
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--w-nll", type=float, default=1.0)
    ap.add_argument("--w-mel", type=float, default=5.0)
    ap.add_argument("--w-stft", type=float, default=2.0)
    ap.add_argument("--w-adv", type=float, default=1.0)
    ap.add_argument("--w-fm", type=float, default=2.0)
    ap.add_argument("--mrd-coeff", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat = PiperMelFeatures(sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80).to(dev)
    ck = torch.load(a.init_from, map_location=dev)
    cfg = ck["config"]
    G = SqueezeWave(**cfg).to(dev); G.load_state_dict(ck["model"])
    criterion = SqueezeWaveLoss(a.sigma)
    mpd = MultiPeriodDiscriminator().to(dev); mrd = MultiResolutionDiscriminator().to(dev)
    dloss = DiscriminatorLoss(); gloss = GeneratorLoss(); fmloss = FeatureMatchingLoss()
    print(f"GAN init from {a.init_from} (step {ck['step']}) cfg nac{cfg['n_audio_channel']}/"
          f"c{cfg['WN_config']['n_channels']} dil={cfg['WN_config'].get('dilation_cycle')} "
          f"w_adv={a.w_adv} w_fm={a.w_fm} w_mel={a.w_mel} w_stft={a.w_stft}", flush=True)

    opt_g = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.8, 0.9))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=a.lr, betas=(0.8, 0.9))
    step0 = 0
    if a.resume and os.path.isfile(a.resume):
        rk = torch.load(a.resume, map_location=dev)
        G.load_state_dict(rk["model"]); mpd.load_state_dict(rk["mpd"]); mrd.load_state_dict(rk["mrd"])
        opt_g.load_state_dict(rk["opt_g"]); opt_d.load_state_dict(rk["opt_d"]); step0 = rk["step"]
        print(f"resumed {a.resume} @ step {step0}", flush=True)

    dl = DataLoader(WavSet(a.filelist, a.num_samples), batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, persistent_workers=True)
    step = step0; t0 = time.time(); G.train(); mpd.train(); mrd.train()
    r = {"d": None, "adv": None, "fm": None, "mel": None, "stft": None}
    def ema(k, v): r[k] = v if r[k] is None else 0.99 * r[k] + 0.01 * v

    while step < a.max_steps:
        for audio in dl:
            audio = audio.to(dev)
            with torch.no_grad():
                mel = feat(audio)
            gen = diff_infer(G, mel, a.sigma)            # (B, L)
            L = gen.shape[-1]; gt = audio[:, :L]
            # ---- discriminator ----
            rmp, gmp, _, _ = mpd(y=gt, y_hat=gen.detach())
            rmr, gmr, _, _ = mrd(y=gt, y_hat=gen.detach())
            lmp, lmp_r, _ = dloss(disc_real_outputs=rmp, disc_generated_outputs=gmp)
            lmr, lmr_r, _ = dloss(disc_real_outputs=rmr, disc_generated_outputs=gmr)
            loss_d = lmp / len(lmp_r) + a.mrd_coeff * (lmr / len(lmr_r))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            # ---- generator ----
            if a.w_nll > 0:
                out = G((mel, audio)); loss_nll = criterion(out)   # extra fwd; skip when w_nll=0
            else:
                loss_nll = torch.zeros((), device=dev)
            _, gmp, frmp, fgmp = mpd(y=gt, y_hat=gen)
            _, gmr, frmr, fgmr = mrd(y=gt, y_hat=gen)
            lg_mp, l_mp = gloss(disc_outputs=gmp); lg_mr, l_mr = gloss(disc_outputs=gmr)
            loss_adv = lg_mp / len(l_mp) + a.mrd_coeff * (lg_mr / len(l_mr))
            loss_fm = fmloss(fmap_r=frmp, fmap_g=fgmp) / len(frmp) \
                      + a.mrd_coeff * (fmloss(fmap_r=frmr, fmap_g=fgmr) / len(frmr))
            mel_gen = feat(gen); Lm = min(mel.size(2), mel_gen.size(2))
            loss_mel = (mel[:, :, :Lm] - mel_gen[:, :, :Lm]).abs().mean()
            sc, lm = mrstft_loss(gt, gen, dev); loss_stft = sc + lm
            loss_g = (a.w_nll * loss_nll + a.w_mel * loss_mel + a.w_stft * loss_stft
                      + a.w_adv * loss_adv + a.w_fm * loss_fm)
            if not torch.isfinite(loss_g):
                print(f"  WARN non-finite G @ {step}, skip", flush=True); opt_g.zero_grad(); step += 1; continue
            opt_g.zero_grad(); loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 20.0)
            opt_g.step()
            ema("d", loss_d.item()); ema("adv", loss_adv.item()); ema("fm", loss_fm.item())
            ema("mel", loss_mel.item()); ema("stft", loss_stft.item())
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{a.max_steps} d {r['d']:.3f} adv {r['adv']:.3f} fm {r['fm']:.3f} "
                      f"mel {r['mel']:.3f} stft {r['stft']:.3f} {step/(time.time()-t0):.1f}it/s", flush=True)
            if step % a.save_every == 0 or step == a.max_steps:
                torch.save({"model": G.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                            "step": step, "config": cfg}, f"{a.out}/sqzwgan_step{step}.pth")
                torch.save({"model": G.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                            "step": step, "config": cfg}, f"{a.out}/sqzwgan_last.pth")
                print(f"  saved {step} (mel {r['mel']:.3f} stft {r['stft']:.3f} adv {r['adv']:.3f})", flush=True)
            if step >= a.max_steps: break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
