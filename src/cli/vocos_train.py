#!/usr/bin/env python3
"""Standalone Vocos mel-vocoder GAN trainer (no PyTorch-Lightning, to avoid the
repo's PL 1.8 -> 2.6 migration wall). Reuses vocos's proven components:
PiperMelFeatures (custom, matches FastSpeech mel) -> VocosBackbone -> ISTFTHead,
with MultiPeriod/MultiResolution discriminators + Vocos losses. Trains on JSUT
16kHz... no, 22050 wavs (filelist). Manual GAN optimization."""
import sys, os, glob, json, math, time, random, argparse
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
from sqzw.flow import WavSet
from sqzw.vocos_gen import Generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="data/filelist_train.txt")
    ap.add_argument("--out", default="data/ckpts")
    ap.add_argument("--max-steps", type=int, default=300000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mel-coeff", type=float, default=45.0)
    ap.add_argument("--mrd-coeff", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--resume", default="", help="load G weights from a checkpoint and fine-tune (step restarts at 0)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator().to(dev)
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        G.load_state_dict(ck["G"])
        print(f"resumed G from {a.resume} (was step {ck.get('step')}) -> fine-tune", flush=True)
    mpd = MultiPeriodDiscriminator().to(dev); mrd = MultiResolutionDiscriminator().to(dev)
    dloss = DiscriminatorLoss(); gloss = GeneratorLoss(); fmloss = FeatureMatchingLoss()
    melloss = MelSpecReconstructionLoss(sample_rate=22050).to(dev)
    print(f"G params={sum(p.numel() for p in G.parameters())/1e6:.2f}M device={dev}", flush=True)

    opt_g = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.8, 0.9))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=a.lr, betas=(0.8, 0.9))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=a.max_steps)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=a.max_steps)

    dl = DataLoader(WavSet(a.filelist, a.num_samples), batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, persistent_workers=True)
    step = 0; t0 = time.time(); G.train()
    while step < a.max_steps:
        for audio in dl:
            audio = audio.to(dev)  # (B, T)
            # ---- discriminator ----
            with torch.no_grad():
                audio_hat = G(audio)
            rmp, gmp, _, _ = mpd(y=audio, y_hat=audio_hat)
            rmr, gmr, _, _ = mrd(y=audio, y_hat=audio_hat)
            lmp, lmp_r, _ = dloss(disc_real_outputs=rmp, disc_generated_outputs=gmp)
            lmr, lmr_r, _ = dloss(disc_real_outputs=rmr, disc_generated_outputs=gmr)
            loss_d = lmp / len(lmp_r) + a.mrd_coeff * (lmr / len(lmr_r))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step(); sch_d.step()
            # ---- generator ----
            audio_hat = G(audio)
            _, gmp, frmp, fgmp = mpd(y=audio, y_hat=audio_hat)
            _, gmr, frmr, fgmr = mrd(y=audio, y_hat=audio_hat)
            lg_mp, l_mp = gloss(disc_outputs=gmp); lg_mr, l_mr = gloss(disc_outputs=gmr)
            lg_mp = lg_mp / len(l_mp); lg_mr = lg_mr / len(l_mr)
            lfm_mp = fmloss(fmap_r=frmp, fmap_g=fgmp) / len(frmp)
            lfm_mr = fmloss(fmap_r=frmr, fmap_g=fgmr) / len(frmr)
            l_mel = melloss(audio_hat, audio)
            loss_g = lg_mp + a.mrd_coeff * lg_mr + lfm_mp + a.mrd_coeff * lfm_mr + a.mel_coeff * l_mel
            opt_g.zero_grad(); loss_g.backward(); opt_g.step(); sch_g.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{a.max_steps} d {loss_d.item():.3f} g {loss_g.item():.3f} mel {l_mel.item():.3f} {time.time()-t0:.0f}s", flush=True)
            if step % a.save_every == 0 or step == a.max_steps:
                torch.save({"G": G.state_dict(), "step": step}, f"{a.out}/vocos_step{step}.pth")
                torch.save({"G": G.state_dict(), "step": step}, f"{a.out}/vocos_last.pth")
                print(f"  saved step {step}", flush=True)
            if step >= a.max_steps: break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
