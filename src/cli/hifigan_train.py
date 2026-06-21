#!/usr/bin/env python3
"""HiFi-GAN mel-vocoder GAN trainer. Same discriminators/losses as vocos_train.py
with the HiFi-GAN generator swapped in, so quality/speed differences are purely
the generator. --init-channels 256 = piper config (default), 512 = V1."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
from sqzw.flow import WavSet
from sqzw.hifigan_gen import Generator
from sqzw.gan_train import disc_loss, gen_adv_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="data/filelist_train.txt")
    ap.add_argument("--out", default="checkpoints/hifigan")
    ap.add_argument("--max-steps", type=int, default=300000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init-channels", type=int, default=256, help="256=piper (default), 512=HiFi-GAN V1")
    ap.add_argument("--mel-coeff", type=float, default=45.0)
    ap.add_argument("--mrd-coeff", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--resume", default="", help="load G weights from a checkpoint and fine-tune (step restarts at 0)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_ch = a.init_channels
    resume_ck = None
    if a.resume:
        resume_ck = torch.load(a.resume, map_location=dev)
        init_ch = resume_ck.get("init_channels", init_ch)
    G = Generator(init_channels=init_ch).to(dev)
    if resume_ck is not None:
        G.load_state_dict(resume_ck["G"])
        print(f"resumed G from {a.resume} (step {resume_ck.get('step')}, ch={init_ch}) -> fine-tune", flush=True)
    mpd = MultiPeriodDiscriminator().to(dev); mrd = MultiResolutionDiscriminator().to(dev)
    dloss_fn = DiscriminatorLoss(); gloss_fn = GeneratorLoss(); fmloss_fn = FeatureMatchingLoss()
    melloss = MelSpecReconstructionLoss(sample_rate=22050).to(dev)
    print(f"HiFi-GAN G params={sum(p.numel() for p in G.parameters())/1e6:.2f}M (init_ch={init_ch}) device={dev}", flush=True)

    opt_g = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.8, 0.9))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=a.lr, betas=(0.8, 0.9))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=a.max_steps)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=a.max_steps)

    dl = DataLoader(WavSet(a.filelist, a.num_samples), batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, persistent_workers=True)
    step = 0; t0 = time.time(); G.train()
    while step < a.max_steps:
        for audio in dl:
            audio = audio.to(dev)
            with torch.no_grad():
                audio_hat = G(audio)
            loss_d = disc_loss(mpd, mrd, dloss_fn, audio, audio_hat, a.mrd_coeff)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step(); sch_d.step()

            audio_hat = G(audio)
            loss_adv, loss_fm = gen_adv_loss(mpd, mrd, gloss_fn, fmloss_fn, audio, audio_hat, a.mrd_coeff)
            l_mel = melloss(audio_hat, audio)
            loss_g = loss_adv + loss_fm + a.mel_coeff * l_mel
            opt_g.zero_grad(); loss_g.backward(); opt_g.step(); sch_g.step()

            step += 1
            if step % 100 == 0:
                print(f"step {step}/{a.max_steps} d {loss_d.item():.3f} g {loss_g.item():.3f} mel {l_mel.item():.3f} {time.time()-t0:.0f}s", flush=True)
            if step % a.save_every == 0 or step == a.max_steps:
                ck = {"G": G.state_dict(), "opt_g": opt_g.state_dict(),
                      "opt_d": opt_d.state_dict(), "step": step, "init_channels": init_ch}
                torch.save(ck, f"{a.out}/hifigan_step{step}.pth")
                torch.save(ck, f"{a.out}/hifigan_last.pth")
                print(f"  saved step {step}", flush=True)
            if step >= a.max_steps: break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
