#!/usr/bin/env python3
"""MB-iSTFT mel-vocoder GAN trainer. Same discriminators/losses as hifigan_train.py
with the MB-iSTFT generator swapped in + sub-band multi-resolution STFT loss.
--init-channels 256 = piper-plus config (default)."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
from common.dataset import WavSet
from decoders.mb_istft.generator import Generator
from decoders.mb_istft.stft_loss import MultiResolutionSTFTLoss
from common.gan_train import disc_loss, gen_adv_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="data/filelist_train.txt")
    ap.add_argument("--out", default="checkpoints/mbistft_jsut")
    ap.add_argument("--max-steps", type=int, default=300000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init-channels", type=int, default=256)
    ap.add_argument("--n-fft", type=int, default=16, choices=[16, 32, 64])
    ap.add_argument("--mel-coeff", type=float, default=45.0)
    ap.add_argument("--mrd-coeff", type=float, default=0.1)
    ap.add_argument("--sub-stft-coeff", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--resume", default="", help="resume full state (G+D+opt+step) from checkpoint")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_ch = a.init_channels
    n_fft = a.n_fft
    resume_ck = None
    if a.resume:
        resume_ck = torch.load(a.resume, map_location=dev)
        init_ch = resume_ck.get("init_channels", init_ch)
        n_fft = resume_ck.get("n_fft", n_fft)
    G = Generator(init_channels=init_ch, n_fft=n_fft).to(dev)
    mpd = MultiPeriodDiscriminator().to(dev); mrd = MultiResolutionDiscriminator().to(dev)
    dloss_fn = DiscriminatorLoss(); gloss_fn = GeneratorLoss(); fmloss_fn = FeatureMatchingLoss()
    melloss = MelSpecReconstructionLoss(sample_rate=22050).to(dev)
    sub_stft_loss = MultiResolutionSTFTLoss().to(dev)
    print(f"MB-iSTFT G params={sum(p.numel() for p in G.parameters())/1e6:.2f}M (init_ch={init_ch}, n_fft={n_fft}) device={dev}", flush=True)

    opt_g = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.8, 0.9))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=a.lr, betas=(0.8, 0.9))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=a.max_steps)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=a.max_steps)

    step = 0
    if resume_ck is not None:
        G.load_state_dict(resume_ck["G"])
        if "mpd" in resume_ck:
            mpd.load_state_dict(resume_ck["mpd"])
            mrd.load_state_dict(resume_ck["mrd"])
            opt_g.load_state_dict(resume_ck["opt_g"])
            opt_d.load_state_dict(resume_ck["opt_d"])
            step = resume_ck.get("step", 0)
            for _ in range(step):
                sch_g.step(); sch_d.step()
            print(f"resumed full state from {a.resume} (step {step}, ch={init_ch})", flush=True)
        else:
            print(f"resumed G-only from {a.resume} (step {resume_ck.get('step')}, ch={init_ch}) — D/opt fresh", flush=True)

    dl = DataLoader(WavSet(a.filelist, a.num_samples), batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, persistent_workers=True, pin_memory=True)
    scaler = torch.amp.GradScaler("cuda")
    if dev.type == "cuda":
        G = torch.compile(G); mpd = torch.compile(mpd); mrd = torch.compile(mrd)
    t0 = time.time(); G.train()
    while step < a.max_steps:
        for audio in dl:
            audio = audio.to(dev, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                audio_hat = G(audio)
            with torch.amp.autocast("cuda"):
                loss_d = disc_loss(mpd, mrd, dloss_fn, audio, audio_hat, a.mrd_coeff)
            opt_d.zero_grad(); scaler.scale(loss_d).backward(); scaler.step(opt_d); sch_d.step()

            with torch.amp.autocast("cuda"):
                audio_hat = G(audio)
                loss_adv, loss_fm = gen_adv_loss(mpd, mrd, gloss_fn, fmloss_fn, audio, audio_hat, a.mrd_coeff)
                l_mel = melloss(audio_hat, audio)

                l_sub = torch.tensor(0.0, device=dev)
                if G.last_subbands is not None and a.sub_stft_coeff > 0:
                    with torch.no_grad():
                        y_mb = G.pqmf.analysis(audio.unsqueeze(1))
                    l_sub = sub_stft_loss(G.last_subbands, y_mb)

                loss_g = loss_adv + loss_fm + a.mel_coeff * l_mel + a.sub_stft_coeff * l_sub
            opt_g.zero_grad(); scaler.scale(loss_g).backward(); scaler.step(opt_g); sch_g.step()
            scaler.update()

            step += 1
            if step % 100 == 0:
                print(f"step {step}/{a.max_steps} d {loss_d.item():.3f} g {loss_g.item():.3f} "
                      f"mel {l_mel.item():.3f} sub {l_sub.item():.3f} {time.time()-t0:.0f}s", flush=True)
            if step % a.save_every == 0 or step == a.max_steps:
                ck = {"G": G.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                      "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                      "step": step, "init_channels": init_ch, "n_fft": n_fft, "vocoder_type": "mbistft"}
                torch.save(ck, f"{a.out}/mbistft_step{step}.pth")
                torch.save(ck, f"{a.out}/mbistft_last.pth")
                print(f"  saved step {step}", flush=True)
            if step >= a.max_steps: break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
