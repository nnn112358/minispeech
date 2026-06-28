#!/usr/bin/env python3
"""Standalone Vocos mel-vocoder GAN trainer (DEFAULT vocoder). Reuses vocos's proven components:
PiperMelFeatures -> VocosBackbone -> ISTFTHead, with MultiPeriod/MultiResolution
discriminators + Vocos losses. Trains on 22050 Hz wavs (filelist)."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
from common.dataset import WavSet
from decoders.vocos.generator import Generator
from common.gan_train import disc_loss, gen_adv_loss


def main():
    ap = argparse.ArgumentParser()
    # --- I/O & schedule (shared across all vocoder trainers) ---
    ap.add_argument("--filelist", default="data/filelist_train.txt")
    ap.add_argument("--out", default="data/ckpts")
    ap.add_argument("--max-steps", type=int, default=300000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save-every", type=int, default=5000)
    # --- loss weights (shared) ---
    ap.add_argument("--mel-coeff", type=float, default=45.0)
    ap.add_argument("--mrd-coeff", type=float, default=0.1)
    # --- model-specific architecture ---
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--intermediate-dim", type=int, default=1536)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--mpd-max-ch", type=int, default=0,
                    help="MPD max channels (0=default 128, decoupled from dim; large G may want more)")
    # --- resume ---
    ap.add_argument("--resume", default="", help="resume full state (G+D+opt+step) from checkpoint")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim, inter_dim, n_layers, n_fft = a.dim, a.intermediate_dim, a.num_layers, a.n_fft
    resume_ck = None
    if a.resume:
        resume_ck = torch.load(a.resume, map_location=dev)
        cfg = resume_ck.get("config", {})
        dim = cfg.get("dim", dim)
        inter_dim = cfg.get("intermediate_dim", inter_dim)
        n_layers = cfg.get("num_layers", n_layers)
        n_fft = cfg.get("n_fft", n_fft)

    G = Generator(dim=dim, intermediate_dim=inter_dim, num_layers=n_layers, n_fft=n_fft).to(dev)
    # MPD width is decoupled from dim. The old `dim*2` over-discriminated small generators
    # (e.g. d128->256, d256->512) — a too-strong MPD injects high-freq GAN artifacts and
    # *worsens* held-out mel-L1 (d256: 0.366 -> 0.243 once retrained at max_ch=128). 128 is
    # validated across the lite configs (d64..d256); large generators (Full/d512, which stays
    # balanced at D/G~3x) can pass a bigger --mpd-max-ch explicitly.
    mpd_max_ch = a.mpd_max_ch if a.mpd_max_ch > 0 else 128
    mpd = MultiPeriodDiscriminator(max_channels=mpd_max_ch).to(dev)
    mrd = MultiResolutionDiscriminator().to(dev)
    dloss_fn = DiscriminatorLoss(); gloss_fn = GeneratorLoss(); fmloss_fn = FeatureMatchingLoss()
    melloss = MelSpecReconstructionLoss(sample_rate=22050).to(dev)
    p_g = sum(p.numel() for p in G.parameters()) / 1e6
    p_mpd = sum(p.numel() for p in mpd.parameters()) / 1e6
    p_mrd = sum(p.numel() for p in mrd.parameters()) / 1e6
    print(f"Vocos G={p_g:.2f}M (dim={dim} inter={inter_dim} layers={n_layers} n_fft={n_fft}) "
          f"D={p_mpd+p_mrd:.2f}M (MPD={p_mpd:.2f}M[max_ch={mpd_max_ch}] MRD={p_mrd:.2f}M) "
          f"D/G={((p_mpd+p_mrd)/p_g):.1f}x device={dev}", flush=True)

    opt_g = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.8, 0.9))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=a.lr, betas=(0.8, 0.9))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=a.max_steps)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=a.max_steps)

    step = 0
    if resume_ck is not None:
        # strict=False: iSTFT basis/window are deterministic buffers recomputed at
        # init (older ckpts didn't save them). Real weight shape mismatches still raise.
        miss, unexp = G.load_state_dict(resume_ck["G"], strict=False)
        bad = [k for k in miss if "basis" not in k and "window" not in k]
        if bad or unexp:
            raise RuntimeError(f"resume G mismatch: missing={bad} unexpected={list(unexp)}")
        if miss:
            print(f"  G resume: recomputed deterministic buffers {list(miss)}", flush=True)
        if "mpd" in resume_ck:
            try:
                mpd.load_state_dict(resume_ck["mpd"])
                mrd.load_state_dict(resume_ck["mrd"])
                opt_g.load_state_dict(resume_ck["opt_g"])
                opt_d.load_state_dict(resume_ck["opt_d"])
                step = resume_ck.get("step", 0)
                for _ in range(step):
                    sch_g.step(); sch_d.step()
                print(f"resumed full state from {a.resume} (step {step})", flush=True)
            except RuntimeError:
                step = resume_ck.get("step", 0)
                for _ in range(step):
                    sch_g.step()
                print(f"resumed G + step={step} from {a.resume} — D shape mismatch, D/opt fresh", flush=True)
        else:
            print(f"resumed G-only from {a.resume} (step {resume_ck.get('step')}) — D/opt fresh", flush=True)

    dl = DataLoader(WavSet(a.filelist, a.num_samples), batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, persistent_workers=True, pin_memory=True)
    scaler = torch.amp.GradScaler("cuda")
    if dev.type == "cuda" and n_fft >= 1024:
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
                loss_g = loss_adv + loss_fm + a.mel_coeff * l_mel
            opt_g.zero_grad(); scaler.scale(loss_g).backward(); scaler.step(opt_g); sch_g.step()
            scaler.update()

            step += 1
            if step % 100 == 0:
                print(f"step {step}/{a.max_steps} d {loss_d.item():.3f} g {loss_g.item():.3f} mel {l_mel.item():.3f} {time.time()-t0:.0f}s", flush=True)
            if step % a.save_every == 0 or step == a.max_steps:
                ck = {"G": G.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                      "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                      "step": step, "config": G.config}
                torch.save(ck, f"{a.out}/vocos_step{step}.pth")
                torch.save(ck, f"{a.out}/vocos_last.pth")
                print(f"  saved step {step}", flush=True)
            if step >= a.max_steps: break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
