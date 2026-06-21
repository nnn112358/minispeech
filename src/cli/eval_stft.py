#!/usr/bin/env python3
"""Multi-resolution STFT copy-synthesis eval (fairer than mel-L1, which favours
the mel-loss-trained Vocos). Compares each vocoder's output magnitude spectrum
to ground truth. Magnitude-based (Spectral Convergence + Log-magnitude L1) so it
is phase-invariant — correct for generative/flow vocoders. Sweeps SqueezeWave
sigma. Lower is better."""
import sys, os, argparse
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import soundfile as sf
from sqzw.model import SqueezeWave
from sqzw.vocos_gen import Generator
from sqzw.features import PiperMelFeatures

SR = 22050
RES = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]  # (n_fft, hop, win)


def mrstft(gt, out, dev):
    """mean over resolutions of (spectral convergence, log-magnitude L1)."""
    L = min(gt.shape[-1], out.shape[-1])
    gt = gt[..., :L]; out = out[..., :L]
    scs, lms = [], []
    for n_fft, hop, win in RES:
        w = torch.hann_window(win, device=dev)
        Y = torch.stft(gt, n_fft, hop, win, w, return_complex=True).abs()
        Yh = torch.stft(out, n_fft, hop, win, w, return_complex=True).abs()
        sc = torch.linalg.norm(Y - Yh) / (torch.linalg.norm(Y) + 1e-8)
        lm = (torch.log(Y + 1e-5) - torch.log(Yh + 1e-5)).abs().mean()
        scs.append(sc.item()); lms.append(lm.item())
    return float(np.mean(scs)), float(np.mean(lms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpts_a256_c128_q_gan/sqzwgan_bnrecal.pth")
    ap.add_argument("--vocos", default="data/ckpts/vocos_last.pth")
    ap.add_argument("--filelist", default="data/filelist_val.txt")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--sigmas", default="0.6,0.7,0.85,1.0")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = PiperMelFeatures(sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80).to(dev)

    ck = torch.load(a.ckpt, map_location=dev)
    sq = SqueezeWave(**ck["config"]).to(dev); sq.load_state_dict(ck["model"])
    sq = SqueezeWave.remove_weightnorm(sq); sq.eval()
    vo = Generator().to(dev); vo.load_state_dict(torch.load(a.vocos, map_location=dev)["G"]); vo.eval()
    sigmas = [float(s) for s in a.sigmas.split(",")]
    print(f"SqueezeWave step {ck['step']}  cfg {ck['config']['n_audio_channel']}/"
          f"{ck['config']['WN_config']['n_channels']}   files n={a.n}", flush=True)

    files = [l for l in open(a.filelist).read().splitlines() if l.strip()][:a.n]
    acc = {f"sqzw s{s}": [[], []] for s in sigmas}; acc["vocos"] = [[], []]
    for f in files:
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1: y = y[:, 0]
        x = torch.from_numpy(y).unsqueeze(0).to(dev)
        gt = x.squeeze(0)
        with torch.no_grad():
            mel = feat(x)
            ov = vo.head(vo.backbone(mel)).squeeze(0)
            sc, lm = mrstft(gt, ov, dev); acc["vocos"][0].append(sc); acc["vocos"][1].append(lm)
            for s in sigmas:
                os_ = sq.infer(mel, sigma=s).squeeze(0)
                sc, lm = mrstft(gt, os_, dev)
                acc[f"sqzw s{s}"][0].append(sc); acc[f"sqzw s{s}"][1].append(lm)

    print(f"\n{'model':14} {'SpecConv':>9} {'LogMagL1':>9}   (lower=better, vs GT)")
    for k in [f"sqzw s{s}" for s in sigmas] + ["vocos"]:
        sc = np.mean(acc[k][0]); lm = np.mean(acc[k][1])
        print(f"{k:14} {sc:9.4f} {lm:9.4f}")


if __name__ == "__main__":
    main()
