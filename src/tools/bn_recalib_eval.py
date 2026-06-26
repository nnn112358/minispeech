#!/usr/bin/env python3
"""Test the hypothesis that BatchNorm train/eval mismatch caps eval quality.
Recalibrate BN running stats on the INFERENCE (reverse-flow) distribution, then
re-evaluate. If eval metrics jump toward the training-loss level, BN was the cap."""
import sys, os, random, argparse
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
import soundfile as sf
from decoders.squeezewave.model import SqueezeWave
from common.features import PiperMelFeatures
from decoders.squeezewave.flow import diff_infer, WavSet
from torch.utils.data import DataLoader

SR = 22050
RES = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]


def mrstft(gt, out, dev):
    L = min(gt.shape[-1], out.shape[-1]); gt = gt[..., :L]; out = out[..., :L]
    scs, lms = [], []
    for n_fft, hop, win in RES:
        w = torch.hann_window(win, device=dev)
        Y = torch.stft(gt, n_fft, hop, win, w, return_complex=True).abs()
        Yh = torch.stft(out, n_fft, hop, win, w, return_complex=True).abs()
        scs.append((torch.linalg.norm(Y - Yh) / (torch.linalg.norm(Y) + 1e-8)).item())
        lms.append((torch.log(Y + 1e-5) - torch.log(Yh + 1e-5)).abs().mean().item())
    return float(np.mean(scs)), float(np.mean(lms))


def evaluate(model, feat, files, dev, sigmas):
    res = {}
    for s in sigmas:
        scs, lms, l1s = [], [], []
        for f in files:
            y, sr = sf.read(f, dtype="float32")
            if y.ndim > 1: y = y[:, 0]
            x = torch.from_numpy(y).unsqueeze(0).to(dev)
            with torch.no_grad():
                mel = feat(x)
                out = model.infer(mel, sigma=s).squeeze(0)
                sc, lm = mrstft(x.squeeze(0), out, dev)
                mel2 = feat(out.unsqueeze(0)); L = min(mel.size(2), mel2.size(2))
                l1 = (mel[:, :, :L] - mel2[:, :, :L]).abs().mean().item()
            scs.append(sc); lms.append(lm); l1s.append(l1)
        res[s] = (np.mean(scs), np.mean(lms), np.mean(l1s))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="_unused/m3_sqzw_ckpts/ckpts_aux/sqzwaux_step40000.pth")
    ap.add_argument("--filelist", default="data/filelist_val.txt")
    ap.add_argument("--trainlist", default="data/filelist_train.txt")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--recalib-batches", type=int, default=40)
    ap.add_argument("--save", default="")  # save recalibrated ckpt path
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = PiperMelFeatures(sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80).to(dev)
    files = [l for l in open(a.filelist).read().splitlines() if l.strip()][:a.n]
    sigmas = [0.6, 0.8, 1.0]

    def build():
        ck = torch.load(a.ckpt, map_location=dev)
        m = SqueezeWave(**ck["config"]).to(dev); m.load_state_dict(ck["model"]); return m, ck

    # --- baseline (as-trained BN stats) ---
    m, ck = build()
    m_eval = SqueezeWave.remove_weightnorm(m); m_eval.eval()
    base = evaluate(m_eval, feat, files, dev, sigmas)

    # --- recalibrated BN on reverse/inference distribution ---
    m2, ck = build()
    nbn = 0
    for mod in m2.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.reset_running_stats(); mod.momentum = None; nbn += 1
    m2.train()  # BN updates running stats from the batches we push through
    dl = DataLoader(WavSet(a.trainlist, 16384), batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    it = iter(dl); done = 0
    with torch.no_grad():
        while done < a.recalib_batches:
            try: audio = next(it)
            except StopIteration: it = iter(dl); audio = next(it)
            audio = audio.to(dev); mel = feat(audio)
            diff_infer(m2, mel, 1.0)   # reverse pass -> BN sees inference-distribution acts
            done += 1
    m2.eval()
    m2_eval = SqueezeWave.remove_weightnorm(m2); m2_eval.eval()
    reca = evaluate(m2_eval, feat, files, dev, sigmas)

    print(f"\nBN modules recalibrated: {nbn}, batches={a.recalib_batches}, files n={a.n}")
    print(f"{'sigma':>6} | {'baseline SC/LM/mel':>26} | {'recalib SC/LM/mel':>26}")
    for s in sigmas:
        b = base[s]; r = reca[s]
        print(f"{s:6.2f} | {b[0]:7.3f} {b[1]:7.3f} {b[2]:7.3f}   | {r[0]:7.3f} {r[1]:7.3f} {r[2]:7.3f}")
    print("Vocos ref     |   0.270   0.649   0.216   |")

    if a.save:
        torch.save({"model": m2.state_dict(), "opt": ck.get("opt", {}), "step": ck["step"],
                    "config": ck["config"]}, a.save)
        print(f"saved recalibrated -> {a.save}")


if __name__ == "__main__":
    main()
