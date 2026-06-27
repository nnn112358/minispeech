#!/usr/bin/env python3
"""SqueezeWave vocoder inference / validation for the JSUT 2-stage TTS.
(a) copy-synthesis: val wav -> PiperMelFeatures mel -> SqueezeWave.infer -> wav
    (isolates vocoder fidelity), or
(b) from a precomputed FS mel .npy (true end-to-end check).
Reports synthesis RTF and a mel-L1 reconstruction error vs the input mel."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import soundfile as sf
from decoders.squeezewave.model import SqueezeWave
from common.features import PiperMelFeatures
from common.eval_vocoder import eval_copysynthesis

SR = 22050


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    cfg = ck["config"]
    model = SqueezeWave(**cfg).to(dev)
    model.load_state_dict(ck["model"])
    model = SqueezeWave.remove_weightnorm(model)
    model.eval()
    return model, ck["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpts_a256_c128_q_gan/sqzwgan_bnrecal.pth")
    ap.add_argument("--filelist", default="data/filelist_val.txt")
    ap.add_argument("--mel-npy", default="")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="sqzw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step = load_model(a.ckpt, dev)
    print(f"loaded {a.ckpt} @ step {step}  sigma={a.sigma}  device={dev}", flush=True)

    if a.mel_npy:
        mel = torch.from_numpy(np.load(a.mel_npy)).float().unsqueeze(0).to(dev)
        with torch.no_grad():
            audio = model.infer(mel, sigma=a.sigma).squeeze(0).cpu().numpy()
        op = f"{a.out}/{a.tag}_melnpy.wav"
        sf.write(op, np.clip(audio, -1, 1), SR)
        print(f"  {os.path.basename(a.mel_npy)} -> {op}  ({len(audio)/SR:.2f}s)", flush=True)
        return

    sigma = a.sigma
    eval_copysynthesis(lambda mel: model.infer(mel, sigma=sigma).squeeze(0),
                       a.filelist, a.n, a.out, a.tag, dev, save_gt=True)


if __name__ == "__main__":
    main()
