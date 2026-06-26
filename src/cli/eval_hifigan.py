#!/usr/bin/env python3
"""HiFi-GAN copy-synthesis eval: same mel-L1 reconstruction metric as infer.py."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from decoders.hifigan.generator import Generator
from common.eval_vocoder import eval_copysynthesis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hifigan/hifigan_last.pth")
    ap.add_argument("--filelist", default="data/filelist_val.txt")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="hifigan")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev)
    G = Generator(init_channels=ck.get("init_channels", 256)).to(dev)
    G.load_state_dict(ck["G"]); G.eval()
    print(f"loaded {a.ckpt} @ step {ck.get('step','?')} ch={ck.get('init_channels',256)}  device={dev}", flush=True)
    eval_copysynthesis(lambda mel: G.hifigan(mel).squeeze(1), a.filelist, a.n, a.out, a.tag, dev)


if __name__ == "__main__":
    main()
