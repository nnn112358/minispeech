#!/usr/bin/env python3
"""Vocos copy-synthesis eval: same mel-L1 reconstruction metric as infer.py."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from decoders.vocos.generator import Generator
from common.eval_vocoder import eval_copysynthesis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/ckpts/vocos_last.pth")
    ap.add_argument("--filelist", default="data/filelist_val.txt")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="vocos")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(dev)
    ck = torch.load(a.ckpt, map_location=dev)
    G.load_state_dict(ck["G"]); G.eval()
    print(f"loaded {a.ckpt} @ step {ck.get('step','?')}  device={dev}", flush=True)
    eval_copysynthesis(lambda mel: G.head(G.backbone(mel)), a.filelist, a.n, a.out, a.tag, dev)


if __name__ == "__main__":
    main()
