#!/usr/bin/env python3
"""MB-iSTFT vocoder copy-synthesis evaluation."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from decoders.mb_istft.generator import Generator
from common.eval_vocoder import eval_copysynthesis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--filelist", default="data/filelist_val.txt")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="outputs/eval_mbistft")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.ckpt, map_location=dev)
    init_ch = ck.get("init_channels", 256)
    G = Generator(init_channels=init_ch).to(dev)
    G.load_state_dict(ck["G"])
    G.eval()

    def vocode(mel):
        return G.mbistft(mel)[0].squeeze(1).squeeze(0)

    print(f"MB-iSTFT eval: {a.ckpt} (step {ck.get('step')}, ch={init_ch})")
    eval_copysynthesis(vocode, a.filelist, a.n, a.out, "mbistft", dev)


if __name__ == "__main__":
    main()
