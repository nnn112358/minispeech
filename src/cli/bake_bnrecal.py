#!/usr/bin/env python3
"""Bake BN recalibration into a checkpoint in the STANDARD (pre-fusion) format,
so the normal infer/export pipeline (build -> load_state_dict -> remove_weightnorm)
works. Recalibrates BN running stats on the inference (reverse-flow) distribution."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from decoders.squeezewave.model import SqueezeWave
from common.features import PiperMelFeatures
from common.dataset import WavSet
from decoders.squeezewave.flow import diff_infer
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="_unused/m3_sqzw_ckpts/ckpts_aux/sqzwaux_step40000.pth")
    ap.add_argument("--out", default="_unused/m3_sqzw_ckpts/ckpts_aux/sqzwaux_step40000_bnrecal.pth")
    ap.add_argument("--trainlist", default="data/filelist_train.txt")
    ap.add_argument("--batches", type=int, default=60)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = PiperMelFeatures().to(dev)
    ck = torch.load(a.ckpt, map_location=dev)
    m = SqueezeWave(**ck["config"]).to(dev); m.load_state_dict(ck["model"])
    nbn = 0
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.reset_running_stats(); mod.momentum = None; nbn += 1
    m.train()
    dl = DataLoader(WavSet(a.trainlist, 16384), batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    it = iter(dl); done = 0
    with torch.no_grad():
        while done < a.batches:
            try: audio = next(it)
            except StopIteration: it = iter(dl); audio = next(it)
            audio = audio.to(dev); diff_infer(m, feat(audio), 1.0); done += 1
    m.eval()
    # save in STANDARD format (weight_norm + recalibrated BN, NOT fused)
    torch.save({"model": m.state_dict(), "opt": ck.get("opt", {}),
                "step": ck["step"], "config": ck["config"]}, a.out)
    print(f"recalibrated {nbn} BN ({a.batches} batches) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
