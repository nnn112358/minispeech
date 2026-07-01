#!/usr/bin/env python3
"""Teacher-forced mel-L1 over the full manifest (GT durations).

Reproduces the training-loss computation (train_minispeech.py) in eval mode
over every utterance: model(ph, du, Tf) -> masked L1 / (n_frames*80).
Reports two aggregations:
  global : sum|.|*mask / (sum(mask)*80)         (frame-weighted, "全5000平均")
  per_utt: mean over utts of per-utt masked L1  (utterance-weighted)
Runs on CPU by default (no env prefix -> avoids the model-load silent-death gotcha)
so it does not compete with a GPU training job.
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from encoder.minispeech import FSSet, collate
from encoder.efficient import encoder_from_config


def evaluate(ckpt, manifest, device="cpu", batch_size=16, threads=4):
    torch.set_num_threads(threads)
    dev = torch.device(device)
    ck = torch.load(ckpt, map_location="cpu")
    sd = {k.replace("dp.net.3.", "dp.net.2."): v for k, v in ck["model"].items()}
    cfg = ck.get("config", {})
    n_sym = ck.get("n_sym") or (sd.get("emb.weight") if "emb.weight" in sd else sd["encoder.emb.weight"]).shape[0]
    model = encoder_from_config(n_sym, cfg).to(dev)
    model.load_state_dict(sd); model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6

    dl = DataLoader(FSSet(manifest), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    tot_abs = 0.0; tot_frames = 0.0; utt_sum = 0.0; n_utt = 0
    t0 = time.time()
    with torch.inference_mode():
        for ph, du, me, pl, ml in dl:
            ph, du, me = ph.to(dev), du.to(dev), me.to(dev)
            Tf = me.shape[2]
            fmask = (torch.arange(Tf, device=dev)[None, :] < ml.to(dev)[:, None]).unsqueeze(1)  # (B,1,Tf)
            mel, _ = model(ph, du, Tf)
            ae = (F.l1_loss(mel, me, reduction='none') * fmask)  # (B,80,Tf)
            tot_abs += ae.sum().item()
            tot_frames += fmask.sum().item()
            per_utt = ae.sum(dim=(1, 2)) / (ml.to(dev).clamp(min=1) * 80)  # (B,)
            utt_sum += per_utt.sum().item(); n_utt += ph.shape[0]
    return {
        "ckpt": ckpt, "params_M": params, "n_utt": n_utt,
        "global": tot_abs / (tot_frames * 80),
        "per_utt": utt_sum / n_utt,
        "sec": time.time() - t0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="checkpoint paths")
    ap.add_argument("--manifest", default="fs_data/fs_manifest.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    print(f"{'ckpt':<40} {'params':>7} {'global':>8} {'per_utt':>8} {'sec':>5}", flush=True)
    for c in a.ckpts:
        r = evaluate(c, a.manifest, a.device, a.batch_size, a.threads)
        print(f"{r['ckpt']:<40} {r['params_M']:>6.3f}M {r['global']:>8.4f} {r['per_utt']:>8.4f} {r['sec']:>5.0f}", flush=True)
