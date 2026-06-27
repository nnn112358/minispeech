#!/usr/bin/env python3
"""Export the actual TRAINED SqueezeWave checkpoints to ONNX and measure ORT CPU
inference time (256-frame / 2.972s cap), vs Vocos. Confirms real deployable speed
incl. dilation. Speed is weight-independent, so in-progress ckpts are fine."""
import sys, os, time, tempfile
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, onnxruntime as ort
from decoders.squeezewave.model import SqueezeWave
from decoders.squeezewave.onnx_export import SqueezeWaveONNX, FRAMES

SR = 22050; AUD = 65536 / SR
D = "checkpoints/"
RUNS = [
    ("a256_c128 (aux+BN)",      "ckpts_aux/sqzwaux_step40000_bnrecal.pth"),
    ("c64_f12_l8 (aux+BN)",     "ckpts_c64_f12_l8_aux/sqzwaux_bnrecal.pth"),
    ("c64_f8_l8+dil (aux)",     "ckpts_c64_f8_l8_dgan_aux/sqzwaux_last.pth"),
]


def export(ckpt, path):
    ck = torch.load(ckpt, map_location="cpu"); cfg = ck["config"]
    m = SqueezeWave(**cfg); m.load_state_dict(ck["model"]); m = SqueezeWave.remove_weightnorm(m).eval()
    nac = cfg["n_audio_channel"]; l = FRAMES * (256 // nac)
    wrap = SqueezeWaveONNX(m).eval()
    mel = torch.randn(1, 80, FRAMES); z = torch.randn(1, nac, l)
    with torch.no_grad(): wrap(mel, z)
    torch.onnx.export(wrap, (mel, z), path, opset_version=18, input_names=["mel", "z"], output_names=["audio"])
    npar = sum(p.numel() for p in m.parameters())
    return npar, l, nac, cfg["n_flows"], cfg["WN_config"]["n_channels"], cfg["WN_config"].get("dilation_cycle")


from decoders.squeezewave.bench import bench


def main():
    tmp = tempfile.mkdtemp(); mel = np.random.randn(1, 80, FRAMES).astype(np.float32)
    rows = []
    for label, ck in RUNS:
        p = f"{tmp}/{label.split()[0]}.onnx"
        npar, l, nac, nf, nc, dil = export(D + ck, p)
        z = np.random.randn(1, nac, l).astype(np.float32)
        md1 = bench(p, {"mel": mel, "z": z}, 1); mda = bench(p, {"mel": mel, "z": z}, 0)
        rows.append((label, npar, dil, md1, mda));
        print(f"  exported {label:24} params {npar/1e6:.2f}M dil={dil}  1t {md1:.1f}ms  allt {mda:.1f}ms", flush=True)
    md1 = bench("npu/model/vocoder_fix.onnx", {"mel": mel}, 1)
    mda = bench("npu/model/vocoder_fix.onnx", {"mel": mel}, 0)
    rows.append(("Vocos (current)", 13_500_000, None, md1, mda))

    print(f"\n{'model':26} {'params':>8} {'dil':>4} {'1thr ms':>8} {'1t xRT':>7} {'all ms':>7} {'vs Vocos(1t)':>12}")
    voc1 = rows[-1][3]
    for label, npar, dil, md1, mda in rows:
        print(f"{label:26} {npar/1e6:6.2f}M {str(dil):>4} {md1:7.1f} {AUD*1000/md1:6.0f}x {mda:6.1f} {voc1/md1:10.2f}x")


if __name__ == "__main__":
    main()
