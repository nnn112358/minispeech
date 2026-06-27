#!/usr/bin/env python3
"""Speed frontier sweep for FURTHER speedup beyond a256_c128. Vary WN width (c),
n_flows, n_layers (all n_audio_channel=256, l=256). Weight-independent latency,
ORT CPU, 256-frame/2.972s cap, vs current a256_c128 and Vocos."""
import sys, time, tempfile
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, onnxruntime as ort
from decoders.squeezewave.model import SqueezeWave
from decoders.squeezewave.onnx_export import SqueezeWaveONNX, FRAMES

SR = 22050; AUD = 65536 / SR

# name: (n_channels, n_flows, n_layers)   [n_audio_channel fixed 256]
CONFIGS = {
    "c128_f12_l8 (current)": (128, 12, 8),
    "c64_f12_l8":  (64, 12, 8),
    "c64_f8_l8":   (64, 8, 8),
    "c64_f6_l8":   (64, 6, 8),
    "c64_f12_l4":  (64, 12, 4),
    "c64_f8_l4":   (64, 8, 4),
    "c96_f8_l8":   (96, 8, 8),
    "c48_f8_l8":   (48, 8, 8),
}


def mkcfg(nc, nf, nl):
    return dict(n_mel_channels=80, n_flows=nf, n_audio_channel=256,
                n_early_every=2, n_early_size=16,
                WN_config=dict(n_layers=nl, n_channels=nc, kernel_size=3))


def export(cfg, path):
    m = SqueezeWave(**cfg); m = SqueezeWave.remove_weightnorm(m); m.eval()
    l = FRAMES * (256 // cfg["n_audio_channel"])
    wrap = SqueezeWaveONNX(m).eval()
    mel = torch.randn(1, 80, FRAMES); z = torch.randn(1, 256, l)
    with torch.no_grad(): wrap(mel, z)
    torch.onnx.export(wrap, (mel, z), path, opset_version=18,
                      input_names=["mel", "z"], output_names=["audio"])
    return sum(p.numel() for p in m.parameters()), l


from decoders.squeezewave.bench import bench


def main():
    tmp = tempfile.mkdtemp()
    mel = np.random.randn(1, 80, FRAMES).astype(np.float32)
    rows = []
    for name, (nc, nf, nl) in CONFIGS.items():
        cfg = mkcfg(nc, nf, nl); path = f"{tmp}/{name.split()[0]}.onnx"
        npar, l = export(cfg, path)
        z = np.random.randn(1, 256, l).astype(np.float32)
        md1 = bench(path, {"mel": mel, "z": z}, 1)
        mda = bench(path, {"mel": mel, "z": z}, 0)
        rows.append((name, npar, md1, mda))
        print(f"  {name:22} params {npar/1e6:5.2f}M  1t {md1:6.1f}ms ({AUD*1000/md1:4.0f}xRT)  allt {mda:6.1f}ms", flush=True)
    md1 = bench("npu/model/vocoder_fix.onnx", {"mel": mel}, 1)
    mda = bench("npu/model/vocoder_fix.onnx", {"mel": mel}, 0)
    print(f"  {'Vocos (ref)':22} params 13.50M  1t {md1:6.1f}ms ({AUD*1000/md1:4.0f}xRT)  allt {mda:6.1f}ms", flush=True)
    print("\nspeedup vs current a256_c128:")
    cur = rows[0][2]
    for name, npar, md1, mda in rows:
        print(f"  {name:22} {cur/md1:4.2f}x  ({md1:.1f}ms)")


if __name__ == "__main__":
    main()
