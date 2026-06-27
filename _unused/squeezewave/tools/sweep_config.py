#!/usr/bin/env python3
"""Speed sweep over SqueezeWave configs (random weights — latency is weight-
independent) to pick the fastest before committing more training. Exports each
to ONNX and benches on ORT CPU at the 256-frame / 2.972s cap, vs Vocos."""
import sys, os, time, tempfile
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, onnxruntime as ort
from decoders.squeezewave.model import SqueezeWave
from decoders.squeezewave.onnx_export import SqueezeWaveONNX, FRAMES

SR = 22050; AUD = 65536 / SR

CONFIGS = {
    "a128_c128": (128, 128), "a128_c256": (128, 256),
    "a256_c128": (256, 128), "a256_c256": (256, 256),
    "a256_c64": (256, 64),   "a128_c64": (128, 64),
}
BASE = dict(n_mel_channels=80, n_flows=12, n_early_every=2, n_early_size=16)


def make_cfg(nac, nc):
    return dict(BASE, n_audio_channel=nac, WN_config=dict(n_layers=8, n_channels=nc, kernel_size=3))


def export(cfg, path):
    m = SqueezeWave(**cfg)
    m = SqueezeWave.remove_weightnorm(m); m.eval()
    nac = cfg["n_audio_channel"]; l = FRAMES * (256 // nac)
    wrap = SqueezeWaveONNX(m).eval()
    mel = torch.randn(1, 80, FRAMES); z = torch.randn(1, nac, l)
    with torch.no_grad():
        wrap(mel, z)   # warmup: precompute lazy W_inverse before tracing
    torch.onnx.export(wrap, (mel, z), path, opset_version=18,
                      input_names=["mel", "z"], output_names=["audio"])
    npar = sum(p.numel() for p in m.parameters())
    return npar, l


from decoders.squeezewave.bench import bench


def main():
    tmp = tempfile.mkdtemp()
    mel = np.random.randn(1, 80, FRAMES).astype(np.float32)
    rows = []
    for name, (nac, nc) in CONFIGS.items():
        path = f"{tmp}/{name}.onnx"
        npar, l = export(make_cfg(nac, nc), path)
        z = np.random.randn(1, nac, l).astype(np.float32)
        feeds = {"mel": mel, "z": z}
        md1 = bench(path, feeds, 1); mda = bench(path, feeds, 0)
        rows.append((f"SqueezeWave {name}", npar, md1, mda))
        print(f"  {name:11} params {npar/1e6:5.2f}M  l={l:4}  1t {md1:6.1f}ms  allt {mda:6.1f}ms", flush=True)
    # Vocos baseline
    vo = "npu/model/vocoder_fix.onnx"
    md1 = bench(vo, {"mel": mel}, 1); mda = bench(vo, {"mel": mel}, 0)
    rows.append(("Vocos (vocoder_fix)", 13_500_000, md1, mda))

    print(f"\n{'model':24} {'params':>8} {'1thr ms':>9} {'1thr xRT':>9} {'all ms':>8} {'all xRT':>9}")
    for name, npar, md1, mda in rows:
        print(f"{name:24} {npar/1e6:6.2f}M {md1:8.1f} {AUD*1000/md1:8.0f}x {mda:7.1f} {AUD*1000/mda:8.0f}x")


if __name__ == "__main__":
    main()
