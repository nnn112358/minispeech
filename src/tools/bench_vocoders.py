#!/usr/bin/env python3
"""Inference-speed benchmark for the mel->audio vocoders (ONNX Runtime CPU).
A vocoder's speed depends only on its architecture/compute, not its weights, so
HiFi-GAN is exported with random weights (no training needed) just to time it.
256-frame mel = 2.972 s audio cap, matching the SqueezeWave speed map."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from sqzw.hifigan_gen import HiFiGAN
from sqzw.bench import bench

FRAMES = 256
SR = 22050
AUD = FRAMES * 256 / SR  # 2.972 s


def export_hifigan(cfg, path):
    m = HiFiGAN(in_channels=80, **cfg).eval()
    m.remove_weight_norm()
    mel = torch.randn(1, 80, FRAMES)
    with torch.no_grad():
        m(mel)
    torch.onnx.export(m, mel, path, input_names=["mel"], output_names=["audio"],
                      opset_version=18, dynamic_axes=None)
    return sum(p.numel() for p in m.parameters())


def export_squeezewave(ckpt, path):
    from sqzw.model import SqueezeWave
    from sqzw.onnx_export import SqueezeWaveONNX
    ck = torch.load(ckpt, map_location="cpu"); cfg = ck["config"]
    model = SqueezeWave(**cfg); model.load_state_dict(ck["model"])
    model = SqueezeWave.remove_weightnorm(model).eval()
    nac = cfg["n_audio_channel"]; l = FRAMES * (256 // nac)
    wrap = SqueezeWaveONNX(model).eval()
    mel, z = torch.randn(1, 80, FRAMES), torch.randn(1, nac, l)
    with torch.no_grad():
        wrap(mel, z)
    torch.onnx.export(wrap, (mel, z), path, input_names=["mel", "z"], output_names=["audio"],
                      opset_version=18, dynamic_axes=None)
    return sum(p.numel() for p in model.parameters()), nac, l


def main():
    tmp = tempfile.gettempdir()
    rows = []
    mel_feed = lambda: np.random.randn(1, 80, FRAMES).astype(np.float32)
    for tag, cfg in [("HiFi-GAN (piper)", {})]:   # project HiFi-GAN == piper config
        p = f"{tmp}/hifigan_piper.onnx"
        n = export_hifigan(cfg, p)
        feeds = {"mel": mel_feed()}
        rows.append((tag, n, bench(p, feeds, 1), bench(p, feeds, 0)))
    vo = "npu/model/vocoder_fix.onnx"
    if os.path.exists(vo):
        feeds = {"mel": mel_feed()}
        rows.append(("Vocos", int(13.5e6), bench(vo, feeds, 1), bench(vo, feeds, 0)))
    for tag, ckpt in [("SqueezeWave a256", "checkpoints/ckpts_a256_c128_q_gan/sqzwgan_bnrecal.pth"),
                      ("SqueezeWave c64", "checkpoints/ckpts_c64_f8_l4_s_gan/sqzwgan_bnrecal.pth")]:
        if not os.path.exists(ckpt):
            continue
        p = f"{tmp}/{tag.replace(' ', '_')}.onnx"
        n, nac, l = export_squeezewave(ckpt, p)
        feeds = {"mel": mel_feed(), "z": np.random.randn(1, nac, l).astype(np.float32)}
        rows.append((tag, n, bench(p, feeds, 1), bench(p, feeds, 0)))

    print(f"\n=== Vocoder mel->audio speed (ONNX Runtime CPU, {FRAMES}-frame mel = {AUD:.2f}s audio) ===")
    print(f"{'model':14} {'params':>8} {'1-thread':>16} {'all-thread':>16}")
    for tag, n, m1, ma in rows:
        print(f"{tag:14} {n/1e6:6.2f}M  {m1:7.1f}ms ({AUD/(m1/1000):4.0f}xRT)  {ma:7.1f}ms ({AUD/(ma/1000):4.0f}xRT)")


if __name__ == "__main__":
    main()
