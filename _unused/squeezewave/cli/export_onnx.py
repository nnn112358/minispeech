#!/usr/bin/env python3
"""Export the (patched) SqueezeWave vocoder to ONNX for the JSUT 2-stage TTS.
The flow's reverse pass needs Gaussian noise; for an ONNX/NPU-friendly static
graph we take that noise as an INPUT `z` (1, n_audio_channel, l) instead of
generating it inside (RandomNormal is unsupported by Pulsar2). Fixed cap of
256 mel frames -> 65536 samples (2.972 s), matching the Vocos vocoder_fix.onnx
so the two are directly comparable."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from decoders.squeezewave.model import SqueezeWave
from decoders.squeezewave.onnx_export import SqueezeWaveONNX, FRAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpts_c64_f8_l4_s_gan/sqzwgan_bnrecal.pth")
    ap.add_argument("--out", default="npu/model/squeezewave.onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    dev = torch.device("cpu")  # export on CPU; don't disturb GPU training
    ck = torch.load(a.ckpt, map_location=dev)
    cfg = ck["config"]
    model = SqueezeWave(**cfg).to(dev)
    model.load_state_dict(ck["model"])
    model = SqueezeWave.remove_weightnorm(model)
    model.eval()

    nac = cfg["n_audio_channel"]
    l = FRAMES * (256 // nac)          # audio groups
    wrap = SqueezeWaveONNX(model).eval()
    mel = torch.randn(1, 80, FRAMES)
    z = torch.randn(1, nac, l)
    with torch.no_grad():
        out = wrap(mel, z)
    print(f"step {ck['step']}  cfg n_audio_channel={nac} n_remaining={model.n_remaining_channels} "
          f"l={l}  out={tuple(out.shape)}", flush=True)

    torch.onnx.export(
        wrap, (mel, z), a.out, opset_version=a.opset,
        input_names=["mel", "z"], output_names=["audio"],
        dynamic_axes=None,   # fully static for NPU
    )
    sz = os.path.getsize(a.out) / 1e6
    print(f"exported -> {a.out}  ({sz:.1f} MB)", flush=True)
    # quick onnx checker
    import onnx
    onnx.checker.check_model(onnx.load(a.out))
    print("onnx.checker OK", flush=True)


if __name__ == "__main__":
    main()
