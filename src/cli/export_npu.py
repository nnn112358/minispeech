#!/usr/bin/env python3
"""Export GAN-finished SqueezeWave to SINGLE-FILE ONNX (Pulsar2-ready) and build
the z (noise) calibration tar. mel calib reuses axmodel/dataset/calib_mel.tar."""
import os, sys, tempfile, tarfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, onnx
from sqzw.model import SqueezeWave
from sqzw.onnx_export import SqueezeWaveONNX, FRAMES

OUTM = "npu/model"
OUTD = "npu/dataset"
CONFIGS = [
    ("a256_c128_q", "checkpoints/ckpts_a256_c128_q_gan/sqzwgan_bnrecal.pth"),
    ("c64_f8_l4_s", "checkpoints/ckpts_c64_f8_l4_s_gan/sqzwgan_bnrecal.pth"),
]


def export_single(ckpt, out):
    ck = torch.load(ckpt, map_location="cpu"); cfg = ck["config"]
    m = SqueezeWave(**cfg); m.load_state_dict(ck["model"]); m = SqueezeWave.remove_weightnorm(m).eval()
    nac = cfg["n_audio_channel"]; l = FRAMES * (256 // nac)
    wrap = SqueezeWaveONNX(m).eval(); mel = torch.randn(1, 80, FRAMES); z = torch.randn(1, nac, l)
    with torch.no_grad(): wrap(mel, z)
    tmp = tempfile.mkdtemp() + "/m.onnx"
    torch.onnx.export(wrap, (mel, z), tmp, opset_version=17, input_names=["mel", "z"], output_names=["audio"])
    # consolidate external data -> single file
    mdl = onnx.load(tmp, load_external_data=True)
    onnx.save(mdl, out, save_as_external_data=False)
    return nac, l, sum(p.numel() for p in m.parameters())


def main():
    for tag, ck in CONFIGS:
        out = f"{OUTM}/sqzw_{tag}.onnx"
        nac, l, npar = export_single(ck, out)
        sz = os.path.getsize(out) / 1e6
        print(f"{tag}: {out}  ({sz:.1f}MB single-file)  z[1,{nac},{l}]  {npar/1e6:.2f}M", flush=True)

    # z calibration tar (N(0,1) noise, shape [1,256,256])
    ztar = f"{OUTD}/calib_z.tar"
    tmp = tempfile.mkdtemp()
    rng = np.random.RandomState(0)
    names = []
    for i in range(32):
        a = rng.randn(1, 256, 256).astype(np.float32)
        p = f"{tmp}/{i:03d}.npy"; np.save(p, a); names.append(p)
    with tarfile.open(ztar, "w") as tf:
        for p in names: tf.add(p, arcname=os.path.basename(p))
    print(f"calib_z: {ztar} (32x [1,256,256] N(0,1))", flush=True)


if __name__ == "__main__":
    main()
