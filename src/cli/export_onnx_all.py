#!/usr/bin/env python3
"""Export MiniSpeechEncoder + vocoder to ONNX.

Supports all vocoder types: vocos, hifigan, mbistft.

Usage:
  python src/cli/export_onnx_all.py --fs-ckpt fs/enc_d192/fs_2000.pth \
      --voc-ckpt checkpoints/vocos_lite_c/vocos_last.pth --voc-type vocos --out onnx/
  python src/cli/export_onnx_all.py \
      --voc-ckpt checkpoints/hifigan_jsut_v2/hifigan_last.pth --voc-type hifigan --out onnx/

Naming convention: {type}_{config_key}_vocoder.onnx
  vocos_d128_n512, hifigan_ch256, mbistft_n64
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from encoder.efficient import encoder_from_config


# ── Encoder wrapper ─────────────────────────────────────────────────

class EncoderONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, phoneme_ids):
        return self.model.forward_infer(phoneme_ids)


# ── Vocoder wrappers (mel -> audio, skip PiperMelFeatures) ──────────

class VocosONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, mel):
        return self.head(self.backbone(mel))


class HiFiGANONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.hifigan = model.hifigan

    def forward(self, mel):
        return self.hifigan(mel)


class MBiSTFTONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mbistft = model.mbistft

    def forward(self, mel):
        fullband, _ = self.mbistft(mel)
        return fullband


# ── Helpers ──────────────────────────────────────────────────────────

def remove_weight_norm_recursive(module):
    for name, child in module.named_children():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except ValueError:
            pass
        remove_weight_norm_recursive(child)


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def onnx_export(wrap, inputs, path, opset, in_names, out_names, dyn_axes):
    torch.onnx.export(wrap, inputs, path, opset_version=opset,
                      input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn_axes, dynamo=False)
    print(f"  -> {path} ({os.path.getsize(path)/1e6:.1f} MB)")


# ── Export: encoder ──────────────────────────────────────────────────

def export_encoder(ckpt_path, out_path, opset=18):
    fck = torch.load(ckpt_path, map_location="cpu")
    sd = {k.replace("dp.net.3.", "dp.net.2."): v for k, v in fck["model"].items()}  # conv DP remap (no-op for efficient)
    n_sym = (sd.get("emb.weight") if "emb.weight" in sd else sd["encoder.emb.weight"]).shape[0]
    cfg = fck.get("config", {})

    fs = encoder_from_config(n_sym, cfg)
    fs.load_state_dict(sd)
    fs.eval()
    if cfg.get("arch") == "efficient":
        w = cfg.get("embed_dim", 128) // cfg.get("reduction", 2)
        key = f"enc_eff{w}"
        print(f"  encoder: efficient embed={cfg.get('embed_dim',128)} reduction={cfg.get('reduction',2)} "
              f"w={w} depth={cfg.get('depth',2)} n_sym={n_sym} params={count_params(fs):.2f}M")
    else:
        dim = cfg.get("dim", 256)
        key = f"enc_d{dim}"
        print(f"  encoder: d={dim} enc={cfg.get('n_enc',4)} dec={cfg.get('n_dec',4)} n_sym={n_sym} "
              f"params={count_params(fs):.2f}M")

    wrap = EncoderONNX(fs).eval()
    ph = torch.LongTensor([[1, 10, 14, 8, 38, 10, 11, 10, 8, 2]])
    with torch.no_grad():
        wrap(ph)

    onnx_export(wrap, (ph,), out_path, opset,
                ["phoneme_ids"], ["mel"],
                {"phoneme_ids": {1: "L"}, "mel": {2: "T"}})
    return key


# ── Export: vocos ────────────────────────────────────────────────────

def export_vocos(ckpt_path, out_path, opset=18):
    from decoders.vocos.generator import Generator as VGen
    vck = torch.load(ckpt_path, map_location="cpu")
    cfg = vck.get("config", {})
    dim = cfg.get("dim", 512)
    inter = cfg.get("intermediate_dim", 1536)
    n_layers = cfg.get("num_layers", 8)
    n_fft = cfg.get("n_fft", 1024)
    hop = cfg.get("hop_length", 256)

    voc = VGen(dim=dim, intermediate_dim=inter, num_layers=n_layers,
               n_fft=n_fft, hop_length=hop)
    voc.load_state_dict(vck["G"], strict=False)
    remove_weight_norm_recursive(voc)
    voc.eval()

    wrap = VocosONNX(voc).eval()
    mel = torch.randn(1, 80, 100)
    with torch.no_grad():
        audio = wrap(mel)
    print(f"  vocos: dim={dim} layers={n_layers} n_fft={n_fft} "
          f"params={count_params(voc):.2f}M  audio={tuple(audio.shape)}")

    onnx_export(wrap, (mel,), out_path, opset,
                ["mel"], ["audio"],
                {"mel": {2: "T"}, "audio": {1: "T_audio"}})
    return f"vocos_d{dim}_n{n_fft}"


# ── Export: hifigan ──────────────────────────────────────────────────

def export_hifigan(ckpt_path, out_path, opset=18):
    from decoders.hifigan.generator import Generator as HGen
    vck = torch.load(ckpt_path, map_location="cpu")
    init_ch = vck.get("init_channels", 256)

    voc = HGen(init_channels=init_ch)
    voc.load_state_dict(vck["G"], strict=False)
    voc.hifigan.remove_weight_norm()
    voc.eval()

    wrap = HiFiGANONNX(voc).eval()
    mel = torch.randn(1, 80, 100)
    with torch.no_grad():
        audio = wrap(mel)
    print(f"  hifigan: init_ch={init_ch} params={count_params(voc):.2f}M  "
          f"audio={tuple(audio.shape)}")

    onnx_export(wrap, (mel,), out_path, opset,
                ["mel"], ["audio"],
                {"mel": {2: "T"}, "audio": {2: "T_audio"}})
    return f"hifigan_ch{init_ch}"


# ── Export: mbistft ──────────────────────────────────────────────────

def export_mbistft(ckpt_path, out_path, opset=18):
    from decoders.mb_istft.generator import Generator as MBGen
    vck = torch.load(ckpt_path, map_location="cpu")
    init_ch = vck.get("init_channels", 256)
    n_fft = vck.get("n_fft", 16)

    voc = MBGen(init_channels=init_ch, n_fft=n_fft)
    voc.load_state_dict(vck["G"], strict=False)
    voc.mbistft.remove_weight_norm()
    voc.eval()

    wrap = MBiSTFTONNX(voc).eval()
    mel = torch.randn(1, 80, 100)
    with torch.no_grad():
        audio = wrap(mel)
    print(f"  mbistft: init_ch={init_ch} n_fft={n_fft} "
          f"params={count_params(voc):.2f}M  audio={tuple(audio.shape)}")

    onnx_export(wrap, (mel,), out_path, opset,
                ["mel"], ["audio"],
                {"mel": {2: "T"}, "audio": {2: "T_audio"}})
    return f"mbistft_n{n_fft}"


# ── Optimize: onnxslim + fp16 (optional, --slim / --fp16) ────────────
# Deps (onnxslim / onnxconverter-common) are imported lazily so plain export
# needs only torch+onnx. Install with: uv pip install onnxslim onnxconverter-common

def _need(mod, pip_name):
    try:
        return __import__(mod)
    except ImportError:
        raise SystemExit(f"ERROR: --slim/--fp16 には '{pip_name}' が必要: "
                         f"uv pip install onnxslim onnxconverter-common onnxruntime")


def _ort_loads(path):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    try:
        ort.InferenceSession(path, providers=["CPUExecutionProvider"]); return True
    except Exception as e:
        print(f"    load 検証 NG: {str(e).splitlines()[-1][:80]}"); return False


def _reconcile_cast_types(model):
    """fp16 変換器は posenc 等由来の Cast(to=FLOAT) の出力型だけ float16 に書き換え
    `to` 属性を放置するため ORT が型不整合で弾くことがある。各 Cast の `to` を宣言
    出力型へ揃える保険 (posenc を定数バッファ化した現行 encoder では 0 件)。"""
    import onnx
    from onnx import TensorProto
    g = model.graph
    decl = {vi.name: vi.type.tensor_type.elem_type
            for vi in list(g.input) + list(g.output) + list(g.value_info)}
    for init in g.initializer:
        decl[init.name] = init.data_type
    n = 0
    for node in g.node:
        if node.op_type != "Cast":
            continue
        out_t = decl.get(node.output[0])
        if out_t not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            continue
        for at in node.attribute:
            if at.name == "to" and at.i != out_t:
                at.i = out_t; n += 1
    return n


def optimize_file(path, do_slim, do_fp16):
    import contextlib, io as _io
    print(f"--- optimize {os.path.basename(path)} ({os.path.getsize(path)/1e6:.1f} MB)")
    stage = path
    if do_slim:
        onnxslim = _need("onnxslim", "onnxslim")
        dst = path[:-5] + "_slim.onnx"
        with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
            onnxslim.slim(path, dst)
        if _ort_loads(dst):
            print(f"  slim  -> {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.1f} MB)"); stage = dst
        else:
            print("  slim  FAILED — orig を使用")
    if do_fp16:
        import onnx
        _need("onnxconverter_common", "onnxconverter-common")
        from onnxconverter_common import float16
        suffix = "_slim_fp16" if stage != path else "_fp16"
        dst = path[:-5] + suffix + ".onnx"
        m16 = float16.convert_float_to_float16(onnx.load(stage), keep_io_types=True)
        nfix = _reconcile_cast_types(m16)
        onnx.save(m16, dst)
        if _ort_loads(dst):
            extra = f"  (Cast 型不整合 {nfix} 件修正)" if nfix else ""
            print(f"  fp16  -> {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.1f} MB){extra}")
        else:
            os.remove(dst); print("  fp16  ロード不可のためスキップ (slim 版を使用)")


# ── Main ─────────────────────────────────────────────────────────────

EXPORTERS = {
    "vocos": export_vocos,
    "hifigan": export_hifigan,
    "mbistft": export_mbistft,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-ckpt", default="", help="MiniSpeechEncoder checkpoint (omit for vocoder-only)")
    ap.add_argument("--voc-ckpt", default="", help="vocoder checkpoint (omit for encoder-only)")
    ap.add_argument("--voc-type", default="", choices=[""] + list(EXPORTERS.keys()))
    ap.add_argument("--out", default="onnx_v2/", help="output directory")
    ap.add_argument("--name", default="", help="override output name prefix")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--slim", action="store_true", help="onnxslim でグラフ最適化 (<name>_slim.onnx)")
    ap.add_argument("--fp16", action="store_true", help="float16 量子化 (<name>_slim_fp16.onnx, サイズ約1/2)")
    a = ap.parse_args()
    if not a.voc_ckpt and not a.fs_ckpt:
        ap.error("--voc-ckpt か --fs-ckpt のどちらかは必須")
    if a.voc_ckpt and not a.voc_type:
        ap.error("--voc-ckpt 指定時は --voc-type が必須")
    os.makedirs(a.out, exist_ok=True)
    produced = []

    # Export vocoder
    if a.voc_ckpt:
        tmp_voc = os.path.join(a.out, "_tmp_vocoder.onnx")
        print(f"=== Export {a.voc_type}: {a.voc_ckpt}")
        voc_key = EXPORTERS[a.voc_type](a.voc_ckpt, tmp_voc, a.opset)

        voc_final = os.path.join(a.out, f"{a.name or voc_key}_vocoder.onnx")
        os.replace(tmp_voc, voc_final)
        print(f"  renamed -> {voc_final}")
        produced.append(voc_final)

    # Export encoder (independent naming)
    if a.fs_ckpt:
        tmp_enc = os.path.join(a.out, "_tmp_encoder.onnx")
        print(f"=== Export encoder: {a.fs_ckpt}")
        enc_key = export_encoder(a.fs_ckpt, tmp_enc, a.opset)
        enc_final = os.path.join(a.out, f"{enc_key}_encoder.onnx")
        if os.path.exists(enc_final):
            os.remove(tmp_enc)
            print(f"  skip (already exists): {enc_final}")
        else:
            os.replace(tmp_enc, enc_final)
            print(f"  renamed -> {enc_final}")
        produced.append(enc_final)

    # Optimize (onnxslim / fp16) — opt-in
    if a.slim or a.fp16:
        print("\n=== Optimize ===")
        for p in produced:
            optimize_file(p, a.slim, a.fp16)

    print(f"\nDone. Files in {a.out}:")
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".onnx"):
            sz = os.path.getsize(os.path.join(a.out, f)) / 1e6
            print(f"  {f}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
