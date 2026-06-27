#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "onnx",
#     "onnxslim",
#     "onnxconverter-common",
#     "onnxruntime",
#     "numpy",
# ]
# ///
"""ONNX グラフ最適化 (onnxslim) + fp16 量子化ツール。

  uv run optimize_onnx.py enc_d192_encoder.onnx vocos_d256_n512_vocoder.onnx --fp16

各モデルに対して:
  1. onnxslim でグラフ最適化 (定数畳み込み・冗長ノード除去)。出力は数値不変。
  2. --fp16 指定時、float16 量子化 (keep_io_types=True で入出力は fp32 のまま)。
  3. 最適化後モデルが onnxruntime でロードできるか検証。fp16 が壊れる
     (Cast 型不整合等) モデルは fp16 をスキップし slim 版のみ残す。

出力命名: `<name>_slim.onnx`, `<name>_slim_fp16.onnx` (--no-slim 時は `<name>_fp16.onnx`)。

⚠️ 速度に関する注意 (CPU 実測):
  - fp16 は **CPU(CPUExecutionProvider) では高速化しない** (Cast 挿入 + fp16 ネイティブ
    演算が無く、同等〜遅い)。利点は **ファイルサイズ約1/2** と GPU/NPU での高速化。
  - onnxslim はエンコーダ等の制御フロー多めのグラフで効く。畳み込み主体のボコーダは
    onnxruntime 自身の最適化と被り中立なことが多い (出力は不変なので入れて損はない)。
  速度は `synth_onnx.py --bench` で各自の実機・実行プロバイダで確認すること。
"""
from __future__ import annotations

import argparse
import io
import contextlib
import os
import sys
import warnings


def _slim(src: str, dst: str) -> None:
    import onnxslim
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        onnxslim.slim(src, dst)


def _reconcile_cast_types(model) -> int:
    """fp16 変換器 (convert_float_to_float16) は posenc の `torch.arange(T).float()`
    由来の既存 Cast(to=FLOAT) ノードの *出力テンソル型* を float16 に書き換える一方で
    ノードの `to` 属性を FLOAT のまま放置するため、ORT が
    `Type (float16) of output ... does not match expected type (float)` で
    ロードに失敗する。各 Cast の `to` を変換後の宣言出力型に揃えて不整合を解消する。
    戻り値: 修正したノード数。"""
    from onnx import TensorProto
    g = model.graph
    declared = {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        declared[vi.name] = vi.type.tensor_type.elem_type
    for init in g.initializer:
        declared[init.name] = init.data_type
    n_fixed = 0
    for node in g.node:
        if node.op_type != "Cast":
            continue
        out_t = declared.get(node.output[0])
        if out_t not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            continue
        for a in node.attribute:
            if a.name == "to" and a.i != out_t:
                a.i = out_t
                n_fixed += 1
    return n_fixed


def _fp16(src: str, dst: str) -> None:
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(src)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    n = _reconcile_cast_types(m16)  # encoder posenc の Cast 型不整合を解消
    if n:
        print(f"  fp16       Cast 型不整合を {n} 件修正 (posenc .float())")
    onnx.save(m16, dst)


def _loads(path: str) -> bool:
    """onnxruntime でロード可能か (fp16 変換の型不整合などを検出)。"""
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return True
    except Exception as e:
        print(f"    load 検証 NG: {str(e).splitlines()[-1][:80]}")
        return False


def _mb(p: str) -> float:
    return os.path.getsize(p) / 1e6


def optimize(src: str, do_slim: bool, do_fp16: bool, outdir: str | None) -> None:
    if not os.path.exists(src):
        print(f"{src}: 見つかりません (skip)"); return
    base = os.path.basename(src)[:-5] if src.endswith(".onnx") else os.path.basename(src)
    out_dir = outdir or os.path.dirname(os.path.abspath(src))
    os.makedirs(out_dir, exist_ok=True)
    print(f"{base}.onnx  ({_mb(src):.1f} MB)")

    stage = src
    if do_slim:
        slim_p = os.path.join(out_dir, f"{base}_slim.onnx")
        _slim(stage, slim_p)
        if _loads(slim_p):
            print(f"  slim       -> {os.path.basename(slim_p)}  ({_mb(slim_p):.1f} MB)")
            stage = slim_p
        else:
            print("  slim       FAILED — orig を使用")

    if do_fp16:
        suffix = "_slim_fp16" if (do_slim and stage != src) else "_fp16"
        fp16_p = os.path.join(out_dir, f"{base}{suffix}.onnx")
        try:
            _fp16(stage, fp16_p)
        except Exception as e:
            print(f"  fp16       変換失敗: {str(e).splitlines()[-1][:70]}"); return
        if _loads(fp16_p):
            print(f"  fp16       -> {os.path.basename(fp16_p)}  ({_mb(fp16_p):.1f} MB)")
        else:
            os.remove(fp16_p)
            print("  fp16       このモデルは fp16 ロード不可のためスキップ (slim 版を使用)")


def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="ONNX 最適化 (onnxslim) + fp16 量子化")
    ap.add_argument("models", nargs="+", help="入力 ONNX (複数可)")
    ap.add_argument("--fp16", action="store_true", help="fp16 量子化も行う (サイズ約1/2)")
    ap.add_argument("--no-slim", action="store_true", help="onnxslim をスキップ")
    ap.add_argument("--out", default=None, help="出力ディレクトリ (既定: 入力と同じ場所)")
    a = ap.parse_args(argv)
    if a.no_slim and not a.fp16:
        sys.exit("ERROR: --no-slim 単独では何もしません (--fp16 を併用)")
    for src in a.models:
        optimize(src, do_slim=not a.no_slim, do_fp16=a.fp16, outdir=a.out)


if __name__ == "__main__":
    main()
