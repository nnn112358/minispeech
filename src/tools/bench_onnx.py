#!/usr/bin/env python3
"""ONNX Runtime CPU latency for a SqueezeWave vocoder ONNX (vs Vocos if present).
Feeds are auto-shaped from the model's declared inputs, so it works for any
exported config. Reports median mel->audio latency and RTF at 1 / all threads.
Export a model first with `python src/cli/export_onnx.py --out npu/model/squeezewave.onnx`."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, onnxruntime as ort

SR = 22050
AUD = 65536 / SR  # 2.972 s cap


def bench(path, feeds, threads, warmup=10, n=100):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for _ in range(warmup):
        sess.run(None, feeds)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return ts.min(), np.median(ts), ts.mean()


def feeds_for(path):
    """Random inputs matching the model's static input shapes."""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return {i.name: np.random.randn(*i.shape).astype(np.float32) for i in sess.get_inputs()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="npu/model/squeezewave.onnx", help="SqueezeWave ONNX (from cli/export_onnx.py)")
    ap.add_argument("--vocos", default="npu/model/vocoder_fix.onnx", help="Vocos baseline ONNX (optional)")
    a = ap.parse_args()
    models = [("SqueezeWave", a.onnx)]
    if os.path.exists(a.vocos):
        models.append(("Vocos", a.vocos))
    for threads in (1, 0):
        tlabel = "1 thread" if threads == 1 else "all threads"
        print(f"\n=== ONNX Runtime CPU, {tlabel}  (cap {AUD:.3f}s audio) ===")
        print(f"{'model':18} {'min':>8} {'median':>8} {'mean':>8}   {'RTF(med)':>9} {'xRT':>7}")
        for name, path in models:
            try:
                mn, md, mean = bench(path, feeds_for(path), threads)
                rtf = (md / 1000) / AUD
                print(f"{name:18} {mn:7.2f}m {md:7.2f}m {mean:7.2f}m   {rtf:9.4f} {1/rtf:6.0f}x")
            except Exception as e:
                print(f"{name:18} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
