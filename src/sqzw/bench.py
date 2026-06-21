"""ONNX Runtime CPU latency helper (was duplicated across onnx_speed /
onnx_eval_trained / config_sweep / frontier_sweep / stacked_bar)."""
import time
import numpy as np
import onnxruntime as ort


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def bench(path, feeds, threads=1, warmup=8, n=60):
    """Median latency [ms] over n runs. feeds auto-filtered to the model inputs."""
    sess = make_session(path, threads)
    feeds = {i.name: feeds[i.name] for i in sess.get_inputs()}
    for _ in range(warmup):
        sess.run(None, feeds)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))
