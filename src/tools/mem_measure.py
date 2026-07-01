#!/usr/bin/env python3
"""Net peak-RSS memory of an encoder as deployed (fp16 ONNX, CPU, 1 thread).

net RSS = peak RSS of a 1-thread ONNX-Runtime inference process minus the ORT
runtime floor (import + a weightless session), i.e. the model-attributable memory
(fp16 weights + activation arena + op-kernel buffers).

Four modes (the driver spawns the others as isolated subprocesses):
  driver (default) : per checkpoint -> export fp16 ONNX (own subprocess, keeps
                     torch out of the driver so forked children's ru_maxrss is
                     clean) -> measure -> net RSS = measure - floor.
  export <ckpt> <out> : build fp16 ONNX (torch); print "FP16 <path>".
  measure <onnx>   : load 1-thread, run the fixed phoneme seq; print
                     "RSS_KB <maxrss> T <mel_frames>".
  mkbase <path>    : build a trivial phoneme_ids->cast model used, via the same
                     measure path, as the import-matched runtime floor.

Runs entirely on CPU (no CUDA, no env prefix) so it never disturbs GPU training.
"""
import os, sys, json, argparse, subprocess, resource, tempfile

# 1-thread determinism for the activation-arena measurement. Set before ORT import.
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# A fixed phoneme sequence used for every model (apples-to-apples workload).
# Picked at driver time from the manifest (median-length utt) and passed via a
# JSON sidecar so the measure subprocesses all read the same input.
FIXED_PH_FILE = os.path.join(tempfile.gettempdir(), "mem_measure_ph.json")


def _maxrss_kb():
    # ru_maxrss is in KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _ort_session(path):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Disable the CPU memory arena + mem-pattern: the arena pre-reserves large
    # quantised chunks that swamp the (tiny) real weight+activation differences
    # between these models. Off => peak RSS tracks actual allocation.
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def mode_measure(onnx_path):
    import numpy as np
    ph = json.load(open(FIXED_PH_FILE))
    x = np.array([ph], dtype=np.int64)
    sess = _ort_session(onnx_path)
    T = 0
    for _ in range(3):  # warm + steady; peak RSS is monotonic
        out = sess.run(None, {"phoneme_ids": x})
        T = out[0].shape[-1]
    print(f"RSS_KB {_maxrss_kb()} T {T}")


def mode_mkbase(path):
    """Build a trivial phoneme_ids->cast model so the floor can be measured
    through the SAME measure path as real models (import-matched: numpy+ort only)."""
    import onnx
    from onnx import helper, TensorProto
    vi_in = helper.make_tensor_value_info("phoneme_ids", TensorProto.INT64, [1, "L"])
    vi_out = helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, 1, "L"])
    cast = helper.make_node("Cast", ["phoneme_ids"], ["f"], to=TensorProto.FLOAT)
    usq = helper.make_node("Unsqueeze", ["f", "ax"], ["mel"])
    ax = helper.make_tensor("ax", TensorProto.INT64, [1], [1])
    g = helper.make_graph([cast, usq], "floor", [vi_in], [vi_out], initializer=[ax])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(m, path)
    print(f"MKBASE {path}")


def mode_export(ckpt, out_onnx):
    """Export encoder -> slim -> fp16 at an explicit path (avoids CLI naming
    collisions). Runs as its OWN subprocess so torch never loads in the driver
    (a torch-heavy driver would poison every forked measure child's ru_maxrss)."""
    import contextlib, io
    from cli.export_onnx_all import export_encoder, optimize_file
    base = out_onnx[:-5] + "_raw.onnx"
    with contextlib.redirect_stdout(io.StringIO()):
        export_encoder(ckpt, base)          # writes fp32 raw
        optimize_file(base, do_slim=True, do_fp16=True)  # writes *_raw_slim_fp16.onnx
    fp16 = base[:-5] + "_slim_fp16.onnx"
    if not os.path.exists(fp16):        # slim or fp16 fell back
        fp16 = base[:-5] + "_fp16.onnx"
    print(f"FP16 {fp16}")


def mode_driver(ckpts, manifest, workdir):
    os.makedirs(workdir, exist_ok=True)
    # fixed input: median-length utterance from the manifest
    items = json.load(open(manifest))
    items_sorted = sorted(items, key=lambda it: it.get("n_frames", len(it["phoneme_ids"])))
    ph = items_sorted[len(items_sorted) // 2]["phoneme_ids"]
    json.dump(ph, open(FIXED_PH_FILE, "w"))
    print(f"fixed input: {len(ph)} phonemes (median-length utt)", flush=True)

    def run_sub(args):
        r = subprocess.run([sys.executable, os.path.abspath(__file__)] + args,
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RSS_KB")]
        if not line:
            print("  SUBPROC ERR:", r.stdout[-300:], r.stderr[-300:])
            return None
        return line[0]

    # floor: trivial phoneme_ids->cast model measured through the SAME path
    # (import-matched). = ORT runtime + numpy + a 1-thread session, no weights.
    base_onnx = os.path.join(workdir, "_floor.onnx")
    subprocess.run([sys.executable, os.path.abspath(__file__), "mkbase", base_onnx],
                   capture_output=True, text=True)
    base_vals = []
    for _ in range(3):
        b = run_sub(["measure", base_onnx])
        if b:
            base_vals.append(int(b.split()[1]))
    baseline = sorted(base_vals)[len(base_vals) // 2] if base_vals else 0
    print(f"ORT runtime floor: {baseline/1024:.1f} MB (median of {len(base_vals)})", flush=True)

    print(f"\n{'ckpt':<28} {'peak MB':>8} {'net MB':>8} {'onnx MB':>8} {'T':>5}", flush=True)
    results = []
    for ckpt in ckpts:
        name = ckpt.split("/")[-2] if "/" in ckpt else ckpt
        onnx_path = os.path.join(workdir, name + ".onnx")
        # export in its OWN subprocess so torch never taints the driver's RSS
        er = subprocess.run([sys.executable, os.path.abspath(__file__), "export", ckpt, onnx_path],
                            capture_output=True, text=True)
        fpl = [l for l in er.stdout.splitlines() if l.startswith("FP16")]
        if not fpl:
            print(f"{name:<28} EXPORT ERR {er.stderr[-120:]}", flush=True); continue
        fp16 = fpl[0].split(None, 1)[1]
        onnx_mb = os.path.getsize(fp16) / 1e6
        # measure a few times, take median peak
        vals = []; Tval = 0
        for _ in range(3):
            m = run_sub(["measure", fp16])
            if m:
                parts = m.split(); vals.append(int(parts[1])); Tval = int(parts[3])
        if not vals:
            print(f"{name:<28} MEASURE ERR", flush=True); continue
        peak = sorted(vals)[len(vals) // 2]
        net_mb = (peak - baseline) / 1024
        print(f"{name:<28} {peak/1024:>8.1f} {net_mb:>8.2f} {onnx_mb:>8.3f} {Tval:>5}", flush=True)
        results.append({"ckpt": ckpt, "name": name, "peak_kb": peak,
                        "net_mb": net_mb, "onnx_mb": onnx_mb, "T": Tval})
    json.dump({"baseline_kb": baseline, "results": results},
              open(os.path.join(workdir, "mem_results.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(workdir, 'mem_results.json')}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "measure":
        mode_measure(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "export":
        mode_export(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "mkbase":
        mode_mkbase(sys.argv[2])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("ckpts", nargs="+")
        ap.add_argument("--manifest", default="fs_data/fs_manifest.json")
        ap.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "mem_measure_onnx"),
                        help="scratch dir for exported fp16 ONNX + results json")
        a = ap.parse_args()
        mode_driver(a.ckpts, a.manifest, a.workdir)
