#!/usr/bin/env python3
"""ONNX-only TTS inference: text -> encoder ONNX -> vocoder ONNX -> wav.
No PyTorch dependency at runtime (only onnxruntime + numpy + soundfile).

Usage:
  python src/cli/synth_onnx.py --encoder onnx/enc_d192_vocos_d128_n512_encoder.onnx \
                                --vocoder onnx/vocos_d128_n512_vocoder.onnx \
                                --text "おはようございます" --out outputs/onnx_test.wav
"""
import os, sys, time, argparse
import numpy as np
import soundfile as sf

SR = 22050


def create_session(onnx_path, threads=1):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = threads
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])


def text_to_phoneme_ids(text):
    from piper_plus_g2p.encode.id_maps import get_phoneme_id_map
    from piper_train.infer_onnx import text_to_phoneme_ids_and_prosody
    id_map = get_phoneme_id_map("ja")
    ids, _ = text_to_phoneme_ids_and_prosody(text, id_map, language="ja", language_id_map=None)
    return [1] + ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="encoder ONNX (phoneme_ids -> mel)")
    ap.add_argument("--vocoder", required=True, help="vocoder ONNX (mel -> audio)")
    ap.add_argument("--text", default="", help="Japanese text (g2p via OpenJTalk)")
    ap.add_argument("--phonemes", default="", help="comma-separated phoneme_ids")
    ap.add_argument("--out", default="outputs/onnx_tts.wav")
    ap.add_argument("--threads", type=int, default=1, help="ORT threads")
    ap.add_argument("--bench", action="store_true", help="run 10x and report avg time")
    a = ap.parse_args()

    encoder_sess = create_session(a.encoder, a.threads)
    vocoder_sess = create_session(a.vocoder, a.threads)

    if a.text:
        ph_ids = text_to_phoneme_ids(a.text)
    elif a.phonemes:
        ph_ids = [int(x) for x in a.phonemes.replace(" ", "").split(",") if x]
    else:
        print("ERROR: specify --text or --phonemes"); sys.exit(1)

    ph = np.array([ph_ids], dtype=np.int64)

    def run_once():
        t0 = time.perf_counter()
        mel = encoder_sess.run(None, {"phoneme_ids": ph})[0]
        t_encoder = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        audio = vocoder_sess.run(None, {"mel": mel.astype(np.float32)})[0].squeeze()
        t_vocoder = (time.perf_counter() - t0) * 1000

        return audio, mel, t_encoder, t_vocoder

    audio, mel, t_encoder, t_vocoder = run_once()
    audio = np.clip(audio, -1, 1)
    dur_s = len(audio) / SR

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out, audio, SR)
    t_total = t_encoder + t_vocoder
    print(f"phonemes={len(ph_ids)} mel={mel.shape} audio={len(audio)} ({dur_s:.2f}s)")
    print(f"  encoder: {t_encoder:.1f}ms  vocoder: {t_vocoder:.1f}ms  total: {t_total:.1f}ms  RTF: {t_total/(dur_s*1000):.4f}")
    print(f"  -> {a.out}")

    if a.bench:
        n = 10
        times = []
        for _ in range(n):
            _, _, te, tv = run_once()
            times.append((te, tv))
        te_avg = np.mean([t[0] for t in times])
        tv_avg = np.mean([t[1] for t in times])
        print(f"\n  bench ({n}x): encoder {te_avg:.1f}ms  vocos {tv_avg:.1f}ms  "
              f"total {te_avg+tv_avg:.1f}ms  RTF {(te_avg+tv_avg)/(dur_s*1000):.4f}")


if __name__ == "__main__":
    main()
