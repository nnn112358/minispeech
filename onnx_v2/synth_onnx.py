#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "onnxruntime",
#     "numpy",
#     "soundfile",
#     "piper-plus-g2p==0.2.0",
#     "pyopenjtalk-plus==0.4.1.post8",
# ]
# ///
"""ONNX-only TTS inference (text -> encoder.onnx -> mel -> vocoder.onnx -> wav).

PyTorch 不要。依存は onnxruntime + numpy + soundfile + piper-plus-g2p のみで、
すべて **uv が自動解決** する (上の PEP 723 インラインメタデータ)。
任意の encoder × vocoder ONNX を組み合わせて推論できる。SqueezeWave 系
(追加入力 `z` を要するボコーダ) にも対応 — z は自動でガウスノイズを供給する。

g2p は piper-plus-g2p (PyPI) の JA-only パス (intersperse なし) を使い、
先頭に BOS=1 を付与する。特定の venv (piper_ft 等) への依存はない。

実行 (依存は初回に自動インストール):
  uv run synth_onnx.py --text "おはようございます" --out out.wav
  ./synth.sh --text "おはようございます"          # 同梱ラッパー (uv run のショートカット)

例:
  # encoder/vocoder を明示
  uv run synth_onnx.py --encoder enc_d192_encoder.onnx \
      --vocoder vocos_d256_n512_vocoder.onnx --text "..." --out out.wav

  # SqueezeWave ボコーダ (z 自動供給)
  uv run synth_onnx.py --vocoder sqzw_f8c64_vocoder.onnx --text "..." --sigma 0.7

  # 音素ID直接指定 (g2p 不要)
  uv run synth_onnx.py --phonemes 1,10,14,8,38,10,11,10,8,2

  # ベンチ (10回平均, RTF)
  uv run synth_onnx.py --text "..." --bench

  # GPU 実行 (onnxruntime-gpu が必要 — synth.sh が依存を切り替えて起動)
  ./synth.sh --gpu --text "..." --encoder enc_d192_encoder_slim_fp16.onnx

ライブラリとしても利用可:
  from synth_onnx import OnnxTTS, text_to_phoneme_ids
  tts = OnnxTTS("enc_d192_encoder.onnx", "vocos_d256_n512_vocoder.onnx", provider="auto")
  result = tts.synthesize(text_to_phoneme_ids("こんにちは"))
  result.audio  # np.float32 [-1, 1], 22.05kHz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf

SR = 22050
HOP = 256  # PiperMelFeatures hop_length (mel frame -> audio sample)
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_ENCODER = "enc_d192_encoder_slim_fp16.onnx"
DEFAULT_VOCODER = "vocos_d256_n512_vocoder_slim_fp16.onnx"
BENCH_RUNS = 10


# ── path / session helpers ───────────────────────────────────────────

def _resolve(path: str) -> str:
    """相対パスはスクリプトのあるディレクトリ基準で解決 (cwd 非依存)。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    cand = os.path.join(HERE, path)
    return cand if os.path.exists(cand) else path


def _resolve_providers(provider: str) -> list[str]:
    """`provider` (auto|cpu|cuda) から onnxruntime の実行プロバイダ列を決める。

    CUDA を使うには onnxruntime-gpu が必要 (通常の onnxruntime と共存不可)。
    GPU 実行は `./synth.sh --gpu ...` が依存を切り替えて起動する。
    """
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in avail:
            sys.exit(
                "ERROR: CUDAExecutionProvider が使えません (onnxruntime-gpu 未導入)。\n"
                "  GPU で実行するには:  ./synth.sh --gpu --text ...\n"
                "  (ラッパーが onnxruntime-gpu を用意して起動します)"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # auto: 使えるなら CUDA、なければ CPU
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _make_session(onnx_path: str, providers: list[str], threads: int = 1):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = threads
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(onnx_path, opts, providers=providers)


def _timed(fn):
    """fn() を実行し (戻り値, 経過ミリ秒) を返す。"""
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


# ── g2p (piper-plus-g2p, JA-only) ────────────────────────────────────

def text_to_phoneme_ids(text: str) -> list[int]:
    """JA-only g2p (intersperse なし) + BOS=1 prepend。piper-plus-g2p (PyPI) のみで完結。

    PiperEncoder.encode() は音素間に blank token を挿入する (intersperse) ため使わず、
    JA phonemizer のトークンを id_map で直接引く。これは piper_train の
    text_to_phoneme_ids_and_prosody の JA-only パスと bit-identical な出力になる
    (検証済み)。
    """
    try:
        from piper_plus_g2p import get_phonemizer
        from piper_plus_g2p.encode.id_maps import get_phoneme_id_map
        from piper_plus_g2p.encode.pua import map_token
    except ImportError as e:
        sys.exit(
            f"ERROR: piper-plus-g2p を import できません ({e}).\n"
            "uv で実行してください (依存は自動解決されます):\n"
            "  uv run synth_onnx.py --text ...   (または ./synth.sh --text ...)\n"
            "(g2p なしで試すなら --phonemes でID列を直接指定)"
        )
    id_map = get_phoneme_id_map("ja")
    phonemes, _ = get_phonemizer("ja").phonemize_with_prosody(text)
    ids = [1]  # BOS
    for ph in phonemes:
        for ch in map_token(ph):
            if ch in id_map:
                ids.extend(id_map[ch])
    return ids


# ── inference engine ─────────────────────────────────────────────────

@dataclass
class SynthResult:
    mel: np.ndarray          # (1, 80, T)
    audio: np.ndarray        # float32 [-1, 1], mono 22.05kHz
    enc_ms: float            # encoder 推論時間
    voc_ms: float            # vocoder 推論時間

    @property
    def total_ms(self) -> float:
        return self.enc_ms + self.voc_ms

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SR


class OnnxTTS:
    """encoder ONNX (phoneme_ids->mel) + vocoder ONNX (mel->audio) の2段推論。"""

    def __init__(self, encoder_path: str, vocoder_path: str, *,
                 provider: str = "auto", threads: int = 1, sigma: float = 0.7, seed: int = 0):
        providers = _resolve_providers(provider)
        self.encoder = _make_session(encoder_path, providers, threads)
        self.vocoder = _make_session(vocoder_path, providers, threads)
        self.encoder_name = os.path.basename(encoder_path)
        self.vocoder_name = os.path.basename(vocoder_path)
        self.device = self.encoder.get_providers()[0].replace("ExecutionProvider", "")
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def synthesize(self, phoneme_ids: list[int]) -> SynthResult:
        ph = np.asarray([phoneme_ids], dtype=np.int64)
        mel, enc_ms = _timed(lambda: self.encoder.run(None, {"phoneme_ids": ph})[0])
        feeds = self._vocoder_feeds(mel)
        out, voc_ms = _timed(lambda: self.vocoder.run(None, feeds)[0])
        audio = np.clip(np.asarray(out).squeeze(), -1.0, 1.0).astype(np.float32)
        return SynthResult(mel=mel, audio=audio, enc_ms=enc_ms, voc_ms=voc_ms)

    def _vocoder_feeds(self, mel: np.ndarray) -> dict:
        """mel と (必要なら) ノイズ z を vocoder の入力名に合わせて組み立てる。

        SqueezeWave 系は追加入力 `z [1, C, T_z]` を要する。ONNX wrapper は内部で
        sigma=1.0 を掛けるため、torch infer(sigma) 相当にするには z を sigma 倍して渡す。
        audio_len = T*HOP = C*T_z より T_z = T*HOP//C。
        """
        feeds = {}
        T = mel.shape[2]
        for inp in self.vocoder.get_inputs():
            if inp.name == "mel":
                feeds["mel"] = mel.astype(np.float32)
            elif inp.name == "z":
                C = inp.shape[1] if isinstance(inp.shape[1], int) else 256  # n_audio_channel
                T_z = T * HOP // C
                feeds["z"] = self.rng.standard_normal((1, C, T_z)).astype(np.float32) * self.sigma
            else:
                raise ValueError(f"未知の vocoder 入力: {inp.name}")
        return feeds


# ── CLI ──────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ONNX-only TTS (text -> mel -> wav)")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER, help="encoder ONNX (phoneme_ids -> mel)")
    ap.add_argument("--vocoder", default=DEFAULT_VOCODER, help="vocoder ONNX (mel -> audio)")
    ap.add_argument("--text", default="", help="日本語テキスト (g2p: piper-plus-g2p)")
    ap.add_argument("--phonemes", default="", help="カンマ区切り phoneme_ids (--text の代わり)")
    ap.add_argument("--out", default="onnx_v2_out.wav")
    ap.add_argument("--sigma", type=float, default=0.7, help="SqueezeWave のノイズスケール")
    ap.add_argument("--seed", type=int, default=0, help="z ノイズの乱数シード (再現性)")
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime スレッド数 (CPU)")
    ap.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto",
                    help="実行プロバイダ (auto: GPU があれば使用)")
    ap.add_argument("--gpu", action="store_true", help="--provider cuda の別名 (GPU 実行)")
    ap.add_argument("--bench", action="store_true", help="10回実行して平均時間/RTFを表示")
    return ap.parse_args(argv)


def _phoneme_ids_from_args(args: argparse.Namespace) -> list[int]:
    if args.text:
        return text_to_phoneme_ids(args.text)
    if args.phonemes:
        return [int(x) for x in args.phonemes.replace(" ", "").split(",") if x]
    sys.exit("ERROR: --text か --phonemes を指定してください")


def _write_wav(path: str, audio: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, SR)


def _report(tts: OnnxTTS, n_phonemes: int, result: SynthResult, out_path: str) -> None:
    dur = result.duration_s
    print(f"device={tts.device}  encoder={tts.encoder_name}  vocoder={tts.vocoder_name}")
    print(f"phonemes={n_phonemes}  mel={tuple(result.mel.shape)}  "
          f"audio={len(result.audio)} ({dur:.2f}s)")
    print(f"  encoder: {result.enc_ms:.1f}ms  vocoder: {result.voc_ms:.1f}ms  "
          f"total: {result.total_ms:.1f}ms  RTF: {result.total_ms / (dur * 1000):.4f}")
    print(f"  -> {out_path}")


def _bench(tts: OnnxTTS, phoneme_ids: list[int], dur_s: float, n: int = BENCH_RUNS) -> None:
    runs = [tts.synthesize(phoneme_ids) for _ in range(n)]
    te = float(np.mean([r.enc_ms for r in runs]))
    tv = float(np.mean([r.voc_ms for r in runs]))
    print(f"\n  bench ({n}x): encoder {te:.1f}ms  vocoder {tv:.1f}ms  "
          f"total {te + tv:.1f}ms  RTF {(te + tv) / (dur_s * 1000):.4f}")


def main(argv=None) -> None:
    args = _parse_args(argv)

    enc_path, voc_path = _resolve(args.encoder), _resolve(args.vocoder)
    for p in (enc_path, voc_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: ONNX が見つかりません: {p}")

    phoneme_ids = _phoneme_ids_from_args(args)
    provider = "cuda" if args.gpu else args.provider
    tts = OnnxTTS(enc_path, voc_path, provider=provider,
                  threads=args.threads, sigma=args.sigma, seed=args.seed)

    result = tts.synthesize(phoneme_ids)
    _write_wav(args.out, result.audio)
    _report(tts, len(phoneme_ids), result, args.out)

    if args.bench:
        _bench(tts, phoneme_ids, result.duration_s)


if __name__ == "__main__":
    main()
