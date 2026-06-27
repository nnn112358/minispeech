#!/usr/bin/env bash
# ONNX TTS 推論ラッパー。
#   CPU (既定): ./synth.sh --text "..."        -> PEP 723 (onnxruntime CPU) を uv が解決
#   GPU       : ./synth.sh --gpu --text "..."   -> onnxruntime-gpu に切り替えて起動
#
# onnxruntime と onnxruntime-gpu は共存できないため、GPU 時は PEP 723 を使わず
# (python をコマンドにして synth_onnx.py を引数として渡す) 依存を別系統で解決する。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU 指定 (--gpu / --provider cuda) を検出
want_gpu=0; prev=""
for a in "$@"; do
  [ "$a" = "--gpu" ] && want_gpu=1
  [ "$prev" = "--provider" ] && [ "$a" = "cuda" ] && want_gpu=1
  prev="$a"
done

if [ "$want_gpu" = "1" ]; then
  exec uv run --no-project \
    --with "onnxruntime-gpu==1.22.0" \
    --with numpy --with soundfile \
    --with "piper-plus-g2p==0.2.0" --with "pyopenjtalk-plus==0.4.1.post8" \
    python "$HERE/synth_onnx.py" "$@"
else
  exec uv run --script "$HERE/synth_onnx.py" "$@"
fi
