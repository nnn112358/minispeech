#!/bin/bash
# Real AX650 benchmark (axcl_run_model, native C) of the GAN-finished SqueezeWave
# vocoders vs Vocos. NPU3 / U16. Must run from /usr/bin/axcl (relative libs).
AXM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/npu"   # repo npu/ (absolute)
cd /usr/bin/axcl || exit 1   # axcl_run_model needs its own dir (relative libs)
for entry in \
  "c64_f8_l4:$AXM/output_sqzw_c64/sqzw_c64_f8_l4_s_AX650_u16_npu3.axmodel" \
  "a256_c128:$AXM/output_sqzw_a256/sqzw_a256_c128_q_AX650_u16_npu3.axmodel" \
  "Vocos:$AXM/output_npu3/vocoder_AX650_u16_npu3.axmodel" \
  ; do
  tag=${entry%%:*}; m=${entry#*:}
  echo "######## $tag ########"
  ./axcl_run_model -m "$m" -r 200 -w 20 -c /usr/bin/axcl/axcl.json 2>&1 \
    | grep -iE "min =|max =|avg =|average|fps|latency|Repeat|inference" | head -10
done
echo "######## NPU BENCH DONE ########"
