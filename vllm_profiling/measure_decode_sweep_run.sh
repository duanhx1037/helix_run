#!/usr/bin/env bash
# 按层扫描 decode_max_tokens（decode prefill=1，与 profile_vllm 一致）；须与 block probe 使用相同 gpu_memory_utilization
set -euo pipefail
cd "$(dirname "$0")"

python measure_decode_sweep.py \
  --model-dir /home/user/code/helix_run/models/llama1_30b \
  --layer-min 1 \
  --layer-max 24 \
  --gpu-memory-utilization 0.9 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --decode-output-tokens 256 \
  --merged-json ./outputs/decode_sweep_merged.json \
  "$@"
