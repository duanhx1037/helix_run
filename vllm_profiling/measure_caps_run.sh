#!/usr/bin/env bash
# 实测 Helix 用的 prefill / decode 并发上界（需与 probe 使用相同的 num_hidden_layers）
# 用法：cd helix_run/vllm_profiling && bash measure_caps_run.sh
# 可覆盖：NUM_LAYERS=24 bash measure_caps_run.sh

set -euo pipefail
cd "$(dirname "$0")"

NUM_LAYERS="${NUM_LAYERS:-24}"

python measure_inference_caps.py \
  --model-dir /home/user/code/helix_run/models/llama1_30b \
  --num-hidden-layers "${NUM_LAYERS}" \
  --gpu-memory-utilization 0.9 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --json-out ./outputs/inference_caps.json \
  "$@"
