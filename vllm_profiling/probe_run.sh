#!/usr/bin/env bash
# 真机探测（默认）：最大 num_hidden_layers + 二分
# 线性扫 num_gpu_blocks：加 SWEEP=1，或对 probe_run.sh 追加参数：
#   --sweep-blocks-min 1 --sweep-blocks-max 24 --json-out ./outputs/probe_blocks_sweep.json
# 用法：在 helix_run 下 source env.sh 后，cd vllm_profiling 再执行。

set -euo pipefail
cd "$(dirname "$0")"

BASE=(python probe_vllm_capacity.py
  --model-dir /home/user/code/helix_run/models/llama1_30b
  --gpu-memory-utilization 0.9
  --dtype float16
  --tensor-parallel-size 1)

if [[ "${SWEEP:-}" == "1" ]]; then
  "${BASE[@]}" \
    --sweep-blocks-min 1 \
    --sweep-blocks-max 24 \
    --json-out ./outputs/probe_blocks_sweep.json \
    "$@"
else
  "${BASE[@]}" \
    --cap-layers 60 \
    --json-out ./outputs/probe_capacity.json \
    "$@"
fi
