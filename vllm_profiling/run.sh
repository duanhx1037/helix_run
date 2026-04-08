python profile_vllm.py \
  --model-dir /home/user/code/helix_run/models/llama-2-30b \
  --output-dir ./outputs \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  # --warmup 5 \
  # --repeat 30