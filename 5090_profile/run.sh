python profile_llama2_70b_one_layer_vllm.py \
  --model-dir /home/user/code/helix_run/models/Llama-2-70b-hf \
  --output-dir ./outputs \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --strict-llama2-70b \
  # --warmup 5 \
  # --repeat 30