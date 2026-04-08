for n in 2 4 6 8 10 12 14 16 18 20 22 24; do
  echo "=== sweep n=$n ==="
  python probe_vllm_capacity.py \
    --model-dir /home/user/code/helix_run/models/llama1_30b \
    --gpu-memory-utilization 0.9 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --sweep-blocks-min "$n" \
    --sweep-blocks-max "$n" \
    --json-out "./outputs/probe_blocks_even_${n}.json"
  sleep 3
done
