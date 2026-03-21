# Helix 外部 RTX5090 Profiling（LLaMA2-70B）

这个目录是独立于 `Helix-ASPLOS25` 的 profiling 工具，用于在**单卡**上生成 Helix 需要的：

- `prompt_bs2time.csv`
- `decode_bs2time.csv`

## 1) 为什么单卡也能做

完整 70B 放不下，但 Helix 的这两个 CSV 本质是**单层 latency 曲线**。做法是：

1. 复制 `config.json`，把 `num_hidden_layers` 改成 `1`
2. 用 **vLLM + `load_format=dummy`**（不加载真实权重）
3. 扫 batch/token 点位，写 CSV（第二列单位**毫秒**）

## 2) 环境

```bash
conda create -n helix-profile python=3.10 -y
conda activate helix-profile
pip install vllm==0.4.0.post1 transformers
```

## 3) 运行 profiling

```bash
cd /home/user/code/helix-rtx5090-profiling
python profile_llama2_70b_one_layer_vllm.py \
  --model-dir /home/user/code/Llama-2-70b-hf \
  --output-dir ./outputs \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --strict-llama2-70b \
  --warmup 2 \
  --repeat 3
```

也可直接 `bash run.sh`（按其中路径改 `model-dir`）。

## 3.1 结构一致性保证

`--strict-llama2-70b` 会校验 `config.json` 的 70B 关键字段，通过后再改为单层 profiling。

输出：

- `./outputs/prompt_bs2time.csv`
- `./outputs/decode_bs2time.csv`

## 4) 拷贝到 Helix 的 RTX5090 目录

```bash
cp ./outputs/prompt_bs2time.csv \
  /home/user/code/Helix-ASPLOS25/simulator/model_manager/llama2_70b/rtx5090/
cp ./outputs/decode_bs2time.csv \
  /home/user/code/Helix-ASPLOS25/simulator/model_manager/llama2_70b/rtx5090/
```

## 5) 校验点

- 第一列 `x` 与 Helix 内置点位一致；首行 `x=0` 为锚点
- 第二列单位**毫秒**；`prompt_bs2time` 第二列为**整数**；`decode_bs2time` 仍保留一位小数风格。`--min-latency-ms`（默认 `0.1`）做下界，避免出现 0
- 曲线不要求单调（与仓库里部分官方 CSV 一致）

## 6) 方法说明（重要）

- **decode_bs2time.csv**：第一列为并发 `bs`；第二列为长生成均摊的**单步 decode**（毫秒）：  
  `((t(max_tokens=K) - t(max_tokens=1)) / (K-1)) * 1000`，`K` 由 `--decode-long-tokens`（默认 1000）指定。

- **prompt_bs2time.csv**：对每个 batch 测 `max_tokens=1` 总延迟，再按 decode 表对同一 `bs` 插值减掉**一步 decode**，近似 prefill；同样经 `--min-latency-ms` 下界。

常用参数：`--decode-long-tokens`、`--min-latency-ms`、`--prompt-len-for-prompt`、`--prompt-len-for-decode`、`--repeat`。
