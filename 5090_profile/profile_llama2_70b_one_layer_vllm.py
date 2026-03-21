#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import tempfile
import time
from typing import Dict, List

from vllm import LLM, SamplingParams


PROMPT_POINTS = [
    0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250,
    3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500,
    6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750,
    10000, 10250, 10500, 10750, 11000, 11250
]
DECODE_POINTS = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
    75, 80, 85, 90, 95, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
    340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
    660, 680, 700, 720, 740, 760, 780, 800
]


def _measure_generate_seconds(llm: LLM, prompt_token_ids: List[List[int]], sampling_params: SamplingParams,
                              warmup: int, repeat: int) -> float:
    for _ in range(max(0, warmup)):
        llm.generate(prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)

    samples = []
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        llm.generate(prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return statistics.mean(samples)


def _validate_llama2_70b_structure(cfg: Dict) -> None:
    required = {
        "model_type": "llama",
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
    }
    for k, v in required.items():
        if cfg.get(k) != v:
            raise ValueError(f"config mismatch: expected {k}={v}, got {cfg.get(k)}")
    if cfg.get("num_hidden_layers") != 80:
        raise ValueError(
            f"config mismatch: expected num_hidden_layers=80 for llama2-70b, got {cfg.get('num_hidden_layers')}"
        )


def _format_ms_for_csv(ms: float) -> str:
    v = round(ms, 1)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


def _format_prompt_ms_int_for_csv(ms: float) -> str:
    """prompt_bs2time second column: integer ms only (rounded)."""
    v = int(round(max(0.0, ms)))
    if v == 0:
        v = 1
    return str(v)


def _floor_latency_ms(ms: float, min_ms: float) -> float:
    return max(min_ms, ms)


def _interpolate_decode_step_ms(decode_rows: List[tuple[int, float]], bs: int) -> float:
    xs = [x for x, _ in decode_rows]
    ys = [y for _, y in decode_rows]
    if bs in xs:
        return ys[xs.index(bs)]
    if bs <= xs[0]:
        return ys[0]
    if bs >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i - 1] < bs < xs[i]:
            x0, y0 = xs[i - 1], ys[i - 1]
            x1, y1 = xs[i], ys[i]
            return y0 + (y1 - y0) * (bs - x0) / (x1 - x0)
    return ys[-1]


def _make_one_layer_model_config(model_dir: str, strict_llama2_70b: bool) -> str:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if strict_llama2_70b:
        _validate_llama2_70b_structure(cfg)
    cfg["num_hidden_layers"] = 1

    temp_dir = tempfile.mkdtemp(prefix="llama2_70b_1layer_")
    with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return temp_dir


def profile(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    prompt_csv = os.path.join(args.output_dir, "prompt_bs2time.csv")
    decode_csv = os.path.join(args.output_dir, "decode_bs2time.csv")

    with open(os.path.join(args.model_dir, "config.json"), "r", encoding="utf-8") as f:
        full_cfg = json.load(f)
    vocab_size = int(full_cfg["vocab_size"])
    one_layer_cfg_dir = _make_one_layer_model_config(args.model_dir, args.strict_llama2_70b)

    llm = LLM(
        model=one_layer_cfg_dir,
        tokenizer=args.model_dir,
        load_format="dummy",
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(
            args.prompt_len_for_prompt,
            args.prompt_len_for_decode + args.decode_long_tokens,
        )
        + 8,
        trust_remote_code=True,
        enforce_eager=True,
    )

    # Use a stable regular token id.
    token_id = args.token_id
    if token_id < 0 or token_id >= vocab_size:
        token_id = 100

    one_token = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    if args.decode_long_tokens <= 1:
        raise ValueError("--decode-long-tokens must be > 1")
    long_decode = SamplingParams(max_tokens=args.decode_long_tokens, temperature=0.0, ignore_eos=True)
    min_ms = float(args.min_latency_ms)

    # Decode: long-run amortized per-step estimate (original style):
    #   A = t(max_tokens=1), B = t(max_tokens=K)  ->  (B-A)/(K-1)  [seconds] -> ms
    decode_rows = [(0, min_ms)]
    for decode_tokens in DECODE_POINTS[1:]:
        bs = decode_tokens
        prompt_token_ids = [[token_id] * args.prompt_len_for_decode for _ in range(bs)]
        t_short = _measure_generate_seconds(llm, prompt_token_ids, one_token, args.warmup, args.repeat)
        t_long = _measure_generate_seconds(llm, prompt_token_ids, long_decode, args.warmup, args.repeat)
        per_step_sec = (t_long - t_short) / (args.decode_long_tokens - 1)
        decode_ms = _floor_latency_ms(per_step_sec * 1000.0, min_ms)
        decode_rows.append((decode_tokens, decode_ms))

    # Prompt: t(max_tokens=1) minus one decode step (interpolated from decode table at same bs).
    prompt_rows = [(0, min_ms)]
    for total_prompt_tokens in PROMPT_POINTS[1:]:
        if total_prompt_tokens % args.prompt_len_for_prompt != 0:
            continue
        bs = total_prompt_tokens // args.prompt_len_for_prompt
        prompt_token_ids = [[token_id] * args.prompt_len_for_prompt for _ in range(bs)]
        t_with_one_decode = _measure_generate_seconds(llm, prompt_token_ids, one_token, args.warmup, args.repeat)
        decode_step_ms = _interpolate_decode_step_ms(decode_rows, bs)
        prompt_only_ms = _floor_latency_ms(t_with_one_decode * 1000.0 - decode_step_ms, min_ms)
        prompt_rows.append((total_prompt_tokens, prompt_only_ms))

    with open(prompt_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for x, y in prompt_rows:
            writer.writerow([x, _format_prompt_ms_int_for_csv(_floor_latency_ms(y, min_ms))])
    with open(decode_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for x, y in decode_rows:
            writer.writerow([x, _format_ms_for_csv(_floor_latency_ms(y, min_ms))])

    print(f"Wrote: {prompt_csv}")
    print(f"Wrote: {decode_csv}")
    print(
        f"Done: decode = (t(K)-t(1))/(K-1) per bs, K={args.decode_long_tokens}; "
        f"prompt = t(1) minus decode step; min_latency_ms={min_ms}."
    )


def parse_args():
    p = argparse.ArgumentParser(description="Profile LLaMA2-70B one-layer latency for Helix CSVs using vLLM.")
    p.add_argument("--model-dir", required=True, help="HF model dir containing config.json + tokenizer files")
    p.add_argument("--output-dir", default="./outputs", help="Directory for prompt_bs2time.csv/decode_bs2time.csv")
    p.add_argument("--dtype", default="float16", help="vLLM dtype, e.g. float16/bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--prompt-len-for-prompt", type=int, default=250, help="Prompt len used for prompt profiling")
    p.add_argument(
        "--prompt-len-for-decode",
        type=int,
        default=1,
        help="Per-request prompt token count for decode profiling; concurrency = CSV first column",
    )
    p.add_argument(
        "--decode-long-tokens",
        type=int,
        default=10,
        help="Decode: max_tokens=K with max_tokens=1 baseline; per-step ms ~= (t_K-t_1)/(K-1)",
    )
    p.add_argument(
        "--min-latency-ms",
        type=float,
        default=0.1,
        help="Floor every CSV latency (ms) to this positive value (>=0.1 recommended: 1-decimal CSV rounds 0.01 to 0)",
    )
    p.add_argument("--warmup", type=int, default=2, help="Warmup runs per point")
    p.add_argument("--repeat", type=int, default=50, help="Timed repeats per point (mean used)")
    p.add_argument("--token-id", type=int, default=100, help="Token id used to build synthetic prompts")
    p.add_argument(
        "--strict-llama2-70b",
        action="store_true",
        help="Validate config.json matches canonical LLaMA2-70B structure before profiling",
    )
    return p.parse_args()


if __name__ == "__main__":
    profile(parse_args())
