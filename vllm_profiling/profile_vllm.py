#!/usr/bin/env python3
"""
Helix 兼容的单层延迟 profiling（vLLM + dummy weights）。

对任意 HuggingFace 式目录（config.json + tokenizer）：将 num_hidden_layers 改为 1 后测量。
输出格式与 simulator/model_manager 下各模型 */ 目录 CSV 约定一致：
  - prompt_bs2time.csv：第一列总 prompt tokens，第二列整数毫秒
  - decode_bs2time.csv：第一列并发 bs，第二列单步 decode 毫秒（一位小数）

单机多卡：--tensor-parallel-size N（如 2 为 TP=2），进程需可见不少于 N 张 GPU；
不同 TP 的 CSV 勿混用。
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import random
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

# -----------------------------------------------------------------------------
# 与 Helix 仓库内置表一致的采样点（勿随意删减，否则仿真器插值范围不一致）
# -----------------------------------------------------------------------------
PROMPT_POINTS: List[int] = [
    0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250,
    3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500,
    6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750,
    10000, 10250, 10500, 10750, 11000, 11250,
]
DECODE_POINTS: List[int] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
    75, 80, 85, 90, 95, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
    340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
    660, 680, 700, 720, 740, 760, 780, 800,
]

Aggregate = Literal["median", "mean", "min"]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _try_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _aggregate(values: Sequence[float], how: Aggregate) -> float:
    if not values:
        return 0.0
    if how == "mean":
        return float(statistics.mean(values))
    if how == "min":
        return float(min(values))
    return float(statistics.median(values))


def _measure_generate_seconds(
    llm: Any,
    prompt_token_ids: List[List[int]],
    sampling_params: Any,
    *,
    warmup: int,
    repeat: int,
    aggregate: Aggregate,
    cuda_sync: bool,
) -> float:
    for _ in range(max(0, warmup)):
        if cuda_sync:
            _try_cuda_sync()
        llm.generate(prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
        if cuda_sync:
            _try_cuda_sync()

    samples: List[float] = []
    for _ in range(max(1, repeat)):
        if cuda_sync:
            _try_cuda_sync()
        t0 = time.perf_counter()
        llm.generate(prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
        if cuda_sync:
            _try_cuda_sync()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return _aggregate(samples, aggregate)


def _format_ms_decode_csv(ms: float) -> str:
    v = round(ms, 1)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


def _format_ms_prompt_csv(ms: float) -> str:
    """Helix prompt 表第二列为整数 ms；0 写成 1 避免下游除零/异常。"""
    v = int(round(max(0.0, ms)))
    return "1" if v == 0 else str(v)


def _floor_latency_ms(ms: float, min_ms: float) -> float:
    return max(min_ms, ms)


def _interpolate_decode_step_ms(decode_rows: List[Tuple[int, float]], bs: int) -> float:
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


def _write_one_layer_config(model_dir: Path, dest_dir: Path) -> None:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg["num_hidden_layers"] = 1
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _destroy_llm(llm: Any) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
    except Exception as e:
        logging.warning("llm engine shutdown failed: %s", e)
    del llm
    gc.collect()
    _try_cuda_sync()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class ProfileSettings:
    model_dir: Path
    output_dir: Path
    dtype: str
    gpu_memory_utilization: float
    tensor_parallel_size: int
    prompt_len_for_prompt: int
    prompt_len_for_decode: int
    decode_long_tokens: int
    min_latency_ms: float
    warmup: int
    repeat: int
    token_id: int
    aggregate: Aggregate
    cuda_sync: bool
    seed: int


def _build_llm(one_layer_cfg_dir: Path, tok: Path, s: ProfileSettings) -> Any:
    from vllm import LLM

    max_len = max(
        s.prompt_len_for_prompt,
        s.prompt_len_for_decode + s.decode_long_tokens,
    ) + 32
    return LLM(
        model=str(one_layer_cfg_dir),
        tokenizer=str(tok),
        load_format="dummy",
        dtype=s.dtype,
        tensor_parallel_size=s.tensor_parallel_size,
        gpu_memory_utilization=s.gpu_memory_utilization,
        max_model_len=max_len,
        trust_remote_code=True,
        enforce_eager=True,
    )


def _write_manifest(path: Path, settings: ProfileSettings, extra: Dict[str, Any]) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": asdict(settings),
        **extra,
    }
    # Path 不可 JSON 序列化
    payload["settings"]["model_dir"] = str(settings.model_dir)
    payload["settings"]["output_dir"] = str(settings.output_dir)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _collect_gpu_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_device_count"] = torch.cuda.device_count()
            out["cuda_device_name"] = torch.cuda.get_device_name(0)
            out["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as e:
        out["cuda_error"] = str(e)
    return out


def run_profile(s: ProfileSettings) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    # vLLM V1 用 get_context(VLLM_WORKER_MULTIPROC_METHOD) 起 EngineCore，默认是 fork，与
    # multiprocessing.set_start_method 无关。父进程若已初始化 CUDA（如 main 里 device_count），
    # fork 子进程会报 “Cannot re-initialize CUDA in forked subprocess”。
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    if s.decode_long_tokens <= 1:
        raise ValueError("decode_long_tokens must be > 1")

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    full_cfg = json.loads((s.model_dir / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(full_cfg["vocab_size"])
    token_id = s.token_id
    if token_id < 0 or token_id >= vocab_size:
        logging.warning("token_id %s out of range [0,%s); using 100", token_id, vocab_size)
        token_id = 100

    random.seed(s.seed)
    try:
        import torch

        torch.manual_seed(s.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s.seed)
    except Exception:
        pass

    with tempfile.TemporaryDirectory(prefix="helix_llama_1layer_") as tmp:
        one_layer_dir = Path(tmp) / "model"
        _write_one_layer_config(s.model_dir, one_layer_dir)
        logging.info(
            "One-layer config at %s (tokenizer=%s), tensor_parallel_size=%d",
            one_layer_dir,
            s.model_dir,
            s.tensor_parallel_size,
        )

        llm: LLM = _build_llm(one_layer_dir, s.model_dir, s)
        try:
            one_token = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            long_decode = SamplingParams(
                max_tokens=s.decode_long_tokens, temperature=0.0, ignore_eos=True
            )

            min_ms = float(s.min_latency_ms)
            decode_rows: List[Tuple[int, float]] = [(0, min_ms)]

            decode_jobs = list(DECODE_POINTS[1:])
            logging.info("Decode points: %d (K=%d for amortized step)", len(decode_jobs), s.decode_long_tokens)
            for j, decode_tokens in enumerate(decode_jobs, start=1):
                bs = decode_tokens
                prompt_token_ids = [[token_id] * s.prompt_len_for_decode for _ in range(bs)]
                t_short = _measure_generate_seconds(
                    llm,
                    prompt_token_ids,
                    one_token,
                    warmup=s.warmup,
                    repeat=s.repeat,
                    aggregate=s.aggregate,
                    cuda_sync=s.cuda_sync,
                )
                t_long = _measure_generate_seconds(
                    llm,
                    prompt_token_ids,
                    long_decode,
                    warmup=s.warmup,
                    repeat=s.repeat,
                    aggregate=s.aggregate,
                    cuda_sync=s.cuda_sync,
                )
                per_step_sec = (t_long - t_short) / (s.decode_long_tokens - 1)
                decode_ms = _floor_latency_ms(per_step_sec * 1000.0, min_ms)
                decode_rows.append((decode_tokens, decode_ms))
                logging.info(
                    "decode [%d/%d] bs=%d per_step_ms=%.3f",
                    j,
                    len(decode_jobs),
                    bs,
                    decode_ms,
                )

            prompt_rows: List[Tuple[int, float]] = [(0, min_ms)]
            prompt_jobs = [
                n for n in PROMPT_POINTS[1:] if n % s.prompt_len_for_prompt == 0
            ]
            logging.info("Prompt points: %d (len_per_req=%d)", len(prompt_jobs), s.prompt_len_for_prompt)
            for j, total_prompt_tokens in enumerate(prompt_jobs, start=1):
                bs = total_prompt_tokens // s.prompt_len_for_prompt
                prompt_token_ids = [[token_id] * s.prompt_len_for_prompt for _ in range(bs)]
                t_with_one_decode = _measure_generate_seconds(
                    llm,
                    prompt_token_ids,
                    one_token,
                    warmup=s.warmup,
                    repeat=s.repeat,
                    aggregate=s.aggregate,
                    cuda_sync=s.cuda_sync,
                )
                decode_step_ms = _interpolate_decode_step_ms(decode_rows, bs)
                prompt_only_ms = _floor_latency_ms(t_with_one_decode * 1000.0 - decode_step_ms, min_ms)
                prompt_rows.append((total_prompt_tokens, prompt_only_ms))
                logging.info(
                    "prompt [%d/%d] tokens=%d bs=%d prompt_ms=%d (minus decode_step=%.3f)",
                    j,
                    len(prompt_jobs),
                    total_prompt_tokens,
                    bs,
                    int(round(prompt_only_ms)),
                    decode_step_ms,
                )

            return decode_rows, prompt_rows
        finally:
            _destroy_llm(llm)


def write_csvs(
    output_dir: Path,
    prompt_rows: List[Tuple[int, float]],
    decode_rows: List[Tuple[int, float]],
    min_ms: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_csv = output_dir / "prompt_bs2time.csv"
    decode_csv = output_dir / "decode_bs2time.csv"

    with prompt_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for x, y in prompt_rows:
            w.writerow([x, _format_ms_prompt_csv(_floor_latency_ms(y, min_ms))])

    with decode_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for x, y in decode_rows:
            w.writerow([x, _format_ms_decode_csv(_floor_latency_ms(y, min_ms))])

    logging.info("Wrote %s", prompt_csv)
    logging.info("Wrote %s", decode_csv)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile single-layer LLM latency for Helix (vLLM dummy weights).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Decode 列: 每步 ms ≈ (t(max_tokens=K)-t(max_tokens=1))/(K-1)。"
            "K 越大曲线越稳但越慢；建议 128~512。"
        ),
    )
    p.add_argument("--model-dir", type=Path, required=True, help="HF 模型目录（config.json + tokenizer）")
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="输出目录")
    p.add_argument("--dtype", default="float16", help="vLLM dtype，如 float16 / bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        metavar="N",
        help="vLLM tensor_parallel_size；>1 时需可见不少于 N 张 GPU（勿与 TP=1 的 CSV 混用）",
    )
    p.add_argument(
        "--prompt-len-for-prompt",
        type=int,
        default=250,
        help="prompt 表：每条请求的 prompt 长度；总 tokens = bs * 此值",
    )
    p.add_argument(
        "--prompt-len-for-decode",
        type=int,
        default=1,
        help="decode 表：每条请求的 prompt 长度；CSV 第一列为并发数",
    )
    p.add_argument(
        "--decode-long-tokens",
        type=int,
        default=256,
        help="decode 摊销用的长生成长度 K（见 epilog）",
    )
    p.add_argument(
        "--min-latency-ms",
        type=float,
        default=0.1,
        help="所有采样点的 ms 下界（避免写出 0 或过小噪声）",
    )
    p.add_argument("--warmup", type=int, default=2, help="每个采样点前 warmup 次数")
    p.add_argument("--repeat", type=int, default=5, help="计时的重复次数（取 aggregate）")
    p.add_argument(
        "--aggregate",
        choices=("median", "mean", "min"),
        default="median",
        help="对 repeat 次测量聚合方式；默认 median 抗离群",
    )
    p.add_argument(
        "--cuda-sync",
        action="store_true",
        help="每次 generate 前后 torch.cuda.synchronize()（更准但更慢）",
    )
    p.add_argument("--token-id", type=int, default=100, help="合成 prompt 使用的 token id")
    p.add_argument("--seed", type=int, default=0, help="随机种子（best-effort）")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG 日志")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验配置与点位数量，不加载 vLLM",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    settings = ProfileSettings(
        model_dir=args.model_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        prompt_len_for_prompt=args.prompt_len_for_prompt,
        prompt_len_for_decode=args.prompt_len_for_decode,
        decode_long_tokens=args.decode_long_tokens,
        min_latency_ms=args.min_latency_ms,
        warmup=args.warmup,
        repeat=args.repeat,
        token_id=args.token_id,
        aggregate=args.aggregate,
        cuda_sync=args.cuda_sync,
        seed=args.seed,
    )

    if not (settings.model_dir / "config.json").is_file():
        logging.error("Missing config.json under %s", settings.model_dir)
        return 1

    n_prompt = len([n for n in PROMPT_POINTS[1:] if n % settings.prompt_len_for_prompt == 0])
    logging.info(
        "Grid: %d decode points + %d prompt points",
        len(DECODE_POINTS) - 1,
        n_prompt,
    )

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="helix_dry_") as tmp:
            _write_one_layer_config(settings.model_dir, Path(tmp) / "m")
        logging.info("Dry-run OK.")
        return 0

    if settings.tensor_parallel_size < 1:
        logging.error("tensor_parallel_size must be >= 1, got %d", settings.tensor_parallel_size)
        return 1
    try:
        import torch

        if torch.cuda.is_available() and settings.tensor_parallel_size > torch.cuda.device_count():
            logging.error(
                "tensor_parallel_size=%d exceeds visible CUDA devices (%d); adjust CUDA_VISIBLE_DEVICES",
                settings.tensor_parallel_size,
                torch.cuda.device_count(),
            )
            return 1
    except Exception:
        pass

    t0 = time.perf_counter()
    decode_rows, prompt_rows = run_profile(settings)
    elapsed = time.perf_counter() - t0

    write_csvs(settings.output_dir, prompt_rows, decode_rows, settings.min_latency_ms)

    manifest_path = settings.output_dir / "profile_manifest.json"
    _write_manifest(
        manifest_path,
        settings,
        {
            "elapsed_sec": round(elapsed, 3),
            "prompt_points_written": len(prompt_rows),
            "decode_points_written": len(decode_rows),
            "gpu": _collect_gpu_info(),
        },
    )
    logging.info("Wrote %s (elapsed %.1fs)", manifest_path, elapsed)
    return 0


if __name__ == "__main__":
    # 与 vLLM 的 VLLM_WORKER_MULTIPROC_METHOD=spawn 配合；部分库仍依赖全局 start method。
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    raise SystemExit(main())
