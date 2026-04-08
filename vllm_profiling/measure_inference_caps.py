#!/usr/bin/env python3
"""
用 vLLM 真机（近似）测量 Helix 里的 prompt_max_requests / decode_max_tokens 量级。

做法（与 probe_vllm_capacity 一致：临时改 num_hidden_layers、可选 dummy）：
  1) Prefill：同一批里 n 条请求，每条 prompt 长度 = --prefill-tokens，只生成 1 个 token；
     二分最大的 n → 对应 Helix 的 prompt_max_requests_dict[该层数] 的实测上界（在固定 prompt 长度下）。
  2) Decode：n 条请求，prefill 很短，每条 max_tokens = --decode-output-tokens（ignore_eos）；
     二分最大的 n → 对应 decode_max_tokens_dict[该层数] 的实测上界（在固定 decode 步数下）。

注意：
  - 每次二分尝试都会新建 LLM 进程/引擎再关掉，避免 OOM 后 GPU 状态异常；因此较慢。
  - load_format=dummy 时与延迟 profiling 一致，但和真实权重的绝对数值可能仍有偏差。
  - vLLM 约束：prompt 长度 + max_tokens 不得超过 max_model_len；prefill 探测用 max_tokens=1，故默认每条 prompt 最长为 max_model_len-1。
  - 请在 helix_run/vllm_profiling 目录下执行：python measure_inference_caps.py ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from probe_vllm_capacity import (
    _derived_max_model_len,
    _destroy_llm,
    _try_build_llm,
    _write_n_layer_config,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _collect_gpu_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_device_count"] = torch.cuda.device_count()
            out["cuda_device_0_name"] = torch.cuda.get_device_name(0)
            out["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as e:
        out["cuda_error"] = str(e)
    return out


def _read_vocab_token_id(model_dir: Path, token_id: int) -> int:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vs = int(cfg["vocab_size"])
    if token_id < 0 or token_id >= vs:
        return min(100, vs - 1)
    return token_id


def _run_single_attempt(
    *,
    model_dir: Path,
    num_hidden_layers: int,
    dtype: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    load_format: str,
    max_model_len: int,
    enforce_eager: bool,
    n_seqs: int,
    run_generate: Callable[[Any, int], Tuple[bool, str]],
) -> Tuple[bool, str, Dict[str, Any]]:
    """新建 stub LLM，对 n_seqs 跑 run_generate(llm, n_seqs)，然后销毁。"""
    tmp = tempfile.mkdtemp(prefix="helix_cap_")
    try:
        stub = Path(tmp) / "model"
        _write_n_layer_config(model_dir, stub, num_hidden_layers)
        ok, err, llm, kv = _try_build_llm(
            model_stub_dir=stub,
            tokenizer_dir=model_dir,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            load_format=load_format,
            enforce_eager=enforce_eager,
        )
        if not ok or llm is None:
            return False, err or "build failed", {}
        try:
            g_ok, g_err = run_generate(llm, n_seqs)
            return g_ok, g_err, kv
        finally:
            _destroy_llm(llm)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _max_feasible_n(
    *,
    model_dir: Path,
    num_hidden_layers: int,
    common: Dict[str, Any],
    max_model_len: int,
    search_hi: int,
    abs_max_concurrency: int,
    run_generate: Callable[[Any, int], Tuple[bool, str]],
    label: str,
) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    """
    求最大可行并发 n。若 initial search_hi 成功，会继续倍增直到失败再二分，避免把 search_hi 误当成真实上限。
    返回 (best_n, log, meta)；meta 可能含 hit_abs_cap。
    """
    log: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"hit_abs_cap": False}

    def attempt(n: int) -> bool:
        t0 = time.perf_counter()
        ok, err, kv = _run_single_attempt(
            model_dir=model_dir,
            num_hidden_layers=num_hidden_layers,
            dtype=common["dtype"],
            gpu_memory_utilization=common["gpu_memory_utilization"],
            tensor_parallel_size=common["tensor_parallel_size"],
            load_format=common["load_format"],
            max_model_len=max_model_len,
            enforce_eager=common["enforce_eager"],
            n_seqs=n,
            run_generate=run_generate,
        )
        elapsed = time.perf_counter() - t0
        log.append(
            {
                "n": n,
                "ok": ok,
                "error": err if not ok else None,
                "elapsed_sec": round(elapsed, 3),
                "kv_at_init": kv,
                "label": label,
            }
        )
        logging.info("[%s] n=%d -> %s (%.1fs)", label, n, "OK" if ok else "FAIL", elapsed)
        return ok

    cap = max(2, abs_max_concurrency)
    if not attempt(1):
        return 0, log, meta

    hi_probe = min(max(1, search_hi), cap)
    ans = 1

    if hi_probe <= 1:
        return ans, log, meta

    if not attempt(hi_probe):
        lo, hi2 = 2, hi_probe - 1
        while lo <= hi2:
            mid = (lo + hi2) // 2
            if attempt(mid):
                ans = mid
                lo = mid + 1
            else:
                hi2 = mid - 1
        return ans, log, meta

    ans = hi_probe
    cur = hi_probe
    while cur < cap:
        nxt = min(cur * 2, cap)
        if nxt <= cur:
            break
        if not attempt(nxt):
            lo, hi3 = ans + 1, nxt - 1
            while lo <= hi3:
                mid = (lo + hi3) // 2
                if attempt(mid):
                    ans = mid
                    lo = mid + 1
                else:
                    hi3 = mid - 1
            return ans, log, meta
        ans = nxt
        cur = nxt

    if ans == cap:
        meta["hit_abs_cap"] = True
        meta["note"] = (
            f"最大并发达到 --abs-max-concurrency={cap} 仍成功；真实上限可能更大，请调大该参数再测。"
        )
    return ans, log, meta


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="实测 vLLM 在固定 num_hidden_layers 下的 prefill / decode 并发上界（供 Helix 字典参考）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument(
        "--num-hidden-layers",
        type=int,
        required=True,
        help="与 probe 一致：stub 里的层数，例如 probe 得到的 24",
    )
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--load-format", default="dummy")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument(
        "--search-hi",
        type=int,
        default=128,
        help="首次探测的并发上界；若该值仍成功会继续倍增直到失败（不超过 --abs-max-concurrency）",
    )
    p.add_argument(
        "--abs-max-concurrency",
        type=int,
        default=1024,
        help="并发搜索硬顶，防止 dummy 下过宽导致尝试次数爆炸",
    )
    p.add_argument(
        "--prefill-tokens",
        type=int,
        default=None,
        help="prefill 实验每条请求的 token 数；默认 min(2048, max_model_len-1)，"
        "因 vLLM 要求 prompt_len + max_new_tokens <= max_model_len（本脚本 prefill 用 max_tokens=1）",
    )
    p.add_argument("--decode-prefill-tokens", type=int, default=32)
    p.add_argument("--decode-output-tokens", type=int, default=256)
    p.add_argument("--token-id", type=int, default=100)
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("--no-enforce-eager", action="store_true")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    model_dir = args.model_dir.resolve()
    if not (model_dir / "config.json").is_file():
        logging.error("缺少 config.json: %s", model_dir)
        return 1

    derived = _derived_max_model_len(model_dir)
    allow_long = os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN") == "1"
    if args.max_model_len is None:
        max_model_len = derived
    else:
        max_model_len = args.max_model_len
        if max_model_len > derived and not allow_long:
            logging.warning("max_model_len 钳制为 config 推导值 %d", derived)
            max_model_len = derived

    # prefill 子实验固定只生成 1 个 token，须满足 prompt_len + 1 <= max_model_len
    prefill_output_tokens = 1

    prefill_tokens = args.prefill_tokens
    max_prompt_for_prefill = max_model_len - prefill_output_tokens
    if max_prompt_for_prefill < 1:
        logging.error("max_model_len=%d 过小，无法进行 prefill 探测（需至少 2）", max_model_len)
        return 1
    if prefill_tokens is None:
        prefill_tokens = min(2048, max_prompt_for_prefill)
    if prefill_tokens + prefill_output_tokens > max_model_len:
        logging.warning(
            "prefill_tokens(%d) + max_new_tokens(%d) > max_model_len(%d)，已钳制 prefill_tokens=%d",
            prefill_tokens,
            prefill_output_tokens,
            max_model_len,
            max_prompt_for_prefill,
        )
        prefill_tokens = max_prompt_for_prefill
    if prefill_tokens < 1:
        logging.error("prefill_tokens 必须 >= 1")
        return 1

    tok = _read_vocab_token_id(model_dir, args.token_id)

    common = {
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "load_format": args.load_format,
        "enforce_eager": not args.no_enforce_eager,
    }

    def make_prefill_runner(prompt_len: int, tid: int) -> Callable[[Any, int], Tuple[bool, str]]:
        def _run(llm: Any, n: int) -> Tuple[bool, str]:
            from vllm import SamplingParams

            sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            ids = [[tid] * prompt_len for _ in range(n)]
            try:
                llm.generate(ids, sampling_params=sp, use_tqdm=False)
                return True, ""
            except Exception as e:
                return False, repr(e)

        return _run

    def make_decode_runner(p_len: int, out_len: int, tid: int) -> Callable[[Any, int], Tuple[bool, str]]:
        def _run(llm: Any, n: int) -> Tuple[bool, str]:
            from vllm import SamplingParams

            need = p_len + out_len + 8
            if need > max_model_len:
                return False, f"need max_model_len>={need}, got {max_model_len}"
            sp = SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True)
            ids = [[tid] * p_len for _ in range(n)]
            try:
                llm.generate(ids, sampling_params=sp, use_tqdm=False)
                return True, ""
            except Exception as e:
                return False, repr(e)

        return _run

    all_logs: List[Dict[str, Any]] = []
    prefill_max = None
    decode_max = None
    prefill_meta: Dict[str, Any] = {}
    decode_meta: Dict[str, Any] = {}

    if not args.skip_prefill:
        prefill_max, lg, prefill_meta = _max_feasible_n(
            model_dir=model_dir,
            num_hidden_layers=args.num_hidden_layers,
            common=common,
            max_model_len=max_model_len,
            search_hi=args.search_hi,
            abs_max_concurrency=args.abs_max_concurrency,
            run_generate=make_prefill_runner(prefill_tokens, tok),
            label="prefill",
        )
        all_logs.extend(lg)
    if not args.skip_decode:
        decode_max, lg2, decode_meta = _max_feasible_n(
            model_dir=model_dir,
            num_hidden_layers=args.num_hidden_layers,
            common=common,
            max_model_len=max_model_len,
            search_hi=args.search_hi,
            abs_max_concurrency=args.abs_max_concurrency,
            run_generate=make_decode_runner(args.decode_prefill_tokens, args.decode_output_tokens, tok),
            label="decode",
        )
        all_logs.extend(lg2)

    payload = {
        "num_hidden_layers": args.num_hidden_layers,
        "max_model_len_used": max_model_len,
        "derived_max_model_len_from_config": derived,
        "prefill": None
        if args.skip_prefill
        else {
            "max_concurrent_requests": prefill_max,
            "tokens_per_request": prefill_tokens,
            "prefill_max_new_tokens": prefill_output_tokens,
            "vllm_length_rule": "prompt_len + max_new_tokens <= max_model_len",
            "probe_meta": prefill_meta,
            "helix_hint": "prompt_max_requests_dict[num_hidden_layers] 可设为 <= max_concurrent_requests（在相同 prefill 长度假设下）",
        },
        "decode": None
        if args.skip_decode
        else {
            "max_concurrent_sequences": decode_max,
            "prefill_tokens_per_seq": args.decode_prefill_tokens,
            "decode_output_tokens": args.decode_output_tokens,
            "probe_meta": decode_meta,
            "helix_hint": "decode_max_tokens_dict[num_hidden_layers] 可设为 <= max_concurrent_sequences（在相同 decode 步数假设下）",
        },
        "gpu": _collect_gpu_info(),
        "attempts": all_logs,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.skip_prefill:
        logging.info("Prefill 最大并发请求数（每条 %d tokens）: %s", prefill_tokens, prefill_max)
    if not args.skip_decode:
        logging.info(
            "Decode 最大并发序列数（prefill=%d, max_tokens=%d）: %s",
            args.decode_prefill_tokens,
            args.decode_output_tokens,
            decode_max,
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("已写入 %s", args.json_out)

    ok = True
    if not args.skip_prefill:
        ok = ok and (prefill_max is not None and prefill_max >= 1)
    if not args.skip_decode:
        ok = ok and (decode_max is not None and decode_max >= 1)
    return 0 if ok else 1


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    raise SystemExit(main())
