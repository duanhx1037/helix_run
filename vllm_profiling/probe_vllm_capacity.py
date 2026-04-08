#!/usr/bin/env python3
"""
在真机上用 vLLM 探测「单进程视角」下能起的最大层数，以及（若可读到）KV block 数。

用途：给 Helix `llama1_30b_rtx5090.py` 里的 `max_num_layers`、`vllm_num_blocks_dict` 提供实测参考。

说明：
  - 默认与 profile_vllm.py 一致：`load_format=dummy`，按 config 结构起引擎；显存占用与「真权重」可能不完全一致，
    但可与现有单层 profiling 流程对齐。若要用真实权重，传 `--load-format auto`（需磁盘上有权重，且显存要够）。
  - 二分搜索每次成功/失败都会关掉引擎并 empty_cache，避免显存碎片影响下一次尝试。
  - max_model_len 默认从 config 读取（max_position_embeddings / model_max_length / max_sequence_length 等取最小），避免大于 vLLM 推导上限而报错。
  - 线性扫 num_gpu_blocks：同时传 --sweep-blocks-min N --sweep-blocks-max M，对每层各起一次引擎（慢，用于填 vllm_num_blocks_dict）。
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _destroy_llm(llm: Any) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
    except Exception as e:
        logging.warning("engine shutdown failed: %s", e)
    del llm
    gc.collect()
    _try_cuda_sync()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _derived_max_model_len(model_dir: Path) -> int:
    """
    估计 vLLM 会采用的上下文上限（取各字段中的最小值，避免大于推导长度而校验失败）。

    LLaMA1 常见只有 max_sequence_length；新卡/config 常有 max_position_embeddings、model_max_length。
    """
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    candidates: List[int] = []
    for key in (
        "max_position_embeddings",
        "model_max_length",
        "max_sequence_length",
        "seq_length",
    ):
        v = cfg.get(key)
        if v is not None:
            try:
                candidates.append(int(v))
            except (TypeError, ValueError):
                pass
    if not candidates:
        return 4096
    return min(candidates)


def _write_n_layer_config(model_dir: Path, dest_dir: Path, num_hidden_layers: int) -> None:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg["num_hidden_layers"] = int(num_hidden_layers)
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _extract_kv_info(llm: Any) -> Dict[str, Any]:
    """尽量从当前 vLLM 版本里读出 KV / block 相关信息。"""
    out: Dict[str, Any] = {}
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return out
    cc = getattr(engine, "cache_config", None)
    if cc is not None:
        for name in (
            "num_gpu_blocks",
            "num_cpu_blocks",
            "block_size",
            "swap_space_bytes",
        ):
            if hasattr(cc, name):
                try:
                    out[name] = getattr(cc, name)
                except Exception:
                    pass
    # 部分版本挂在 vllm_config 上
    vc = getattr(engine, "vllm_config", None)
    if vc is not None and "cache_config" not in str(type(vc)):
        try:
            c2 = getattr(vc, "cache_config", None)
            if c2 is not None and "num_gpu_blocks" not in out:
                out["num_gpu_blocks"] = getattr(c2, "num_gpu_blocks", None)
        except Exception:
            pass
    return out


def _try_build_llm(
    *,
    model_stub_dir: Path,
    tokenizer_dir: Path,
    dtype: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: int,
    load_format: str,
    enforce_eager: bool,
) -> Tuple[bool, str, Optional[Any], Dict[str, Any]]:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    try:
        from vllm import LLM

        llm = LLM(
            model=str(model_stub_dir),
            tokenizer=str(tokenizer_dir),
            load_format=load_format,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
        )
        info = _extract_kv_info(llm)
        return True, "", llm, info
    except Exception as e:
        return False, repr(e), None, {}


def probe_max_layers(
    *,
    model_dir: Path,
    dtype: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    load_format: str,
    max_model_len: int,
    enforce_eager: bool,
    cap_layers: int,
    linear: bool,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    返回 (max_ok_layers, attempt_log)。
    """
    full_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    total_in_config = int(full_cfg.get("num_hidden_layers", cap_layers))
    hi = min(cap_layers, total_in_config)
    log: List[Dict[str, Any]] = []

    def one_try(n: int) -> Tuple[bool, str, Dict[str, Any]]:
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"helix_probe_{n}L_") as tmp:
            stub = Path(tmp) / "model"
            _write_n_layer_config(model_dir, stub, n)
            ok, err, llm, kv_info = _try_build_llm(
                model_stub_dir=stub,
                tokenizer_dir=model_dir,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                load_format=load_format,
                enforce_eager=enforce_eager,
            )
            elapsed = time.perf_counter() - t0
            rec = {
                "num_hidden_layers": n,
                "ok": ok,
                "error": err if not ok else None,
                "elapsed_sec": round(elapsed, 3),
                "kv": kv_info,
            }
            log.append(rec)
            if llm is not None:
                _destroy_llm(llm)
            else:
                gc.collect()
                _try_cuda_sync()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            return ok, err, kv_info

    # 先快速试 1 层
    ok1, err1, _ = one_try(1)
    if not ok1:
        logging.error("连 1 层都无法初始化，请检查 CUDA / vLLM / 模型目录。错误: %s", err1)
        return 0, log

    if linear:
        best = 1
        for n in range(2, hi + 1):
            ok, err, kv = one_try(n)
            if ok:
                best = n
            else:
                logging.info("线性探测在 n=%d 失败，最大可用=%d。错误: %s", n, best, err[:500] if err else "")
                break
        return best, log

    # 二分：最大满足条件的 n
    if one_try(hi)[0]:
        return hi, log

    lo, ans = 1, 1
    mid_lo, mid_hi = 2, hi - 1
    while mid_lo <= mid_hi:
        mid = (mid_lo + mid_hi) // 2
        ok, err, _ = one_try(mid)
        if ok:
            ans = mid
            mid_lo = mid + 1
        else:
            logging.debug("mid=%d 失败: %s", mid, err[:300] if err else "")
            mid_hi = mid - 1
    return ans, log


def sweep_layer_gpu_blocks(
    *,
    model_dir: Path,
    dtype: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    load_format: str,
    max_model_len: int,
    enforce_eager: bool,
    layer_min: int,
    layer_max: int,
) -> List[Dict[str, Any]]:
    """
    对 n = layer_min..layer_max 逐个起 vLLM，记录每次初始化后的 num_gpu_blocks（失败也记一行）。
    """
    log: List[Dict[str, Any]] = []
    for n in range(layer_min, layer_max + 1):
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"helix_sweep_{n}L_") as tmp:
            stub = Path(tmp) / "model"
            _write_n_layer_config(model_dir, stub, n)
            ok, err, llm, kv_info = _try_build_llm(
                model_stub_dir=stub,
                tokenizer_dir=model_dir,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                load_format=load_format,
                enforce_eager=enforce_eager,
            )
            elapsed = time.perf_counter() - t0
            blocks = kv_info.get("num_gpu_blocks") if kv_info else None
            rec: Dict[str, Any] = {
                "num_hidden_layers": n,
                "ok": ok,
                "num_gpu_blocks": blocks,
                "error": err if not ok else None,
                "elapsed_sec": round(elapsed, 3),
                "kv": kv_info,
            }
            log.append(rec)
            logging.info(
                "sweep n=%d -> num_gpu_blocks=%s ok=%s (%.1fs)",
                n,
                blocks,
                ok,
                elapsed,
            )
            if llm is not None:
                _destroy_llm(llm)
            else:
                gc.collect()
                _try_cuda_sync()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
    return log


def _format_vllm_blocks_dict_python(blocks_by_layer: Dict[int, int]) -> str:
    lines = ["vllm_num_blocks_dict: Dict[int, int] = {"]
    keys = sorted(blocks_by_layer.keys())
    row: List[str] = []
    for k in keys:
        row.append(f"{k}: {blocks_by_layer[k]}")
        if len(row) >= 6:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("}")
    return "\n".join(lines)


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="vLLM 真机探测 max num_hidden_layers（及可读时的 num_gpu_blocks）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, required=True, help="HF 模型目录（config.json + tokenizer）")
    p.add_argument("--dtype", default="float16", help="如 float16 / bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--load-format",
        default="dummy",
        help="dummy 与 profile_vllm.py 一致；真实权重用 auto（需显存足够）",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        metavar="N",
        help="传给 vLLM 的 max_model_len；默认从 config 的 model_max_length / max_position_embeddings 读取。"
        "若显式传入的值大于推导上限，会钳制到推导值，除非设置环境变量 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1。",
    )
    p.add_argument(
        "--cap-layers",
        type=int,
        default=60,
        help="探测上界（不超过 config 里 num_hidden_layers）",
    )
    p.add_argument(
        "--linear",
        action="store_true",
        help="线性从 2..cap 试（更慢但更直观）；默认二分",
    )
    p.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="不传 enforce_eager=True（默认与 profile 一致为 True，初始化更稳）",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="将结果写入 JSON（含探测日志）",
    )
    p.add_argument(
        "--sweep-blocks-min",
        type=int,
        default=None,
        metavar="N",
        help="与 --sweep-blocks-max 一起使用：对 num_hidden_layers 从 N 到 M 线性逐个起引擎，"
        "记录 num_gpu_blocks（用于填 vllm_num_blocks_dict）；此模式下不做「最大层数」二分。",
    )
    p.add_argument(
        "--sweep-blocks-max",
        type=int,
        default=None,
        metavar="M",
        help="见 --sweep-blocks-min",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    model_dir = args.model_dir.resolve()
    if not (model_dir / "config.json").is_file():
        logging.error("缺少 %s", model_dir / "config.json")
        return 1

    if args.tensor_parallel_size < 1:
        logging.error("tensor_parallel_size 必须 >= 1")
        return 1
    try:
        import torch

        if torch.cuda.is_available() and args.tensor_parallel_size > torch.cuda.device_count():
            logging.error(
                "tensor_parallel_size=%d 超过可见 GPU 数 %d",
                args.tensor_parallel_size,
                torch.cuda.device_count(),
            )
            return 1
    except Exception:
        pass

    derived_ctx = _derived_max_model_len(model_dir)
    allow_long = os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN") == "1"
    if args.max_model_len is None:
        max_model_len = derived_ctx
        logging.info("max_model_len 使用 config 推导值: %d", max_model_len)
    else:
        max_model_len = args.max_model_len
        if max_model_len > derived_ctx and not allow_long:
            logging.warning(
                "max_model_len=%d 大于 config 推导上限 %d，已钳制为 %d。"
                "若确需更长上下文，请设置 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1（参见 vLLM 文档风险提示）。",
                max_model_len,
                derived_ctx,
                derived_ctx,
            )
            max_model_len = derived_ctx

    args_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    sweep_min, sweep_max = args.sweep_blocks_min, args.sweep_blocks_max
    if sweep_min is not None or sweep_max is not None:
        if sweep_min is None or sweep_max is None:
            logging.error("线性扫 num_gpu_blocks 需同时指定 --sweep-blocks-min 与 --sweep-blocks-max")
            return 1
        if sweep_min < 1 or sweep_max < sweep_min:
            logging.error("无效的 sweep 范围: %s..%s", sweep_min, sweep_max)
            return 1

        t0 = time.perf_counter()
        log = sweep_layer_gpu_blocks(
            model_dir=model_dir,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            load_format=args.load_format,
            max_model_len=max_model_len,
            enforce_eager=not args.no_enforce_eager,
            layer_min=sweep_min,
            layer_max=sweep_max,
        )
        elapsed = time.perf_counter() - t0

        blocks_by_layer: Dict[int, int] = {}
        for rec in log:
            if rec.get("ok") and rec.get("num_gpu_blocks") is not None:
                blocks_by_layer[int(rec["num_hidden_layers"])] = int(rec["num_gpu_blocks"])

        payload = {
            "mode": "sweep_gpu_blocks",
            "layer_min": sweep_min,
            "layer_max": sweep_max,
            "gpu": _collect_gpu_info(),
            "derived_max_model_len_from_config": derived_ctx,
            "max_model_len_used": max_model_len,
            "args": args_dump,
            "model_dir": str(model_dir),
            "probe_elapsed_sec": round(elapsed, 3),
            "num_gpu_blocks_by_layer": {str(k): v for k, v in sorted(blocks_by_layer.items())},
            "results": log,
            "vllm_num_blocks_dict_python": _format_vllm_blocks_dict_python(blocks_by_layer),
        }

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        logging.info(
            "sweep 完成: 成功读到 num_gpu_blocks 的层数 = %d / %d",
            len(blocks_by_layer),
            sweep_max - sweep_min + 1,
        )
        print("\n# --- 可复制到 llama1_30b_rtx5090.py ---\n", file=sys.stderr)
        print(_format_vllm_blocks_dict_python(blocks_by_layer), file=sys.stderr)

        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logging.info("已写入 %s", args.json_out)

        return 0 if len(blocks_by_layer) == (sweep_max - sweep_min + 1) else 1

    t0 = time.perf_counter()
    max_ok, log = probe_max_layers(
        model_dir=model_dir,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        load_format=args.load_format,
        max_model_len=max_model_len,
        enforce_eager=not args.no_enforce_eager,
        cap_layers=args.cap_layers,
        linear=args.linear,
    )
    elapsed = time.perf_counter() - t0

    kv_at_max: Dict[str, Any] = {}
    for rec in reversed(log):
        if rec.get("ok") and rec.get("num_hidden_layers") == max_ok:
            kv_at_max = dict(rec.get("kv") or {})
            break

    payload = {
        "max_num_hidden_layers_ok": max_ok,
        "kv_info_at_max": kv_at_max,
        "gpu": _collect_gpu_info(),
        "derived_max_model_len_from_config": derived_ctx,
        "max_model_len_used": max_model_len,
        "args": args_dump,
        "model_dir": str(model_dir),
        "probe_elapsed_sec": round(elapsed, 3),
        "attempts": log,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    logging.info(
        "结论: 当前设置下最大可初始化 num_hidden_layers = %d（见上 JSON 中 kv_info_at_max）",
        max_ok,
    )
    logging.info(
        "Helix 里可把 llama1_30b_rtx5090.py 的 max_num_layers 设为 <= 该值；"
        "vllm_num_blocks_dict 可按 num_gpu_blocks 填一条或按层数扫表。"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("已写入 %s", args.json_out)

    return 0 if max_ok > 0 else 1


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    raise SystemExit(main())
