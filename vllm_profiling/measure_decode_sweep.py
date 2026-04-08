#!/usr/bin/env python3
"""
按 num_hidden_layers 扫描 Helix 用的 decode 并发上界（decode_max_tokens_dict）。

- 每层单独子进程调用 measure_inference_caps.py（decode-only），减少 GPU 残留。
- decode 侧每条序列 prefill 固定 1 token，与 profile_vllm.py --prompt-len-for-decode 默认一致。

用法（在 vllm_profiling 目录）：
  python measure_decode_sweep.py --model-dir .../llama1_30b --layer-min 1 --layer-max 24

合并结果只写入 --merged-json（每层子进程用临时文件，跑完即删）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _format_decode_dict_python(d: Dict[int, int]) -> str:
    lines = ["decode_max_tokens_dict: Dict[int, int] = {"]
    keys = sorted(d.keys())
    row: List[str] = []
    for k in keys:
        row.append(f"{k}: {d[k]}")
        if len(row) >= 6:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("}")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按层扫描 decode 并发上界（供 decode_max_tokens_dict），prefill=1 对齐 profile_vllm。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--layer-min", type=int, default=1)
    p.add_argument("--layer-max", type=int, default=24)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--load-format", default="dummy")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--decode-output-tokens", type=int, default=256)
    p.add_argument("--search-hi", type=int, default=128)
    p.add_argument("--abs-max-concurrency", type=int, default=1024)
    p.add_argument(
        "--merged-json",
        type=Path,
        default=Path("./outputs/decode_sweep_merged.json"),
        help="唯一输出：合并结果（含每层摘要与 decode_max_tokens_dict 文本）",
    )
    p.add_argument(
        "--sleep-sec",
        type=float,
        default=2.0,
        help="两层之间休眠秒数，便于子进程释放显存",
    )
    p.add_argument("--no-enforce-eager", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    if args.layer_min < 1 or args.layer_max < args.layer_min:
        logging.error("无效的 layer 范围: %d..%d", args.layer_min, args.layer_max)
        return 1

    script_dir = Path(__file__).resolve().parent
    mic = script_dir / "measure_inference_caps.py"
    if not mic.is_file():
        logging.error("找不到 %s", mic)
        return 1

    model_dir = args.model_dir.resolve()
    args.merged_json.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    decode_by_layer: Dict[int, int] = {}

    import subprocess

    for n in range(args.layer_min, args.layer_max + 1):
        fd, tmp_name = tempfile.mkstemp(suffix=".json", prefix="helix_decode_cap_")
        os.close(fd)
        out_json = Path(tmp_name)
        cmd: List[str] = [
            sys.executable,
            str(mic),
            "--model-dir",
            str(model_dir),
            "--num-hidden-layers",
            str(n),
            "--dtype",
            args.dtype,
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--load-format",
            args.load_format,
            "--skip-prefill",
            "--decode-prefill-tokens",
            "1",
            "--decode-output-tokens",
            str(args.decode_output_tokens),
            "--search-hi",
            str(args.search_hi),
            "--abs-max-concurrency",
            str(args.abs_max_concurrency),
            "--json-out",
            str(out_json),
        ]
        if args.max_model_len is not None:
            cmd.extend(["--max-model-len", str(args.max_model_len)])
        if args.no_enforce_eager:
            cmd.append("--no-enforce-eager")
        if args.verbose:
            cmd.append("-v")

        logging.info(">>> layer n=%d", n)
        try:
            r = subprocess.run(cmd, cwd=str(script_dir))
            rec: Dict[str, Any] = {
                "num_hidden_layers": n,
                "exit_code": r.returncode,
            }
            if out_json.is_file():
                try:
                    data = json.loads(out_json.read_text(encoding="utf-8"))
                    dec = data.get("decode")
                    if isinstance(dec, dict):
                        mcs = dec.get("max_concurrent_sequences")
                        rec["max_concurrent_sequences"] = mcs
                        rec["probe_meta"] = dec.get("probe_meta")
                        if mcs is not None and isinstance(mcs, int) and mcs >= 1:
                            decode_by_layer[n] = mcs
                except Exception as e:
                    rec["parse_error"] = str(e)
            rows.append(rec)
        finally:
            if out_json.is_file():
                try:
                    out_json.unlink()
                except OSError:
                    pass

        if n < args.layer_max and args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    payload = {
        "note": (
            "decode 压测：每条序列 prefill=1 token（对齐 profile_vllm --prompt-len-for-decode）；"
            "max_concurrent_sequences 可写入 decode_max_tokens_dict[n]（建议略留余量）。"
        ),
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "decode_prefill_tokens": 1,
        "decode_output_tokens": args.decode_output_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_format": args.load_format,
        "model_dir": str(model_dir),
        "decode_max_concurrent_by_layer": {str(k): v for k, v in sorted(decode_by_layer.items())},
        "layers": rows,
        "decode_max_tokens_dict_python": _format_decode_dict_python(decode_by_layer)
        if decode_by_layer
        else "",
    }

    args.merged_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("已写入合并结果 %s", args.merged_json)
    if decode_by_layer:
        print(_format_decode_dict_python(decode_by_layer))

    expected = args.layer_max - args.layer_min + 1
    ok_count = len(decode_by_layer)
    if ok_count < expected:
        logging.warning("成功层数 %d / %d，请检查 %s 内 layers[].exit_code / parse_error", ok_count, expected, args.merged_json)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
