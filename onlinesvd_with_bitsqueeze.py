#!/usr/bin/env python3
"""
OnlineSVD + BitSqz PPL benchmark.

This script mirrors the OnlineSVD flow used in
`onlinesvd_with_outlier_separation.py`, but additionally quantizes
U/S/Vh with BitSqz before reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from major_entry.compressor import Compressor, Payload
from major_entry.eval_ppl import run_ppl_eval


def _import_bitsqueeze():
    try:
        return importlib.import_module("bitsqueeze")
    except ImportError as err:
        if "libc10.so" not in str(err):
            raise
        torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{torch_lib_dir}:{prev}" if prev else str(torch_lib_dir)
        )
        return importlib.import_module("bitsqueeze")


bitsqueeze = _import_bitsqueeze()


FORMAT_TO_METHOD = {
    "fp32": None,  # baseline: no BitSqz
    "fp16": "FP16",
    "bf16": "BF16",
    "q8_0": "Q8_0",
    "mxfp8": "MXFP8",
    "fp8": "FP8",
    "q4_0": "Q4_0",
    "nf4": "NF4",
    "mxfp4": "MXFP4",
    "nf4_dq": "NF4_DQ",
    "q2_k": "Q2_K",
}

REQUESTED_FORMATS = list(FORMAT_TO_METHOD.keys())
REQUESTED_MODES = ["trunc_slice", "trunc_approx"]


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


class OnlineSVDBitSqueezeCompressor(Compressor):
    name = "OnlineSVDBitSqueeze"

    def __init__(
        self,
        rank: int,
        mode: str,
        fmt_uv: str,
        fmt_s: str | None = None,
        niter: int = 2,
        svd_device: str = "auto",
    ):
        if mode not in REQUESTED_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if fmt_uv not in FORMAT_TO_METHOD:
            raise ValueError(f"Unknown U/Vh format: {fmt_uv}")
        if fmt_s is None:
            fmt_s = fmt_uv
        if fmt_s not in FORMAT_TO_METHOD:
            raise ValueError(f"Unknown S format: {fmt_s}")
        self.rank = rank
        self.mode = mode
        self.fmt_uv = fmt_uv
        self.fmt_s = fmt_s
        self.niter = niter
        self.svd_device = svd_device
        self.name = f"OnlineSVDBitSqueeze_{mode}_{rank}_{fmt_uv}_s-{fmt_s}"

    def _pick_svd_device(self, x: torch.Tensor) -> torch.device:
        if self.svd_device == "auto":
            # Default to the input tensor device to avoid extra cross-device movement.
            return x.device
        if self.svd_device == "cpu":
            return torch.device("cpu")
        if self.svd_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("svd_device=cuda requested but CUDA is unavailable.")
            return torch.device("cuda")
        raise ValueError(f"Unknown svd_device: {self.svd_device}")

    def _quantize_component(self, x: torch.Tensor, fmt: str) -> tuple[Any, int]:
        x_cpu = x.detach().to(dtype=torch.float32, device="cpu").contiguous()
        method = FORMAT_TO_METHOD[fmt]
        if method is None:
            return x_cpu, x_cpu.numel() * x_cpu.element_size()
        buffer = bitsqueeze.BitSqueezeBuffer.compress(x_cpu, method)
        return buffer, int(buffer.size)

    def _dequantize_component(self, data: Any, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(data):
            out = data
        else:
            out = data.decompress()
        return out.to(device=device, dtype=torch.float32)

    def _svd_decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_dim = x.shape[-2]
        feature_dim = x.shape[-1]
        max_rank = min(seq_dim, feature_dim)
        k = min(self.rank, max_rank)

        if self.mode == "trunc_slice":
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
            return U[..., :k], S[..., :k], Vh[..., :k, :]

        if self.mode == "trunc_approx":
            # svd_lowrank is applied per matrix for robust batched behavior.
            flat_x = x.reshape(-1, seq_dim, feature_dim)
            U_list = []
            S_list = []
            Vh_list = []
            for i in range(flat_x.shape[0]):
                U_i, S_i, V_i = torch.svd_lowrank(flat_x[i], q=k, niter=self.niter)
                U_list.append(U_i)
                S_list.append(S_i)
                Vh_list.append(V_i.mT)

            lead_shape = x.shape[:-2]
            U = torch.stack(U_list, dim=0).reshape(*lead_shape, seq_dim, k)
            S = torch.stack(S_list, dim=0).reshape(*lead_shape, k)
            Vh = torch.stack(Vh_list, dim=0).reshape(*lead_shape, k, feature_dim)
            return U, S, Vh

        raise ValueError(f"Unknown mode: {self.mode}")

    def compress(self, x: torch.Tensor) -> Payload:
        if x.dim() < 2:
            raise ValueError(f"Expected x.dim() >= 2, got shape={tuple(x.shape)}")

        svd_device = self._pick_svd_device(x)
        x_work = x.to(device=svd_device, dtype=torch.float32)
        U, S, Vh = self._svd_decompose(x_work)

        qU, u_bytes = self._quantize_component(U, fmt=self.fmt_uv)
        qS, s_bytes = self._quantize_component(S, fmt=self.fmt_s)
        qVh, vh_bytes = self._quantize_component(Vh, fmt=self.fmt_uv)

        return Payload(
            data=(qU, qS, qVh),
            meta={
                "orig_dtype": str(x.dtype),
                "shape": tuple(x.shape),
                "seq_dim": x.shape[-2],
                "feature_dim": x.shape[-1],
                "effective_rank": S.shape[-1],
                "mode": self.mode,
                "format_uv": self.fmt_uv,
                "format_s": self.fmt_s,
            },
            nbytes=u_bytes + s_bytes + vh_bytes,
        )

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        qU, qS, qVh = p.data
        U = self._dequantize_component(qU, device=device)
        S = self._dequantize_component(qS, device=device)
        Vh = self._dequantize_component(qVh, device=device)
        # Reconstruct via element-wise singular value scaling without materializing diag(S).
        recon = (U * S.unsqueeze(-2)) @ Vh
        return recon.to(device=device, dtype=dtype)


def _fmt(v: Any, digits: int = 6) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fields = [
        "mode",
        "rank",
        "uv_format",
        "s_format",
        "avg_ppl",
        "bytes_per_token",
        "total_bytes",
        "total_tokens",
        "total_tx",
        "eval_time_s",
        "run_json",
        "status",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    rows: list[dict[str, Any]],
    md_path: Path,
    csv_path: Path,
    args: argparse.Namespace,
    combo_count: int,
) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    err_rows = [r for r in rows if r["status"] != "ok"]

    lines: list[str] = []
    lines.append("# OnlineSVD + BitSqz PPL Benchmark")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append(f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Model name: {args.model_name}")
    lines.append(f"- Model dir: {args.model_dir}")
    lines.append(f"- dtype: {args.dtype}")
    lines.append(f"- load_in_8bit: {args.load_in_8bit}")
    lines.append(f"- load_in_4bit: {args.load_in_4bit}")
    lines.append(f"- max_length: {args.max_length}")
    lines.append(f"- stride: {args.stride}")
    lines.append(f"- first_k_tokens: {args.first_k_tokens}")
    lines.append(f"- batch_windows: {args.batch_windows}")
    lines.append(f"- svd_lowrank niter (`trunc_approx`): {args.niter}")
    lines.append(f"- svd_device: {args.svd_device}")
    lines.append(f"- Modes: {', '.join(args.modes)}")
    lines.append(f"- Ranks: {', '.join(str(r) for r in args.ranks)}")
    lines.append(f"- U/Vh formats: {', '.join(args.uv_formats)}")
    lines.append(f"- S formats: {', '.join(args.s_formats)}")
    lines.append(f"- Combination count: {combo_count}")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append("")

    lines.append("## Results")
    lines.append("| mode | rank | uv_format | s_format | status | avg_ppl | B/tok | total_bytes | eval_time_s |")
    lines.append("|---|---:|---|---|---|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (r["mode"], r["rank"], r["uv_format"], r["s_format"])):
        lines.append(
            f"| {_fmt(row['mode'])} | {_fmt(row['rank'])} | {_fmt(row['uv_format'])} | {_fmt(row['s_format'])} | {_fmt(row['status'])} "
            f"| {_fmt(row['avg_ppl'])} | {_fmt(row['bytes_per_token'])} | {_fmt(row['total_bytes'])} | {_fmt(row['eval_time_s'])} |"
        )

    if err_rows:
        lines.append("")
        lines.append("## Errors")
        for row in err_rows:
            lines.append(
                f"- mode={row['mode']}, rank={row['rank']}, uv_format={row['uv_format']}, s_format={row['s_format']}: {row['error']}"
            )

    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Successful combos: {len(ok_rows)}/{len(rows)}")
    if ok_rows:
        best = min(ok_rows, key=lambda r: float(r["avg_ppl"]))
        lines.append(
            f"- Best (lowest) avg_ppl: mode={best['mode']}, rank={best['rank']}, uv_format={best['uv_format']}, s_format={best['s_format']}, avg_ppl={_fmt(best['avg_ppl'])}"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OnlineSVD + BitSqz PPL benchmark runner")

    # python onlinesvd_with_bitsqueeze.py --uv-formats fp32 fp16 bf16 q8_0 mxfp8 fp8 q4_0 nf4 mxfp4 nf4_dq q2_k --s-formats fp32 fp16 fp8 --ranks 512 1024 2>&1 | tee process.log

    parser.add_argument("--ranks", type=int, nargs="+", default=[16])
    # Backward-compatible shorthand: if used alone, this sets both U/Vh and S formats.
    parser.add_argument("--modes", nargs="+", default=REQUESTED_MODES)
    parser.add_argument("--formats", nargs="+", default=REQUESTED_FORMATS)
    parser.add_argument("--uv-formats", nargs="+", default=None)
    parser.add_argument("--s-formats", nargs="+", default=None)
    parser.add_argument("--niter", type=int, default=2)
    parser.add_argument("--svd-device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--model-dir", type=str, default="/mnt/ssd/liaw/Qwen/Qwen3-8B")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")

    parser.add_argument("--load-in-8bit", dest="load_in_8bit", action="store_true")
    parser.add_argument("--no-load-in-8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=True)

    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=False)

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--first-k-tokens", type=int, default=0)
    parser.add_argument("--batch-windows", type=int, default=2)

    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="decentralized-infer")
    parser.add_argument("--wandb-log-every", type=int, default=10)

    parser.add_argument("--output-dir", type=Path, default=Path("results/onlinesvd_bitsqueeze"))
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--md-path", type=Path, default=None)
    parser.add_argument("--ppl-result-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_formats = [f.lower() for f in args.formats]
    args.uv_formats = [f.lower() for f in (args.uv_formats or base_formats)]
    args.s_formats = [f.lower() for f in args.s_formats] if args.s_formats is not None else None

    for mode in args.modes:
        if mode not in REQUESTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}. Expected one of {REQUESTED_MODES}.")
    for fmt in args.uv_formats:
        if fmt not in FORMAT_TO_METHOD:
            raise ValueError(
                f"Unsupported format: {fmt}. Expected one of {list(FORMAT_TO_METHOD.keys())}."
            )
    if args.s_formats is None:
        format_pairs = [(fmt, fmt) for fmt in args.uv_formats]
        args.s_formats = list(args.uv_formats)
    else:
        for fmt in args.s_formats:
            if fmt not in FORMAT_TO_METHOD:
                raise ValueError(
                    f"Unsupported format: {fmt}. Expected one of {list(FORMAT_TO_METHOD.keys())}."
                )
        format_pairs = [(uv_fmt, s_fmt) for uv_fmt in args.uv_formats for s_fmt in args.s_formats]

    combo_count = len(args.ranks) * len(args.modes) * len(format_pairs)
    print(
        f"Running OnlineSVD+BitSqz PPL benchmark: {combo_count} combos "
        f"({len(args.ranks)} ranks x {len(args.modes)} modes x {len(format_pairs)} format_pairs)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.csv_path is None:
        args.csv_path = args.output_dir / f"onlinesvd_bitsqueeze_ppl_{stamp}.csv"
    if args.md_path is None:
        args.md_path = args.output_dir / f"onlinesvd_bitsqueeze_ppl_{stamp}.md"
    if args.ppl_result_root is None:
        args.ppl_result_root = args.output_dir / f"ppl_runs_{stamp}"
    args.ppl_result_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for rank in args.ranks:
        for mode in args.modes:
            for uv_fmt, s_fmt in format_pairs:
                    idx = len(rows) + 1
                    print(f"[{idx}/{combo_count}] rank={rank} mode={mode} uv_format={uv_fmt} s_format={s_fmt}")

                    compressor = OnlineSVDBitSqueezeCompressor(
                        rank=rank,
                        mode=mode,
                        fmt_uv=uv_fmt,
                        fmt_s=s_fmt,
                        niter=args.niter,
                        svd_device=args.svd_device,
                    )
                    combo_dir = args.ppl_result_root / f"{mode}_{rank}_uv-{uv_fmt}_s-{s_fmt}"
                    combo_dir.mkdir(parents=True, exist_ok=True)

                    t0 = time.time()
                    try:
                        ppl, totals, out_json = run_ppl_eval(
                            model_name=args.model_name,
                            model_dir=args.model_dir,
                            dtype=args.dtype,
                            load_in_8bit=args.load_in_8bit,
                            load_in_4bit=args.load_in_4bit,
                            compressor=compressor,
                            max_length=args.max_length,
                            stride=args.stride,
                            first_k_tokens=args.first_k_tokens,
                            batch_windows=args.batch_windows,
                            wandb=args.wandb,
                            wandb_project=args.wandb_project,
                            wandb_log_every=args.wandb_log_every,
                            result_dir=str(combo_dir),
                        )

                        row = {
                            "mode": mode,
                            "rank": rank,
                            "uv_format": uv_fmt,
                            "s_format": s_fmt,
                            "avg_ppl": _to_float(ppl),
                            "bytes_per_token": _to_float(totals.get("bytes_per_token")),
                            "total_bytes": _to_int(totals.get("total_bytes")),
                            "total_tokens": _to_int(totals.get("total_tokens")),
                            "total_tx": _to_int(totals.get("total_tx")),
                            "eval_time_s": time.time() - t0,
                            "run_json": out_json,
                            "status": "ok",
                            "error": "",
                        }
                    except Exception as err:
                        row = {
                            "mode": mode,
                            "rank": rank,
                            "uv_format": uv_fmt,
                            "s_format": s_fmt,
                            "avg_ppl": float("nan"),
                            "bytes_per_token": float("nan"),
                            "total_bytes": 0,
                            "total_tokens": 0,
                            "total_tx": 0,
                            "eval_time_s": time.time() - t0,
                            "run_json": "",
                            "status": "error",
                            "error": str(err),
                        }

                    rows.append(row)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    write_csv(rows, args.csv_path)
    write_markdown_report(rows, args.md_path, args.csv_path, args, combo_count)

    ok_count = sum(1 for r in rows if r["status"] == "ok")
    print(f"Done. Successful combos: {ok_count}/{len(rows)}")
    print(f"CSV written to: {args.csv_path}")
    print(f"Markdown report written to: {args.md_path}")


if __name__ == "__main__":
    main()
