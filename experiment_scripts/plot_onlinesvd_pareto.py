#!/usr/bin/env python3
"""Plot Pareto frontier comparison for OnlineSVD + BitSqz PPL results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


MODE_ALIASES = {
    "trunc_alice": "trunc_slice",
}


SCATTER_COLORS = {
    512: "#9ec9ff",   # light blue bullets
    1024: "#ffb3b3",  # light red bullets
}

LINE_COLORS = {
    512: "#1f77b4",   # blue line for rank 512 frontier
    1024: "#d62728",  # red line for rank 1024 frontier
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/onlinesvd_bitsqueeze/onlinesvd_bitsqueeze_ppl_20260226_221024.csv"),
        help="Path to results CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/onlinesvd_bitsqueeze/pareto_frontier_trunc_slice_s-fp32_rank512_vs1024.png"),
        help="Output figure path (.png/.pdf/etc).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="trunc_alice",
        help="Mode to filter. Alias 'trunc_alice' maps to 'trunc_slice'.",
    )
    parser.add_argument(
        "--s-format",
        type=str,
        default="fp32",
        help="s_format to filter.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[512, 1024],
        help="Ranks to include.",
    )
    parser.add_argument(
        "--avg-ppl-range",
        type=str,
        default=None,
        help="Optional y-axis range for avg_ppl, e.g. '8.2~10' or '8.2,10'.",
    )
    return parser.parse_args()


def to_float(value: str) -> float:
    try:
        out = float(value)
    except ValueError:
        return math.nan
    return out


def resolve_mode(requested_mode: str, available_modes: Iterable[str]) -> str:
    if requested_mode in available_modes:
        return requested_mode
    alias = MODE_ALIASES.get(requested_mode)
    if alias and alias in available_modes:
        return alias
    raise ValueError(
        f"Mode '{requested_mode}' not found in CSV. "
        f"Available modes: {', '.join(sorted(set(available_modes)))}"
    )


def parse_avg_ppl_range(raw: str | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    text = raw.strip()
    for sep in ("~", ",", ":"):
        if sep in text:
            left, right = text.split(sep, 1)
            low = float(left.strip())
            high = float(right.strip())
            if low >= high:
                raise ValueError(f"Invalid --avg-ppl-range '{raw}': min must be < max.")
            return (low, high)
    raise ValueError(
        f"Invalid --avg-ppl-range '{raw}'. Use formats like '8.2~10' or '8.2,10'."
    )


def pareto_frontier(points: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    ordered = sorted(points, key=lambda p: (p["b_tok"], p["avg_ppl"], p["uv_format"]))
    frontier: list[dict[str, float | str]] = []
    best_y = math.inf
    eps = 1e-12
    for point in ordered:
        y = float(point["avg_ppl"])
        if y < best_y - eps:
            frontier.append(point)
            best_y = y
    return frontier


def main() -> None:
    args = parse_args()
    y_range = parse_avg_ppl_range(args.avg_ppl_range)

    rows: list[dict[str, str]] = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    available_modes = {row["mode"] for row in rows}
    resolved_mode = resolve_mode(args.mode, available_modes)

    grouped: dict[int, list[dict[str, float | str]]] = {}
    dropped_by_range: dict[int, int] = {}
    for rank in args.ranks:
        selected = []
        for row in rows:
            if row["status"] != "ok":
                continue
            if row["mode"] != resolved_mode:
                continue
            if row["s_format"] != args.s_format:
                continue
            if int(row["rank"]) != rank:
                continue
            selected.append(
                {
                    "uv_format": row["uv_format"],
                    "b_tok": to_float(row["bytes_per_token"]),
                    "avg_ppl": to_float(row["avg_ppl"]),
                }
            )

        selected = [p for p in selected if math.isfinite(float(p["b_tok"])) and math.isfinite(float(p["avg_ppl"]))]
        initial_count = len(selected)
        if y_range is not None:
            low, high = y_range
            selected = [p for p in selected if low <= float(p["avg_ppl"]) <= high]
        dropped_by_range[rank] = initial_count - len(selected)
        if not selected:
            raise ValueError(
                f"No rows after filtering for rank={rank}, mode={resolved_mode}, "
                f"s_format={args.s_format}, avg_ppl_range={args.avg_ppl_range}."
            )
        grouped[rank] = selected

    frontiers = {rank: pareto_frontier(points) for rank, points in grouped.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for rank in args.ranks:
        points = grouped[rank]
        frontier = frontiers[rank]

        x_all = [float(p["b_tok"]) for p in points]
        y_all = [float(p["avg_ppl"]) for p in points]
        x_pf = [float(p["b_tok"]) for p in frontier]
        y_pf = [float(p["avg_ppl"]) for p in frontier]

        ax.scatter(
            x_all,
            y_all,
            s=72,
            color=SCATTER_COLORS.get(rank, "#bbbbbb"),
            edgecolors="black",
            linewidths=0.5,
            label=f"rank {rank} (all uv_format)",
            zorder=2,
        )
        ax.plot(
            x_pf,
            y_pf,
            color=LINE_COLORS.get(rank, "#333333"),
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            label=f"rank {rank} Pareto frontier",
            zorder=3,
        )
        text_color = LINE_COLORS.get(rank, "#333333")
        for idx, point in enumerate(frontier):
            x = float(point["b_tok"])
            y = float(point["avg_ppl"])
            uv_label = str(point["uv_format"])
            x_off = 6 if rank == 512 else -6
            y_off = 8 if idx % 2 == 0 else -10
            ax.annotate(
                uv_label,
                xy=(x, y),
                xytext=(x_off, y_off),
                textcoords="offset points",
                fontsize=8,
                color=text_color,
                ha="left" if x_off > 0 else "right",
                va="bottom" if y_off > 0 else "top",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
                zorder=4,
            )

    title_mode = f"{args.mode} -> {resolved_mode}" if args.mode != resolved_mode else resolved_mode
    ax.set_title(f"Pareto Frontier Comparison (mode={title_mode}, s_format={args.s_format})")
    ax.set_xlabel("B/tok (bytes per token, lower is better)")
    ax.set_ylabel("avg_ppl (lower is better)")
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output)

    print(f"Saved figure: {args.output}")
    if y_range is not None:
        print(f"Applied avg_ppl analysis filter: {y_range[0]}~{y_range[1]}")
    for rank in args.ranks:
        print(
            f"rank {rank}: kept points={len(grouped[rank])}, "
            f"dropped_by_range={dropped_by_range[rank]}, pareto points={len(frontiers[rank])}"
        )


if __name__ == "__main__":
    main()
