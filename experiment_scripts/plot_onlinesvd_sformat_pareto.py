#!/usr/bin/env python3
"""Plot Pareto frontier comparison across s_format values (B/tok vs avg_ppl)."""

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

SERIES_STYLES = {
    "fp32": {
        "scatter": "#9ec9ff",  # light blue
        "line": "#1f77b4",     # blue
    },
    "fp16": {
        "scatter": "#ffb3b3",  # light red
        "line": "#d62728",     # red
    },
    "fp8": {
        "scatter": "#b7e3b7",  # light green
        "line": "#2ca02c",     # green
    },
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
        default=Path("results/onlinesvd_bitsqueeze/pareto_frontier_s-format_fp32-fp16-fp8_mode-trunc_slice_rank512_btok-vs-ppl.png"),
        help="Output figure path (.png/.pdf/etc).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="trunc_slice",
        help="Mode to filter. Alias 'trunc_alice' maps to 'trunc_slice'.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=512,
        help="Rank to analyze.",
    )
    parser.add_argument(
        "--s-formats",
        type=str,
        nargs="+",
        default=["fp32", "fp16", "fp8"],
        help="s_format values to compare.",
    )
    parser.add_argument(
        "--avg-ppl-range",
        type=str,
        default=None,
        help="Optional analysis range for avg_ppl. Points outside are removed. Example: '8.2~10'.",
    )
    return parser.parse_args()


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


def to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def pareto_frontier(points: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    # Minimize both x (b_tok) and y (avg_ppl).
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

    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    resolved_mode = resolve_mode(args.mode, {row["mode"] for row in rows})
    requested_s_formats = list(dict.fromkeys(args.s_formats))
    available_s_formats = {row["s_format"] for row in rows}
    missing_s_formats = [value for value in requested_s_formats if value not in available_s_formats]
    if missing_s_formats:
        raise ValueError(
            f"Requested s_formats not found in CSV: {missing_s_formats}. "
            f"Available s_formats: {sorted(available_s_formats)}"
        )

    grouped: dict[str, list[dict[str, float | str]]] = {}
    dropped_by_range: dict[str, int] = {}
    for s_format in requested_s_formats:
        selected: list[dict[str, float | str]] = []
        for row in rows:
            if row["status"] != "ok":
                continue
            if row["mode"] != resolved_mode:
                continue
            if int(row["rank"]) != args.rank:
                continue
            if row["s_format"] != s_format:
                continue
            selected.append(
                {
                    "uv_format": row["uv_format"],
                    "b_tok": to_float(row["bytes_per_token"]),
                    "avg_ppl": to_float(row["avg_ppl"]),
                }
            )

        selected = [
            p for p in selected
            if (
                math.isfinite(float(p["b_tok"]))
                and math.isfinite(float(p["avg_ppl"]))
            )
        ]
        initial_count = len(selected)
        if y_range is not None:
            low, high = y_range
            selected = [p for p in selected if low <= float(p["avg_ppl"]) <= high]
        dropped_by_range[s_format] = initial_count - len(selected)
        if not selected:
            raise ValueError(
                f"No rows after filtering for mode={resolved_mode}, rank={args.rank}, "
                f"s_format={s_format}, avg_ppl_range={args.avg_ppl_range}."
            )
        grouped[s_format] = selected

    frontiers = {s_format: pareto_frontier(points) for s_format, points in grouped.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for s_format in requested_s_formats:
        points = grouped[s_format]
        frontier = frontiers[s_format]
        style = SERIES_STYLES.get(s_format, {"scatter": "#cccccc", "line": "#333333"})

        x_all = [float(p["b_tok"]) for p in points]
        y_all = [float(p["avg_ppl"]) for p in points]
        x_pf = [float(p["b_tok"]) for p in frontier]
        y_pf = [float(p["avg_ppl"]) for p in frontier]

        ax.scatter(
            x_all,
            y_all,
            s=72,
            color=style["scatter"],
            edgecolors="black",
            linewidths=0.5,
            label=f"{s_format} (all uv_format)",
            zorder=2,
        )
        ax.plot(
            x_pf,
            y_pf,
            color=style["line"],
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            label=f"{s_format} Pareto frontier",
            zorder=3,
        )

    title_mode = f"{args.mode} -> {resolved_mode}" if args.mode != resolved_mode else resolved_mode
    ax.set_title(f"Pareto Frontier: s_format Comparison (mode={title_mode}, rank={args.rank})")
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
    for s_format in requested_s_formats:
        print(
            f"s_format {s_format}: kept points={len(grouped[s_format])}, "
            f"dropped_by_range={dropped_by_range[s_format]}, pareto points={len(frontiers[s_format])}"
        )


if __name__ == "__main__":
    main()
