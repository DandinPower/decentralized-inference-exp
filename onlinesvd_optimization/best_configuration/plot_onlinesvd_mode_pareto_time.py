#!/usr/bin/env python3
"""Plot Pareto frontier comparison between experiment modes (B/tok vs ppl)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


SERIES_STYLES = {
    "trunc_approx": {
        "scatter": "#9ec9ff",  # light blue
        "line": "#1f77b4",  # blue
    },
    "trunc_slice": {
        "scatter": "#ffb3b3",  # light red
        "line": "#d62728",  # red
    },
    # Custom experiment families for the 6 requested curves.
    "bitsqz_topk_0_005": {
        "scatter": "#1f77b4",
        "line": "#1f77b4",
    },
    "bitsqz_topk_0_001": {
        "scatter": "#ff7f0e",
        "line": "#ff7f0e",
    },
    "bitsqz": {
        "scatter": "#2ca02c",
        "line": "#2ca02c",
    },
    "bitsqz_only": {
        "scatter": "#d62728",
        "line": "#d62728",
    },
    "bitsqz_only_topk_0_001": {
        "scatter": "#9467bd",
        "line": "#9467bd",
    },
    "error_correction": {
        "scatter": "#8c564b",
        "line": "#8c564b",
    },
}

SERIES_LABELS = {
    "trunc_approx": "trunc approx",
    "trunc_slice": "trunc slice",
    "bitsqz_topk_0_005": "SVD + BitSqz + Topk (0.005)",
    "bitsqz_topk_0_001": "SVD + BitSqz + Topk (0.001)",
    "bitsqz": "SVD + BitSqz",
    "bitsqz_only": "BitSqz",
    "bitsqz_only_topk_0_001": "BitSqz + Topk (0.001)",
    "error_correction": "SVD + Error Correction + BitSqz + Topk (0.001)",
}

FALLBACK_SERIES_COLORS = [
    "#17becf",
    "#e377c2",
    "#7f7f7f",
    "#8c564b",
    "#bcbd22",
]


def series_style(series_name: str, fallback_index: int) -> dict[str, str]:
    if series_name in SERIES_STYLES:
        return SERIES_STYLES[series_name]
    color = FALLBACK_SERIES_COLORS[fallback_index % len(FALLBACK_SERIES_COLORS)]
    return {"scatter": color, "line": color}


def series_label(series_name: str) -> str:
    return SERIES_LABELS.get(series_name, series_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "pareto_frontier_bitsqz_6_experiments.csv"
        ),
        help="Path to results CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "pareto_frontier.png"
        ),
        help="Output figure path (.png/.pdf/etc).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=512,
        help="Rank to analyze.",
    )
    parser.add_argument(
        "--s-format",
        type=str,
        default="fp32",
        help="s_format to filter.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=[
            "bitsqz_topk_0_005",
            "bitsqz_topk_0_001",
            "bitsqz",
            "bitsqz_only",
            "bitsqz_only_topk_0_001",
            "error_correction",
        ],
        help="Modes to compare.",
    )
    parser.add_argument(
        "--avg-ppl-range",
        type=str,
        default="8.5~9.9",
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


def to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def pareto_frontier(
    points: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
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

    available_modes = {row["mode"] for row in rows}
    requested_modes = list(dict.fromkeys(args.modes))
    missing_modes = [mode for mode in requested_modes if mode not in available_modes]
    if missing_modes:
        raise ValueError(
            f"Requested modes not found in CSV: {missing_modes}. "
            f"Available modes: {sorted(available_modes)}"
        )

    grouped: dict[str, list[dict[str, float | str]]] = {}
    dropped_by_range: dict[str, int] = {}
    for mode in requested_modes:
        selected: list[dict[str, float | str]] = []
        for row in rows:
            if row["status"] != "ok":
                continue
            if row["mode"] != mode:
                continue
            if int(row["rank"]) != args.rank:
                continue
            if row["s_format"] != args.s_format:
                continue
            selected.append(
                {
                    "uv_format": row["uv_format"],
                    "b_tok": to_float(row["bytes_per_token"]),
                    "eval_time_s": to_float(row["eval_time_s"]),
                    "avg_ppl": to_float(row["avg_ppl"]),
                }
            )

        selected = [
            p
            for p in selected
            if (
                math.isfinite(float(p["b_tok"]))
                and math.isfinite(float(p["eval_time_s"]))
                and math.isfinite(float(p["avg_ppl"]))
            )
        ]
        initial_count = len(selected)
        if y_range is not None:
            low, high = y_range
            selected = [p for p in selected if low <= float(p["avg_ppl"]) <= high]
        dropped_by_range[mode] = initial_count - len(selected)
        if not selected:
            raise ValueError(
                f"No rows after filtering for mode={mode}, rank={args.rank}, "
                f"s_format={args.s_format}, avg_ppl_range={args.avg_ppl_range}."
            )
        grouped[mode] = selected

    frontiers = {mode: pareto_frontier(points) for mode, points in grouped.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for mode_idx, mode in enumerate(requested_modes):
        points = grouped[mode]
        frontier = frontiers[mode]
        style = series_style(mode, mode_idx)

        x_all = [float(p["b_tok"]) for p in points]
        y_all = [float(p["avg_ppl"]) for p in points]
        x_pf = [float(p["b_tok"]) for p in frontier]
        y_pf = [float(p["avg_ppl"]) for p in frontier]

        display_name = series_label(mode)

        ax.scatter(
            x_all,
            y_all,
            s=72,
            color=style["scatter"],
            edgecolors="black",
            linewidths=0.5,
            label="_nolegend_",
            zorder=2,
        )
        ax.plot(
            x_pf,
            y_pf,
            color=style["line"],
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            label=display_name,
            zorder=3,
        )

        for frontier_idx, point in enumerate(frontier):
            x = float(point["b_tok"])
            y = float(point["avg_ppl"])
            ppl_label = f"{y:.5f}"
            x_off = 7 if mode_idx % 2 == 0 else -7
            y_off = 8 if frontier_idx % 2 == 0 else -10
            ax.annotate(
                ppl_label,
                xy=(x, y),
                xytext=(x_off, y_off),
                textcoords="offset points",
                fontsize=8,
                color=style["line"],
                ha="left" if x_off > 0 else "right",
                va="bottom" if y_off > 0 else "top",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.75,
                },
                zorder=4,
            )

    ax.set_title(
        f"Pareto frontiers by mode (rank={args.rank}, s_format={args.s_format})"
    )
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
    for mode in requested_modes:
        print(
            f"mode {mode}: kept points={len(grouped[mode])}, "
            f"dropped_by_range={dropped_by_range[mode]}, pareto points={len(frontiers[mode])}"
        )


if __name__ == "__main__":
    main()
