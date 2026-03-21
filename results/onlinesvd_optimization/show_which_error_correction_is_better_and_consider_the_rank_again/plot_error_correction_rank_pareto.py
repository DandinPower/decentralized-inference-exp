#!/usr/bin/env python3
"""Plot one Pareto frontier for error-correction runs, colored by rank."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


FOLDER_PATTERN = re.compile(
    r"^error_correction_(?P<error>[0-9.]+)_nf4_dq_rank_(?P<rank>\d+)$"
)
BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory containing folders named like error_correction_0.01_nf4_dq_rank_256.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "pareto_error_correction_colored_by_rank.png",
        help="Output figure path (.png/.pdf/etc).",
    )
    parser.add_argument(
        "--avg-ppl-range",
        type=str,
        default='8.5~9.9',
        help="Optional y-axis range, e.g. '8.2~10' or '8.2,10'.",
    )
    parser.add_argument(
        "--avg-ppl-max",
        type=float,
        default=None,
        help="Optional y-axis upper bound for display only (no row filtering).",
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


def pareto_frontier(
    points: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    ordered = sorted(
        points,
        key=lambda p: (
            float(p["b_tok"]),
            float(p["avg_ppl"]),
            int(p["rank"]),
            float(p["error_correction"]),
        ),
    )
    frontier: list[dict[str, float | int | str]] = []
    best_y = math.inf
    eps = 1e-12
    for point in ordered:
        y = float(point["avg_ppl"])
        if y < best_y - eps:
            frontier.append(point)
            best_y = y
    return frontier


def parse_folder_meta(folder_name: str) -> tuple[float, int] | None:
    match = FOLDER_PATTERN.match(folder_name)
    if not match:
        return None
    error_correction = float(match.group("error"))
    rank = int(match.group("rank"))
    return (error_correction, rank)


def ppl_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def main() -> None:
    args = parse_args()
    y_range = parse_avg_ppl_range(args.avg_ppl_range)

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    all_points: list[dict[str, float | int | str]] = []

    for run_json in sorted(args.results_dir.glob("*/run_*.json")):
        folder = run_json.parent.name
        meta = parse_folder_meta(folder)
        if meta is None:
            continue
        error_correction, rank = meta

        data = json.loads(run_json.read_text(encoding="utf-8"))
        avg_ppl = float(data["results"]["avg_ppl"])
        b_tok = float(data["traffic_totals"]["bytes_per_token"])

        if not (math.isfinite(avg_ppl) and math.isfinite(b_tok)):
            continue

        all_points.append(
            {
                "error_correction": error_correction,
                "rank": rank,
                "avg_ppl": avg_ppl,
                "b_tok": b_tok,
                "run_id": data.get("run_id", ""),
            }
        )

    if not all_points:
        raise ValueError(f"No valid runs found in {args.results_dir}")

    dropped_by_range = 0
    if y_range is not None:
        low, high = y_range
        before = len(all_points)
        all_points = [p for p in all_points if low <= float(p["avg_ppl"]) <= high]
        dropped_by_range = before - len(all_points)
        if not all_points:
            raise ValueError(
                f"No rows after --avg-ppl-range filter: {args.avg_ppl_range}"
            )

    frontier = pareto_frontier(all_points)

    ranks = sorted({int(p["rank"]) for p in all_points})
    cmap = plt.get_cmap("tab10")
    rank_colors = {rank: cmap(i % cmap.N) for i, rank in enumerate(ranks)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for rank in ranks:
        rank_points = [p for p in all_points if int(p["rank"]) == rank]
        x_vals = [float(p["b_tok"]) for p in rank_points]
        y_vals = [float(p["avg_ppl"]) for p in rank_points]
        ax.scatter(
            x_vals,
            y_vals,
            s=76,
            color=rank_colors[rank],
            edgecolors="black",
            linewidths=0.5,
            label=f"rank {rank}",
            zorder=2,
        )

    x_pf = [float(p["b_tok"]) for p in frontier]
    y_pf = [float(p["avg_ppl"]) for p in frontier]
    ax.plot(
        x_pf,
        y_pf,
        color="#111111",
        linewidth=2.3,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor="#111111",
        label="Pareto frontier",
        zorder=3,
    )

    for idx, point in enumerate(frontier):
        x = float(point["b_tok"])
        y = float(point["avg_ppl"])
        x_off = 6
        y_off = 8 if idx % 2 == 0 else -10
        ax.annotate(
            ppl_label(y),
            xy=(x, y),
            xytext=(x_off, y_off),
            textcoords="offset points",
            fontsize=8,
            color="#111111",
            ha="left",
            va="bottom" if y_off > 0 else "top",
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.78,
            },
            zorder=4,
        )

    ax.set_title("Pareto Frontier (all error-correction runs, colored by rank)")
    ax.set_xlabel("B/tok (bytes per token, lower is better)")
    ax.set_ylabel("avg_ppl (lower is better)")
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
    elif args.avg_ppl_max is not None:
        ax.set_ylim(top=args.avg_ppl_max)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="Rank", ncol=2)
    fig.tight_layout()
    fig.savefig(args.output)

    print(f"Saved figure: {args.output}")
    if y_range is not None:
        print(f"Applied avg_ppl analysis filter: {y_range[0]}~{y_range[1]}")
    print(
        f"kept points={len(all_points)}, dropped_by_range={dropped_by_range}, "
        f"pareto points={len(frontier)}"
    )


if __name__ == "__main__":
    main()
