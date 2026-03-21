#!/usr/bin/env python3
"""Plot Pareto frontier comparing trunc_approx vs trunc_slice runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


SCATTER_COLORS = {
    "trunc_approx": "#ffb3b3",
    "trunc_slice": "#9ec9ff",
}

LINE_COLORS = {
    "trunc_approx": "#d62728",
    "trunc_slice": "#1f77b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "show_the_better_onlinesvd_niters_is_comparable_to_fullsvd"
        ),
        help="Directory containing subfolders like trunc_approx_fp16 and trunc_slice_fp16.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "show_the_better_onlinesvd_niters_is_comparable_to_fullsvd/"
            "pareto_trunc_approx_vs_trunc_slice.png"
        ),
        help="Output figure path (.png/.pdf/etc).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["trunc_approx", "trunc_slice"],
        help="Mode prefixes to compare.",
    )
    parser.add_argument(
        "--avg-ppl-range",
        type=str,
        default=None,
        help="Optional y-axis range, e.g. '8.2~10' or '8.2,10'.",
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
    points: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
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


def detect_mode(folder_name: str, modes: Iterable[str]) -> str | None:
    for mode in modes:
        if folder_name.startswith(mode + "_"):
            return mode
    return None


def extract_uv_format(folder_name: str, mode: str) -> str:
    return folder_name[len(mode) + 1 :]


def main() -> None:
    args = parse_args()
    y_range = parse_avg_ppl_range(args.avg_ppl_range)

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    grouped: dict[str, list[dict[str, float | str]]] = {mode: [] for mode in args.modes}

    for run_json in sorted(args.results_dir.glob("*/run_*.json")):
        folder = run_json.parent.name
        mode = detect_mode(folder, args.modes)
        if mode is None:
            continue

        uv_format = extract_uv_format(folder, mode)

        data = json.loads(run_json.read_text(encoding="utf-8"))
        avg_ppl = float(data["results"]["avg_ppl"])
        b_tok = float(data["traffic_totals"]["bytes_per_token"])

        if not (math.isfinite(avg_ppl) and math.isfinite(b_tok)):
            continue

        grouped[mode].append(
            {
                "uv_format": uv_format,
                "avg_ppl": avg_ppl,
                "b_tok": b_tok,
                "run_id": data.get("run_id", ""),
            }
        )

    for mode, points in grouped.items():
        if not points:
            raise ValueError(
                f"No valid runs found for mode '{mode}' in {args.results_dir}"
            )

    dropped_by_range: dict[str, int] = {}
    if y_range is not None:
        low, high = y_range
        for mode in args.modes:
            before = len(grouped[mode])
            grouped[mode] = [
                p for p in grouped[mode] if low <= float(p["avg_ppl"]) <= high
            ]
            dropped_by_range[mode] = before - len(grouped[mode])
            if not grouped[mode]:
                raise ValueError(
                    f"No rows after --avg-ppl-range for mode={mode}: {args.avg_ppl_range}"
                )
    else:
        dropped_by_range = {mode: 0 for mode in args.modes}

    frontiers = {mode: pareto_frontier(points) for mode, points in grouped.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for mode in args.modes:
        points = grouped[mode]
        frontier = frontiers[mode]

        x_all = [float(p["b_tok"]) for p in points]
        y_all = [float(p["avg_ppl"]) for p in points]
        x_pf = [float(p["b_tok"]) for p in frontier]
        y_pf = [float(p["avg_ppl"]) for p in frontier]

        ax.scatter(
            x_all,
            y_all,
            s=72,
            color=SCATTER_COLORS.get(mode, "#bbbbbb"),
            edgecolors="black",
            linewidths=0.5,
            label=f"{mode} (all uv_format)",
            zorder=2,
        )
        ax.plot(
            x_pf,
            y_pf,
            color=LINE_COLORS.get(mode, "#333333"),
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            label=f"{mode} Pareto frontier",
            zorder=3,
        )

        text_color = LINE_COLORS.get(mode, "#333333")
        for idx, point in enumerate(frontier):
            x = float(point["b_tok"])
            y = float(point["avg_ppl"])
            uv_label = str(point["uv_format"])
            x_off = 6 if mode == args.modes[0] else -6
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
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.75,
                },
                zorder=4,
            )

    ax.set_title(f"Pareto Frontier Comparison ({' vs '.join(args.modes)})")
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
    for mode in args.modes:
        print(
            f"{mode}: kept points={len(grouped[mode])}, "
            f"dropped_by_range={dropped_by_range[mode]}, pareto points={len(frontiers[mode])}"
        )


if __name__ == "__main__":
    main()
