#!/usr/bin/env python3
"""Plot Pareto frontiers for final_benchmark result families."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


SERIES_CONFIG = {
    "bitsqz_only": "bitsqz",
    "bitsqz_topk_0_001": "bitsqz_topk",
    "svd_bitsqz_topk_0_001": "svd_bitsqz_topk",
    "error_correction": "error_correction",
}

SERIES_STYLES = {
    "bitsqz_only": {"scatter": "#d62728", "line": "#d62728"},
    "bitsqz_topk_0_001": {"scatter": "#ff7f0e", "line": "#ff7f0e"},
    "svd_bitsqz_topk_0_001": {"scatter": "#1f77b4", "line": "#1f77b4"},
    "error_correction": {"scatter": "#2ca02c", "line": "#2ca02c"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/Qwen/Qwen3.5-9B"),
        help="Directory that contains bitsqz/bitsqz_topk/svd_bitsqz_topk/error_correction.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help="Output figure path (.png/.pdf/etc). Default: <results-dir>/pareto_final_benchmark.png",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: <results-dir>/pareto_final_benchmark_data.csv"
        ),
    )
    parser.add_argument(
        "--avg-ppl-range",
        type=str,
        default=None,
        help="Optional analysis range for avg_ppl. Example: '7.6~8.2'.",
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
        f"Invalid --avg-ppl-range '{raw}'. Use formats like '7.6~8.2' or '7.6,8.2'."
    )


def pareto_frontier(
    points: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    ordered = sorted(
        points,
        key=lambda p: (
            float(p["bytes_per_token"]),
            float(p["avg_ppl"]),
            str(p["approach_name"]),
        ),
    )
    frontier: list[dict[str, float | str]] = []
    best_y = math.inf
    eps = 1e-12
    for point in ordered:
        y = float(point["avg_ppl"])
        if y < best_y - eps:
            frontier.append(point)
            best_y = y
    return frontier


def latest_run_json(approach_dir: Path) -> Path | None:
    run_files = sorted(approach_dir.glob("run_*.json"))
    if not run_files:
        return None
    return run_files[-1]


def short_label(approach_name: str, series_name: str) -> str:
    if approach_name.startswith(series_name + "_"):
        return approach_name[len(series_name) + 1 :]
    return approach_name


def build_x_ticks(points: list[dict[str, float | str]], step: int = 1000) -> list[int]:
    x_vals = [float(p["bytes_per_token"]) for p in points]
    if not x_vals:
        return []

    x_min = min(x_vals)
    x_max = max(x_vals)

    start = max(step, int(math.floor(x_min / step)) * step)
    end = int(math.ceil(x_max / step)) * step
    if end < start:
        end = start

    return list(range(start, end + step, step))


def frontier_label(
    point: dict[str, float | str],
    series_name: str,
    label_mode: str,
) -> str:
    if label_mode == "avg_ppl":
        return f"{float(point['avg_ppl']):.2f}"
    if label_mode == "bytes_per_token":
        return f"{int(point['bytes_per_token'])}"
    return short_label(str(point["approach_name"]), series_name)


def variant_output_path(
    base_output: Path,
    label_mode: str,
    frontier_only: bool,
) -> Path:
    stem = base_output.stem
    if frontier_only:
        stem = f"{stem}_frontier_only"
    if label_mode != "approach_name":
        stem = f"{stem}_label_{label_mode}"
    return base_output.with_name(f"{stem}{base_output.suffix}")


def render_figure(
    output_path: Path,
    grouped: dict[str, list[dict[str, float | str]]],
    frontiers: dict[str, list[dict[str, float | str]]],
    all_rows: list[dict[str, float | str]],
    y_range: tuple[float, float] | None,
    label_mode: str,
    frontier_only: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for series_idx, series_name in enumerate(SERIES_CONFIG):
        points = grouped[series_name]
        frontier = frontiers[series_name]
        style = SERIES_STYLES[series_name]

        x_all = [float(p["bytes_per_token"]) for p in points]
        y_all = [float(p["avg_ppl"]) for p in points]
        x_pf = [float(p["bytes_per_token"]) for p in frontier]
        y_pf = [float(p["avg_ppl"]) for p in frontier]

        if not frontier_only:
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
            label=series_name,
            zorder=3,
        )

        for frontier_idx, point in enumerate(frontier):
            x = float(point["bytes_per_token"])
            y = float(point["avg_ppl"])
            label = frontier_label(point, series_name, label_mode)
            x_off = 6 if series_idx % 2 == 0 else -6
            y_off = 8 if frontier_idx % 2 == 0 else -10
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(x_off, y_off),
                textcoords="offset points",
                fontsize=7,
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

    ax.set_title("Pareto frontiers for final_benchmark series")
    ax.set_xlabel("bytes per token (lower is better)")
    ax.set_ylabel("avg_ppl (lower is better)")
    x_ticks = build_x_ticks(all_rows, step=1000)
    if x_ticks:
        ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    y_range = parse_avg_ppl_range(args.avg_ppl_range)

    results_dir = args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    output_figure = args.output_figure or (results_dir / "pareto_final_benchmark.png")
    output_csv = args.output_csv or (results_dir / "pareto_final_benchmark_data.csv")

    grouped: dict[str, list[dict[str, float | str]]] = {
        name: [] for name in SERIES_CONFIG
    }

    for series_name, relative_dir in SERIES_CONFIG.items():
        series_dir = results_dir / relative_dir
        if not series_dir.exists():
            raise FileNotFoundError(f"Series directory not found: {series_dir}")

        for approach_dir in sorted(
            path for path in series_dir.iterdir() if path.is_dir()
        ):
            run_json = latest_run_json(approach_dir)
            if run_json is None:
                continue

            data = json.loads(run_json.read_text(encoding="utf-8"))
            avg_ppl = float(data["results"]["avg_ppl"])
            bytes_per_token = float(data["traffic_totals"]["bytes_per_token"])

            if not (math.isfinite(avg_ppl) and math.isfinite(bytes_per_token)):
                continue

            grouped[series_name].append(
                {
                    "series_name": series_name,
                    "approach_name": approach_dir.name,
                    "avg_ppl": avg_ppl,
                    "bytes_per_token": bytes_per_token,
                }
            )

    dropped_by_range: dict[str, int] = {}
    for series_name, points in grouped.items():
        if not points:
            raise ValueError(
                f"No valid runs found for series '{series_name}' under {results_dir}"
            )
        before = len(points)
        if y_range is not None:
            low, high = y_range
            points = [p for p in points if low <= float(p["avg_ppl"]) <= high]
        dropped_by_range[series_name] = before - len(points)
        if not points:
            raise ValueError(
                f"No rows after filtering for series '{series_name}' "
                f"with --avg-ppl-range={args.avg_ppl_range}"
            )
        grouped[series_name] = points

    all_rows: list[dict[str, float | str]] = []
    for series_name in SERIES_CONFIG:
        all_rows.extend(grouped[series_name])

    all_rows = sorted(
        all_rows,
        key=lambda p: (
            str(p["series_name"]),
            float(p["bytes_per_token"]),
            float(p["avg_ppl"]),
            str(p["approach_name"]),
        ),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["approach_name", "avg_ppl", "bytes_per_token"]
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    "approach_name": row["approach_name"],
                    "avg_ppl": f"{float(row['avg_ppl']):.10f}",
                    "bytes_per_token": f"{float(row['bytes_per_token']):.10f}",
                }
            )

    frontiers = {
        series_name: pareto_frontier(points) for series_name, points in grouped.items()
    }

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure_modes = ["approach_name", "avg_ppl", "bytes_per_token"]
    figure_variants = [False, True]
    figure_paths: list[Path] = []
    for frontier_only in figure_variants:
        for mode in figure_modes:
            fig_path = variant_output_path(
                output_figure,
                mode,
                frontier_only=frontier_only,
            )
            figure_paths.append(fig_path)
            render_figure(
                output_path=fig_path,
                grouped=grouped,
                frontiers=frontiers,
                all_rows=all_rows,
                y_range=y_range,
                label_mode=mode,
                frontier_only=frontier_only,
            )

    for fig_path in figure_paths:
        print(f"Saved figure: {fig_path}")
    print(f"Saved CSV: {output_csv}")
    if y_range is not None:
        print(f"Applied avg_ppl analysis filter: {y_range[0]}~{y_range[1]}")
    for series_name in SERIES_CONFIG:
        print(
            f"series {series_name}: kept points={len(grouped[series_name])}, "
            f"dropped_by_range={dropped_by_range[series_name]}, "
            f"pareto points={len(frontiers[series_name])}"
        )


if __name__ == "__main__":
    main()
