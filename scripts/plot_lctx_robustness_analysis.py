#!/usr/bin/env python3
"""Plot long-context robustness normalized against baseline runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("results/Qwen/Qwen3-32B/lctx_robustness_baseline"),
        help="Directory that contains baseline long-context result subdirectories.",
    )
    parser.add_argument(
        "--compression-dir",
        type=Path,
        default=Path("results/Qwen/Qwen3-32B/lctx_robustness"),
        help="Directory that contains compression long-context result subdirectories.",
    )
    parser.add_argument(
        "--output-subplots",
        type=Path,
        default=None,
        help=(
            "Output path for the multi-subplot figure. "
            "Default: <shared-parent>/lctx_robustness_analysis_subplots.png"
        ),
    )
    parser.add_argument(
        "--output-combined",
        type=Path,
        default=None,
        help=(
            "Output path for the combined figure. "
            "Default: <shared-parent>/lctx_robustness_analysis_combined.png"
        ),
    )
    return parser.parse_args()


def latest_run_json(result_dir: Path) -> Path | None:
    run_files = sorted(result_dir.glob("run_*.json"))
    if not run_files:
        return None
    return run_files[-1]


def load_run_record(run_json: Path) -> dict[str, float | int | str]:
    data = json.loads(run_json.read_text(encoding="utf-8"))

    try:
        model_name = str(data["args"]["model_name"])
        max_length = int(data["args"]["max_length"])
        avg_ppl = float(data["results"]["avg_ppl"])
        bytes_per_token = float(data["traffic_totals"]["bytes_per_token"])
    except KeyError as exc:
        raise KeyError(f"Missing required field in {run_json}: {exc}") from exc

    if not math.isfinite(avg_ppl):
        raise ValueError(f"Non-finite avg_ppl in {run_json}: {avg_ppl}")
    if not math.isfinite(bytes_per_token):
        raise ValueError(
            f"Non-finite bytes_per_token in {run_json}: {bytes_per_token}"
        )
    if bytes_per_token <= 0:
        raise ValueError(f"bytes_per_token must be > 0 in {run_json}: {bytes_per_token}")
    if avg_ppl <= 0:
        raise ValueError(f"avg_ppl must be > 0 in {run_json}: {avg_ppl}")

    return {
        "source_dir": run_json.parent.name,
        "run_json": str(run_json),
        "model_name": model_name,
        "max_length": max_length,
        "avg_ppl": avg_ppl,
        "bytes_per_token": bytes_per_token,
    }


def load_result_root(root_dir: Path) -> list[dict[str, float | int | str]]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {root_dir}")

    result_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    if not result_dirs:
        raise ValueError(f"No result subdirectories found under {root_dir}")

    rows: list[dict[str, float | int | str]] = []
    skipped_dirs: list[Path] = []
    for result_dir in result_dirs:
        run_json = latest_run_json(result_dir)
        if run_json is None:
            skipped_dirs.append(result_dir)
            continue
        rows.append(load_run_record(run_json))

    if not rows:
        raise ValueError(f"No run_*.json files found under {root_dir}")

    if skipped_dirs:
        skipped_list = ", ".join(str(path.name) for path in skipped_dirs)
        print(f"Skipped directories without run_*.json under {root_dir}: {skipped_list}")

    return rows


def validate_and_build_baseline(
    rows: list[dict[str, float | int | str]],
) -> tuple[str, dict[int, dict[str, float | int | str]]]:
    model_names = {str(row["model_name"]) for row in rows}
    if len(model_names) != 1:
        raise ValueError(
            "Baseline model_name mismatch: "
            + ", ".join(sorted(model_names))
        )
    model_name = next(iter(model_names))

    baseline_by_max_length: dict[int, dict[str, float | int | str]] = {}
    for row in rows:
        max_length = int(row["max_length"])
        if max_length in baseline_by_max_length:
            previous = baseline_by_max_length[max_length]
            raise ValueError(
                f"Multiple baseline runs found for max_length={max_length}: "
                f"{previous['source_dir']} and {row['source_dir']}"
            )
        baseline_by_max_length[max_length] = row

    if not baseline_by_max_length:
        raise ValueError("No valid baseline rows found.")

    return model_name, baseline_by_max_length


def build_compression_groups(
    rows: list[dict[str, float | int | str]],
    expected_model_name: str,
    baseline_by_max_length: dict[int, dict[str, float | int | str]],
) -> dict[int, list[dict[str, float | int | str]]]:
    groups: dict[int, list[dict[str, float | int | str]]] = {
        max_length: [] for max_length in sorted(baseline_by_max_length)
    }

    for row in rows:
        model_name = str(row["model_name"])
        if model_name != expected_model_name:
            raise ValueError(
                f"Compression model_name mismatch in {row['source_dir']}: "
                f"expected '{expected_model_name}', got '{model_name}'"
            )

        max_length = int(row["max_length"])
        if max_length not in baseline_by_max_length:
            raise ValueError(
                f"Compression run {row['source_dir']} has max_length={max_length}, "
                "but baseline does not contain that context length."
            )

        baseline = baseline_by_max_length[max_length]
        baseline_avg_ppl = float(baseline["avg_ppl"])
        baseline_bytes_per_token = float(baseline["bytes_per_token"])
        compression_avg_ppl = float(row["avg_ppl"])
        compression_bytes_per_token = float(row["bytes_per_token"])

        groups[max_length].append(
            {
                **row,
                "bits_per_weight": 16.0
                * compression_bytes_per_token
                / baseline_bytes_per_token,
                "normalized_avg_ppl": compression_avg_ppl / baseline_avg_ppl,
            }
        )

    missing_groups = [max_length for max_length, points in groups.items() if not points]
    if missing_groups:
        missing_text = ", ".join(str(max_length) for max_length in missing_groups)
        raise ValueError(
            "No compression runs found for baseline max_length values: "
            f"{missing_text}"
        )

    for points in groups.values():
        points.sort(
            key=lambda row: (
                float(row["bits_per_weight"]),
                float(row["normalized_avg_ppl"]),
                str(row["source_dir"]),
            )
        )

    return groups


def default_output_paths(
    baseline_dir: Path,
    compression_dir: Path,
) -> tuple[Path, Path]:
    shared_parent = Path(
        os.path.commonpath(
            [
                str(baseline_dir.resolve()),
                str(compression_dir.resolve()),
            ]
        )
    )
    return (
        shared_parent / "lctx_robustness_analysis_subplots.png",
        shared_parent / "lctx_robustness_analysis_combined.png",
    )


def point_label(point: dict[str, float | int | str]) -> str:
    return (
        f"({float(point['normalized_avg_ppl']):.3f}, "
        f"{float(point['bits_per_weight']):.2f})"
    )


def annotate_points(
    ax: plt.Axes,
    points: list[dict[str, float | int | str]],
    color: str,
    group_index: int,
) -> None:
    for point_index, point in enumerate(points):
        x = float(point["bits_per_weight"])
        y = float(point["normalized_avg_ppl"])
        x_off = 6 if (group_index + point_index) % 2 == 0 else -6
        y_off = 8 if point_index % 2 == 0 else -10
        ax.annotate(
            point_label(point),
            xy=(x, y),
            xytext=(x_off, y_off),
            textcoords="offset points",
            fontsize=7,
            color=color,
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


def axis_limits(
    groups: dict[int, list[dict[str, float | int | str]]]
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [float(point["bits_per_weight"]) for points in groups.values() for point in points]
    ys = [
        float(point["normalized_avg_ppl"]) for points in groups.values() for point in points
    ]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_pad = 0.05 * max(x_max - x_min, 0.5)
    y_pad = 0.05 * max(y_max - y_min, 0.05)

    return (
        (max(0.0, x_min - x_pad), x_max + x_pad),
        (max(0.0, min(0.95, y_min - y_pad)), y_max + y_pad),
    )


def render_subplots(
    output_path: Path,
    model_name: str,
    groups: dict[int, list[dict[str, float | int | str]]],
    colors: dict[int, str],
) -> None:
    max_lengths = sorted(groups)
    plot_count = len(max_lengths)
    ncols = min(3, plot_count)
    nrows = math.ceil(plot_count / ncols)
    figsize = (5.0 * ncols, 4.0 * nrows)

    (x_min, x_max), (y_min, y_max) = axis_limits(groups)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=200,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    flat_axes = list(axes.flat)
    for axis_index, max_length in enumerate(max_lengths):
        ax = flat_axes[axis_index]
        points = groups[max_length]
        color = colors[max_length]
        xs = [float(point["bits_per_weight"]) for point in points]
        ys = [float(point["normalized_avg_ppl"]) for point in points]

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5.5,
            zorder=3,
        )
        ax.scatter(
            xs,
            ys,
            s=68,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=2,
        )
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.6)
        annotate_points(ax, points, color=color, group_index=axis_index)
        ax.set_title(f"max_length={max_length}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, linestyle="--")

    for ax in flat_axes[plot_count:]:
        ax.set_visible(False)

    fig.suptitle(f"{model_name} long-context robustness vs baseline", fontsize=14)
    fig.supxlabel("bits_per_weight")
    fig.supylabel("normalized avg_ppl")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def render_combined(
    output_path: Path,
    model_name: str,
    groups: dict[int, list[dict[str, float | int | str]]],
    colors: dict[int, str],
) -> None:
    (x_min, x_max), (y_min, y_max) = axis_limits(groups)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for group_index, max_length in enumerate(sorted(groups)):
        points = groups[max_length]
        color = colors[max_length]
        xs = [float(point["bits_per_weight"]) for point in points]
        ys = [float(point["normalized_avg_ppl"]) for point in points]

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            label=f"max_length={max_length}",
            zorder=3,
        )
        ax.scatter(
            xs,
            ys,
            s=68,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=2,
        )
        annotate_points(ax, points, color=color, group_index=group_index)

    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{model_name} long-context robustness by max_length")
    ax.set_xlabel("bits_per_weight")
    ax.set_ylabel("normalized avg_ppl")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    baseline_rows = load_result_root(args.baseline_dir)
    model_name, baseline_by_max_length = validate_and_build_baseline(baseline_rows)

    compression_rows = load_result_root(args.compression_dir)
    groups = build_compression_groups(
        compression_rows,
        expected_model_name=model_name,
        baseline_by_max_length=baseline_by_max_length,
    )

    default_subplots, default_combined = default_output_paths(
        args.baseline_dir,
        args.compression_dir,
    )
    output_subplots = args.output_subplots or default_subplots
    output_combined = args.output_combined or default_combined

    cmap = plt.get_cmap("tab10")
    colors = {
        max_length: cmap(index % cmap.N)
        for index, max_length in enumerate(sorted(groups))
    }

    render_subplots(
        output_path=output_subplots,
        model_name=model_name,
        groups=groups,
        colors=colors,
    )
    render_combined(
        output_path=output_combined,
        model_name=model_name,
        groups=groups,
        colors=colors,
    )

    print(f"Model: {model_name}")
    print(
        "Baseline max_length values: "
        + ", ".join(str(max_length) for max_length in sorted(baseline_by_max_length))
    )
    for max_length in sorted(groups):
        print(
            f"max_length={max_length}: baseline avg_ppl="
            f"{float(baseline_by_max_length[max_length]['avg_ppl']):.10f}, "
            f"baseline bytes_per_token="
            f"{float(baseline_by_max_length[max_length]['bytes_per_token']):.10f}, "
            f"compression points={len(groups[max_length])}"
        )
    print(f"Saved subplot figure: {output_subplots}")
    print(f"Saved combined figure: {output_combined}")


if __name__ == "__main__":
    main()
