#!/usr/bin/env python3
"""Plot long-context robustness comparisons normalized against baseline runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

Row = dict[str, float | int | str]
Groups = dict[int, dict[str, list[Row]]]

SERIES_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
SERIES_LINESTYLES = ["-", "--", "-.", ":"]
ANNOTATION_OFFSETS = [
    (6, 10),
    (-6, -12),
    (8, 16),
    (-8, -18),
    (10, 6),
    (-10, 12),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("results/Qwen/Qwen3-32B/lctx_robustness_baseline"),
        help="Directory that contains baseline long-context result subdirectories.",
    )
    parser.add_argument(
        "--comparison-dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "Directory that contains comparison long-context result subdirectories. "
            "Pass this argument multiple times; at least two directories are required."
        ),
    )
    parser.add_argument(
        "--output-subplots",
        type=Path,
        default=None,
        help=(
            "Output path for the multi-subplot figure. "
            "Default: <shared-parent>/lctx_robustness_comparison_subplots.png"
        ),
    )
    parser.add_argument(
        "--output-combined",
        type=Path,
        default=None,
        help=(
            "Output path for the combined figure. "
            "Default: <shared-parent>/lctx_robustness_comparison_combined.png"
        ),
    )
    return parser.parse_args()


def latest_run_json(result_dir: Path) -> Path | None:
    run_files = sorted(result_dir.glob("run_*.json"))
    if not run_files:
        return None
    return run_files[-1]


def load_run_record(run_json: Path) -> Row:
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


def load_result_root(root_dir: Path) -> list[Row]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {root_dir}")

    result_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    if not result_dirs:
        raise ValueError(f"No result subdirectories found under {root_dir}")

    rows: list[Row] = []
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


def validate_and_build_baseline(rows: list[Row]) -> tuple[str, dict[int, Row]]:
    model_names = {str(row["model_name"]) for row in rows}
    if len(model_names) != 1:
        raise ValueError("Baseline model_name mismatch: " + ", ".join(sorted(model_names)))
    model_name = next(iter(model_names))

    baseline_by_max_length: dict[int, Row] = {}
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


def validate_comparison_dirs(comparison_dirs: list[Path]) -> list[str]:
    if len(comparison_dirs) < 2:
        raise ValueError("Pass --comparison-dir at least twice.")

    labels = [path.name for path in comparison_dirs]
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(
            "Comparison directory basenames must be unique so the legend is unambiguous: "
            + ", ".join(duplicate_labels)
        )

    return labels


def build_comparison_groups(
    comparison_rows_by_label: dict[str, list[Row]],
    expected_model_name: str,
    baseline_by_max_length: dict[int, Row],
    comparison_order: list[str],
) -> Groups:
    groups: Groups = {
        max_length: {label: [] for label in comparison_order}
        for max_length in sorted(baseline_by_max_length)
    }

    for label in comparison_order:
        rows = comparison_rows_by_label[label]
        for row in rows:
            model_name = str(row["model_name"])
            if model_name != expected_model_name:
                raise ValueError(
                    f"Comparison model_name mismatch in {label}/{row['source_dir']}: "
                    f"expected '{expected_model_name}', got '{model_name}'"
                )

            max_length = int(row["max_length"])
            if max_length not in baseline_by_max_length:
                raise ValueError(
                    f"Comparison run {label}/{row['source_dir']} has "
                    f"max_length={max_length}, but baseline does not contain "
                    "that context length."
                )

            baseline = baseline_by_max_length[max_length]
            baseline_avg_ppl = float(baseline["avg_ppl"])
            baseline_bytes_per_token = float(baseline["bytes_per_token"])
            comparison_avg_ppl = float(row["avg_ppl"])
            comparison_bytes_per_token = float(row["bytes_per_token"])

            groups[max_length][label].append(
                {
                    **row,
                    "comparison_label": label,
                    "bits_per_weight": 16.0
                    * comparison_bytes_per_token
                    / baseline_bytes_per_token,
                    "normalized_avg_ppl": comparison_avg_ppl / baseline_avg_ppl,
                }
            )

        missing_groups = [
            max_length
            for max_length, by_label in groups.items()
            if not by_label[label]
        ]
        if missing_groups:
            missing_text = ", ".join(str(max_length) for max_length in missing_groups)
            raise ValueError(
                f"No comparison runs found in {label} for baseline max_length values: "
                f"{missing_text}"
            )

    for by_label in groups.values():
        for points in by_label.values():
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
    comparison_dirs: list[Path],
) -> tuple[Path, Path]:
    shared_parent = Path(
        os.path.commonpath(
            [
                str(baseline_dir.resolve()),
                *(str(path.resolve()) for path in comparison_dirs),
            ]
        )
    )
    return (
        shared_parent / "lctx_robustness_comparison_subplots.png",
        shared_parent / "lctx_robustness_comparison_combined.png",
    )


def point_label(point: Row) -> str:
    return (
        f"({float(point['normalized_avg_ppl']):.3f}, "
        f"{float(point['bits_per_weight']):.2f})"
    )


def annotate_points(
    ax: plt.Axes,
    points: list[Row],
    color: str,
    group_index: int,
    series_index: int,
) -> None:
    for point_index, point in enumerate(points):
        x = float(point["bits_per_weight"])
        y = float(point["normalized_avg_ppl"])
        offset_index = (
            group_index * 3 + series_index * 2 + point_index
        ) % len(ANNOTATION_OFFSETS)
        x_off, y_off = ANNOTATION_OFFSETS[offset_index]
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


def axis_limits(groups: Groups) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [
        float(point["bits_per_weight"])
        for by_label in groups.values()
        for points in by_label.values()
        for point in points
    ]
    ys = [
        float(point["normalized_avg_ppl"])
        for by_label in groups.values()
        for points in by_label.values()
        for point in points
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


def build_series_styles(comparison_order: list[str]) -> dict[str, dict[str, object]]:
    styles: dict[str, dict[str, object]] = {}
    for index, label in enumerate(comparison_order):
        styles[label] = {
            "marker": SERIES_MARKERS[index % len(SERIES_MARKERS)],
            "linestyle": SERIES_LINESTYLES[index % len(SERIES_LINESTYLES)],
        }
    return styles


def max_length_legend_handles(
    max_lengths: list[int],
    colors: dict[int, str],
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=colors[max_length],
            marker="o",
            linestyle="-",
            markersize=6,
            linewidth=2.2,
            label=f"max_length={max_length}",
        )
        for max_length in max_lengths
    ]


def comparison_legend_handles(
    comparison_order: list[str],
    styles: dict[str, dict[str, object]],
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="black",
            marker=str(styles[label]["marker"]),
            linestyle=styles[label]["linestyle"],
            markersize=6,
            linewidth=2.0,
            label=label,
        )
        for label in comparison_order
    ]


def render_subplots(
    output_path: Path,
    model_name: str,
    groups: Groups,
    colors: dict[int, str],
    comparison_order: list[str],
    styles: dict[str, dict[str, object]],
) -> None:
    max_lengths = sorted(groups)
    plot_count = len(max_lengths)
    ncols = min(3, plot_count)
    nrows = math.ceil(plot_count / ncols)
    figsize = (5.0 * ncols, 4.0 * nrows)
    legend_handles = comparison_legend_handles(comparison_order, styles)

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
        color = colors[max_length]

        for series_index, label in enumerate(comparison_order):
            points = groups[max_length][label]
            xs = [float(point["bits_per_weight"]) for point in points]
            ys = [float(point["normalized_avg_ppl"]) for point in points]

            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.0,
                marker=str(styles[label]["marker"]),
                linestyle=styles[label]["linestyle"],
                markersize=5.5,
                zorder=3,
            )
            ax.scatter(
                xs,
                ys,
                s=68,
                color=color,
                marker=str(styles[label]["marker"]),
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )
            annotate_points(
                ax,
                points,
                color=color,
                group_index=axis_index,
                series_index=series_index,
            )

        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(f"max_length={max_length}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(
            handles=legend_handles,
            title="comparison_dir",
            loc="best",
            fontsize=8,
            title_fontsize=8,
            framealpha=0.9,
        )

    for ax in flat_axes[plot_count:]:
        ax.set_visible(False)

    fig.suptitle(f"{model_name} long-context robustness comparison", fontsize=14)
    fig.supxlabel("bits_per_weight")
    fig.supylabel("normalized avg_ppl")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def render_combined(
    output_path: Path,
    model_name: str,
    groups: Groups,
    colors: dict[int, str],
    comparison_order: list[str],
    styles: dict[str, dict[str, object]],
) -> None:
    max_lengths = sorted(groups)
    (x_min, x_max), (y_min, y_max) = axis_limits(groups)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    for group_index, max_length in enumerate(max_lengths):
        color = colors[max_length]
        for series_index, label in enumerate(comparison_order):
            points = groups[max_length][label]
            xs = [float(point["bits_per_weight"]) for point in points]
            ys = [float(point["normalized_avg_ppl"]) for point in points]

            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.2,
                marker=str(styles[label]["marker"]),
                linestyle=styles[label]["linestyle"],
                markersize=5.5,
                zorder=3,
            )
            ax.scatter(
                xs,
                ys,
                s=68,
                color=color,
                marker=str(styles[label]["marker"]),
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )
            annotate_points(
                ax,
                points,
                color=color,
                group_index=group_index,
                series_index=series_index,
            )

    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{model_name} long-context robustness by max_length")
    ax.set_xlabel("bits_per_weight")
    ax.set_ylabel("normalized avg_ppl")
    ax.grid(True, alpha=0.3, linestyle="--")

    max_length_legend = ax.legend(
        handles=max_length_legend_handles(max_lengths, colors),
        title="max_length",
        loc="upper left",
    )
    ax.add_artist(max_length_legend)
    ax.legend(
        handles=comparison_legend_handles(comparison_order, styles),
        title="comparison_dir",
        loc="upper right",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    comparison_dirs = args.comparison_dir
    comparison_order = validate_comparison_dirs(comparison_dirs)

    baseline_rows = load_result_root(args.baseline_dir)
    model_name, baseline_by_max_length = validate_and_build_baseline(baseline_rows)

    comparison_rows_by_label = {
        label: load_result_root(path)
        for label, path in zip(comparison_order, comparison_dirs)
    }
    groups = build_comparison_groups(
        comparison_rows_by_label,
        expected_model_name=model_name,
        baseline_by_max_length=baseline_by_max_length,
        comparison_order=comparison_order,
    )

    default_subplots, default_combined = default_output_paths(
        args.baseline_dir,
        comparison_dirs,
    )
    output_subplots = args.output_subplots or default_subplots
    output_combined = args.output_combined or default_combined

    cmap = plt.get_cmap("tab10")
    colors = {
        max_length: cmap(index % cmap.N)
        for index, max_length in enumerate(sorted(groups))
    }
    styles = build_series_styles(comparison_order)

    render_subplots(
        output_path=output_subplots,
        model_name=model_name,
        groups=groups,
        colors=colors,
        comparison_order=comparison_order,
        styles=styles,
    )
    render_combined(
        output_path=output_combined,
        model_name=model_name,
        groups=groups,
        colors=colors,
        comparison_order=comparison_order,
        styles=styles,
    )

    print(f"Model: {model_name}")
    print(
        "Baseline max_length values: "
        + ", ".join(str(max_length) for max_length in sorted(baseline_by_max_length))
    )
    for label, path in zip(comparison_order, comparison_dirs):
        print(f"Comparison directory: {label} ({path})")
        for max_length in sorted(groups):
            print(
                f"  max_length={max_length}: baseline avg_ppl="
                f"{float(baseline_by_max_length[max_length]['avg_ppl']):.10f}, "
                f"baseline bytes_per_token="
                f"{float(baseline_by_max_length[max_length]['bytes_per_token']):.10f}, "
                f"comparison points={len(groups[max_length][label])}"
            )
    print(f"Saved subplot figure: {output_subplots}")
    print(f"Saved combined figure: {output_combined}")


if __name__ == "__main__":
    main()
