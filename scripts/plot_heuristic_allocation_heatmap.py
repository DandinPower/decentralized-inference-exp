#!/usr/bin/env python3
"""Plot offline heuristic allocation search figures from benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


DEFAULT_BENCHMARK_DIR = Path("results/Qwen/Qwen3-8B/heuristic_benchmark")
DEFAULT_OUTPUT = Path("results/Qwen/Qwen3-8B/8192_bits_budget_0.75_heatmap.png")
OFFLINE_SEARCH_INTERPOLATION_POINTS = 120
OFFLINE_SEARCH_CMAP = LinearSegmentedColormap.from_list(
    "offline_search_blue_teal_warm",
    ["#2166AC", "#1B9E77", "#FEE08B", "#E69F00", "#D55E00"],
    N=256,
)
MISSING_CELL_FACE = "#D9D9D9"
MISSING_CELL_EDGE = "#8C8C8C"
MISSING_CELL_HATCH = "////"
OPTIMUM_FACE = "#FFD166"
OPTIMUM_EDGE = "#111111"
NEAR_OPTIMAL_EDGE = "#E69F00"
OFFLINE_SEARCH_CMAP.set_bad(color=MISSING_CELL_FACE)

POLICY_COLORS = {
    "svd": "#4C78A8",
    "topk": "#F58518",
    "error": "#54A24B",
}

RUN_DIR_RE = re.compile(
    r"^error_correction_"
    r"(?P<max_length>\d+)_"
    r"(?P<stride>\d+)_"
    r"bits_budget_(?P<bits_budget>\d+(?:\.\d+)?)_"
    r"topk_portion_(?P<topk>\d+(?:\.\d+)?)_"
    r"error_topk_portion_(?P<error>\d+(?:\.\d+)?)$"
)


@dataclass(frozen=True)
class BenchmarkRow:
    max_length: int
    stride: int
    bits_budget: float
    topk_ratio: float
    error_ratio: float
    avg_ppl: float
    run_json: str

    @property
    def svd_ratio(self) -> float:
        return 1.0 - self.topk_ratio - self.error_ratio


@dataclass(frozen=True)
class SliceKey:
    max_length: int
    stride: int
    bits_budget: float


@dataclass(frozen=True)
class PolicyFitPoint:
    max_length: int
    stride: int
    bits_budget: float
    best_topk_ratio: float
    best_error_ratio: float
    best_svd_ratio: float
    best_avg_ppl: float
    fitted_topk_ratio: float
    fitted_error_ratio: float
    fitted_svd_ratio: float
    near_min_topk_ratio: float
    near_max_topk_ratio: float
    near_min_error_ratio: float
    near_max_error_ratio: float
    near_min_svd_ratio: float
    near_max_svd_ratio: float
    near_count: int
    total_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory containing heuristic benchmark result subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for --all outputs. Default: parent of --benchmark-dir "
            "(for example results/Qwen/Qwen3-8B)."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow missing topk/error cells and render them as masked cells.",
    )
    parser.add_argument(
        "--near-optimal-rel",
        type=float,
        default=0.003,
        help=(
            "Relative regret threshold for near-optimal basins, e.g. 0.01 means "
            "within 1%% of the best avg_ppl in a slice."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="Soft centroid temperature applied to relative regret.",
    )
    return parser.parse_args()


def latest_run_json(result_dir: Path) -> Path | None:
    run_files = sorted(result_dir.glob("run_*.json"))
    if not run_files:
        return None
    return run_files[-1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_avg_ppl(path: Path) -> float:
    data = load_json(path)
    try:
        avg_ppl = float(data["results"]["avg_ppl"])
    except KeyError as exc:
        raise KeyError(f"Missing results.avg_ppl in {path}") from exc

    if not math.isfinite(avg_ppl) or avg_ppl <= 0:
        raise ValueError(f"avg_ppl must be finite and > 0 in {path}: {avg_ppl}")
    return avg_ppl


def parse_result_dir(result_dir: Path) -> tuple[int, int, float, float, float] | None:
    match = RUN_DIR_RE.match(result_dir.name)
    if match is None:
        return None

    return (
        int(match.group("max_length")),
        int(match.group("stride")),
        float(match.group("bits_budget")),
        float(match.group("topk")),
        float(match.group("error")),
    )


def is_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def load_all_rows(benchmark_dir: Path) -> list[BenchmarkRow]:
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    rows: list[BenchmarkRow] = []
    skipped_without_runs: list[str] = []

    for result_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
        parsed = parse_result_dir(result_dir)
        if parsed is None:
            continue

        max_length, stride, bits_budget, topk_ratio, error_ratio = parsed
        run_json = latest_run_json(result_dir)
        if run_json is None:
            skipped_without_runs.append(result_dir.name)
            continue

        rows.append(
            BenchmarkRow(
                max_length=max_length,
                stride=stride,
                bits_budget=bits_budget,
                topk_ratio=topk_ratio,
                error_ratio=error_ratio,
                avg_ppl=load_avg_ppl(run_json),
                run_json=str(run_json),
            )
        )

    if skipped_without_runs:
        print(
            "Skipped directories without run_*.json: "
            + ", ".join(skipped_without_runs)
        )

    if not rows:
        raise ValueError(f"No heuristic benchmark runs found under {benchmark_dir}")

    return rows


# def load_grid_rows(
#     benchmark_dir: Path,
#     max_length: int,
#     stride: int,
#     bits_budget: float,
#     allow_incomplete: bool = False,
# ) -> list[BenchmarkRow]:
#     rows: list[BenchmarkRow] = []
#     for row in load_all_rows(benchmark_dir):
#         if (
#             row.max_length == max_length
#             and row.stride == stride
#             and is_close(row.bits_budget, bits_budget)
#         ):
#             rows.append(row)

#     if not rows:
#         raise ValueError(
#             "No matching benchmark runs found for "
#             f"max_length={max_length}, stride={stride}, bits_budget={bits_budget}"
#         )

#     validate_grid(
#         rows,
#         allow_incomplete=allow_incomplete,
#         context=f"L={max_length}, stride={stride}, budget={bits_budget:g}",
#     )
#     return rows


def validate_grid(
    rows: list[BenchmarkRow],
    allow_incomplete: bool,
    context: str,
) -> bool:
    topk_values = sorted({row.topk_ratio for row in rows})
    error_values = sorted({row.error_ratio for row in rows})
    expected_count = len(topk_values) * len(error_values)
    seen = {(row.topk_ratio, row.error_ratio) for row in rows}

    if len(seen) != len(rows):
        raise ValueError(f"Duplicate grid points found in benchmark rows: {context}")

    if len(rows) == expected_count:
        return True

    missing = [
        (topk, error)
        for topk in topk_values
        for error in error_values
        if (topk, error) not in seen
    ]
    missing_text = ", ".join(
        f"(topk={topk:.2f}, error={error:.2f})" for topk, error in missing
    )
    message = f"Incomplete grid for {context}. Missing points: {missing_text}"
    if not allow_incomplete:
        raise ValueError(message)

    print(f"Warning: {message}")
    return False


def group_by_slice(rows: list[BenchmarkRow]) -> dict[SliceKey, list[BenchmarkRow]]:
    groups: dict[SliceKey, list[BenchmarkRow]] = {}
    for row in rows:
        key = SliceKey(row.max_length, row.stride, row.bits_budget)
        groups.setdefault(key, []).append(row)
    return groups


def grid_matrix(
    rows: list[BenchmarkRow]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topk_values = np.array(sorted({row.topk_ratio for row in rows}))
    error_values = np.array(sorted({row.error_ratio for row in rows}))
    z = np.full((len(error_values), len(topk_values)), np.nan, dtype=float)

    topk_index = {value: idx for idx, value in enumerate(topk_values)}
    error_index = {value: idx for idx, value in enumerate(error_values)}
    for row in rows:
        x_idx = topk_index[row.topk_ratio]
        y_idx = error_index[row.error_ratio]
        z[y_idx, x_idx] = row.avg_ppl

    return topk_values, error_values, z


def cell_edges(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        step = 0.05
        return np.array([values[0] - step / 2, values[0] + step / 2])

    mids = (values[:-1] + values[1:]) / 2
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def finite_value_limits(values: np.ndarray, context: str) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if not len(finite_values):
        raise ValueError(f"No finite avg_ppl values found for {context}")

    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    if math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-12):
        pad = max(abs(vmin), 1.0) * 1e-6
        vmin -= pad
        vmax += pad

    return vmin, vmax


def interpolation_axis_indices(
    source_values: np.ndarray,
    target_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped_targets = np.clip(target_values, source_values[0], source_values[-1])
    if len(source_values) == 1:
        indices = np.zeros_like(target_values, dtype=int)
        weights = np.zeros_like(target_values, dtype=float)
        return indices, indices, weights

    lower_indices = np.searchsorted(source_values, clipped_targets, side="right") - 1
    lower_indices = np.clip(lower_indices, 0, len(source_values) - 2)
    upper_indices = lower_indices + 1
    spans = source_values[upper_indices] - source_values[lower_indices]
    weights = (clipped_targets - source_values[lower_indices]) / spans

    return lower_indices, upper_indices, weights


def interpolate_heatmap_surface(
    topk_values: np.ndarray,
    error_values: np.ndarray,
    z: np.ndarray,
    resolution: int = OFFLINE_SEARCH_INTERPOLATION_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = cell_edges(topk_values)
    y_edges = cell_edges(error_values)
    smooth_topk = np.linspace(x_edges[0], x_edges[-1], resolution)
    smooth_error = np.linspace(y_edges[0], y_edges[-1], resolution)

    x0, x1, wx = interpolation_axis_indices(topk_values, smooth_topk)
    y0, y1, wy = interpolation_axis_indices(error_values, smooth_error)

    z00 = z[y0[:, None], x0[None, :]]
    z10 = z[y0[:, None], x1[None, :]]
    z01 = z[y1[:, None], x0[None, :]]
    z11 = z[y1[:, None], x1[None, :]]
    valid = (
        np.isfinite(z00)
        & np.isfinite(z10)
        & np.isfinite(z01)
        & np.isfinite(z11)
    )

    wx_grid = wx[None, :]
    wy_grid = wy[:, None]
    smooth_z = (
        z00 * (1.0 - wx_grid) * (1.0 - wy_grid)
        + z10 * wx_grid * (1.0 - wy_grid)
        + z01 * (1.0 - wx_grid) * wy_grid
        + z11 * wx_grid * wy_grid
    )
    smooth_z = np.where(valid, smooth_z, np.nan)

    return smooth_topk, smooth_error, smooth_z


def add_missing_cell_overlay(
    ax: plt.Axes,
    topk_values: np.ndarray,
    error_values: np.ndarray,
    z: np.ndarray,
) -> None:
    x_edges = cell_edges(topk_values)
    y_edges = cell_edges(error_values)

    for y_idx in range(len(error_values)):
        for x_idx in range(len(topk_values)):
            if math.isfinite(float(z[y_idx, x_idx])):
                continue

            ax.add_patch(
                Rectangle(
                    (x_edges[x_idx], y_edges[y_idx]),
                    x_edges[x_idx + 1] - x_edges[x_idx],
                    y_edges[y_idx + 1] - y_edges[y_idx],
                    facecolor=MISSING_CELL_FACE,
                    edgecolor=MISSING_CELL_EDGE,
                    hatch=MISSING_CELL_HATCH,
                    linewidth=0.45,
                    zorder=2,
                )
            )


def cell_text_color(value: float, vmin: float, vmax: float) -> str:
    red, green, blue, _ = OFFLINE_SEARCH_CMAP(
        Normalize(vmin=vmin, vmax=vmax, clip=True)(value)
    )
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111111" if luminance >= 0.56 else "#FFFFFF"


def near_optimal_rows(
    rows: list[BenchmarkRow],
    best: BenchmarkRow,
    near_optimal_rel: float,
) -> list[BenchmarkRow]:
    if near_optimal_rel < 0:
        raise ValueError("--near-optimal-rel must be >= 0")

    return [
        row
        for row in rows
        if (row.avg_ppl - best.avg_ppl) / best.avg_ppl <= near_optimal_rel
    ]


def scatter_near_optimal(
    ax: plt.Axes,
    rows: list[BenchmarkRow],
    size: float,
    zorder: float,
) -> None:
    if not rows:
        return

    x_values = [row.topk_ratio for row in rows]
    y_values = [row.error_ratio for row in rows]
    ax.scatter(
        x_values,
        y_values,
        marker="o",
        s=size * 1.18,
        facecolors="none",
        edgecolors=OPTIMUM_EDGE,
        linewidths=2.1,
        zorder=zorder,
    )
    ax.scatter(
        x_values,
        y_values,
        marker="o",
        s=size,
        facecolors="none",
        edgecolors=NEAR_OPTIMAL_EDGE,
        linewidths=1.35,
        zorder=zorder + 0.1,
    )


def scatter_empirical_optimum(
    ax: plt.Axes,
    best: BenchmarkRow,
    size: float,
    zorder: float,
    label: str | None = None,
) -> None:
    ax.scatter(
        [best.topk_ratio],
        [best.error_ratio],
        marker="*",
        s=size,
        facecolor=OPTIMUM_FACE,
        edgecolor=OPTIMUM_EDGE,
        linewidth=1.05,
        zorder=zorder,
        label=label,
    )


def offline_search_legend_handles(
    near_optimal_rel: float,
    optimum_label: str = "Empirical optimum",
) -> list[Line2D | Patch]:
    return [
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor=OPTIMUM_FACE,
            markeredgecolor=OPTIMUM_EDGE,
            markeredgewidth=1.05,
            markersize=12,
            label=optimum_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=NEAR_OPTIMAL_EDGE,
            markeredgewidth=1.8,
            markersize=8,
            label=f"Near-optimal (<= {near_optimal_rel * 100:g}% regret)",
        ),
        Patch(
            facecolor=MISSING_CELL_FACE,
            edgecolor=MISSING_CELL_EDGE,
            hatch=MISSING_CELL_HATCH,
            label="Missing/infeasible",
        ),
    ]


def compute_policy_fit_point(
    key: SliceKey,
    rows: list[BenchmarkRow],
    near_optimal_rel: float,
    temperature: float,
) -> PolicyFitPoint:
    if near_optimal_rel < 0:
        raise ValueError("--near-optimal-rel must be >= 0")
    if temperature <= 0:
        raise ValueError("--temperature must be > 0")

    best = min(rows, key=lambda row: row.avg_ppl)
    near_rows = [
        row
        for row in rows
        if (row.avg_ppl - best.avg_ppl) / best.avg_ppl <= near_optimal_rel
    ]
    if not near_rows:
        near_rows = [best]

    weights = np.array(
        [
            math.exp(-((row.avg_ppl - best.avg_ppl) / best.avg_ppl) / temperature)
            for row in rows
        ],
        dtype=float,
    )
    total_weight = float(weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError(
            f"Invalid soft-fit weights for L={key.max_length}, budget={key.bits_budget:g}"
        )

    topk_values = np.array([row.topk_ratio for row in rows], dtype=float)
    error_values = np.array([row.error_ratio for row in rows], dtype=float)
    fitted_topk = float(np.dot(weights, topk_values) / total_weight)
    fitted_error = float(np.dot(weights, error_values) / total_weight)
    fitted_svd = 1.0 - fitted_topk - fitted_error

    near_topk = [row.topk_ratio for row in near_rows]
    near_error = [row.error_ratio for row in near_rows]
    near_svd = [row.svd_ratio for row in near_rows]

    return PolicyFitPoint(
        max_length=key.max_length,
        stride=key.stride,
        bits_budget=key.bits_budget,
        best_topk_ratio=best.topk_ratio,
        best_error_ratio=best.error_ratio,
        best_svd_ratio=best.svd_ratio,
        best_avg_ppl=best.avg_ppl,
        fitted_topk_ratio=fitted_topk,
        fitted_error_ratio=fitted_error,
        fitted_svd_ratio=fitted_svd,
        near_min_topk_ratio=min(near_topk),
        near_max_topk_ratio=max(near_topk),
        near_min_error_ratio=min(near_error),
        near_max_error_ratio=max(near_error),
        near_min_svd_ratio=min(near_svd),
        near_max_svd_ratio=max(near_svd),
        near_count=len(near_rows),
        total_count=len(rows),
    )


def compute_policy_fit_points(
    groups: dict[SliceKey, list[BenchmarkRow]],
    near_optimal_rel: float,
    temperature: float,
) -> list[PolicyFitPoint]:
    fit_points = [
        compute_policy_fit_point(key, rows, near_optimal_rel, temperature)
        for key, rows in groups.items()
    ]
    return sorted(
        fit_points,
        key=lambda point: (point.max_length, point.stride, point.bits_budget),
    )


def values_for_share(
    points: list[PolicyFitPoint],
    share_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    budgets = np.array([point.bits_budget for point in points], dtype=float)
    if share_name == "svd":
        fitted = np.array([point.fitted_svd_ratio for point in points], dtype=float)
        best = np.array([point.best_svd_ratio for point in points], dtype=float)
        near_low = np.array([point.near_min_svd_ratio for point in points], dtype=float)
        near_high = np.array([point.near_max_svd_ratio for point in points], dtype=float)
    elif share_name == "topk":
        fitted = np.array([point.fitted_topk_ratio for point in points], dtype=float)
        best = np.array([point.best_topk_ratio for point in points], dtype=float)
        near_low = np.array([point.near_min_topk_ratio for point in points], dtype=float)
        near_high = np.array([point.near_max_topk_ratio for point in points], dtype=float)
    elif share_name == "error":
        fitted = np.array([point.fitted_error_ratio for point in points], dtype=float)
        best = np.array([point.best_error_ratio for point in points], dtype=float)
        near_low = np.array([point.near_min_error_ratio for point in points], dtype=float)
        near_high = np.array([point.near_max_error_ratio for point in points], dtype=float)
    else:
        raise ValueError(f"Unknown share name: {share_name}")

    return budgets, fitted, best, near_low, near_high


def render_sequence_heatmaps(
    output_dir: Path,
    rows_by_slice: dict[SliceKey, list[BenchmarkRow]],
    near_optimal_rel: float,
) -> list[Path]:
    output_paths: list[Path] = []
    max_lengths = sorted({key.max_length for key in rows_by_slice})

    for max_length in max_lengths:
        keys = sorted(
            [key for key in rows_by_slice if key.max_length == max_length],
            key=lambda key: (key.bits_budget, key.stride),
        )
        if not keys:
            continue

        ncols = min(4, len(keys))
        nrows = int(math.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.65 * ncols, 3.35 * nrows),
            dpi=220,
            squeeze=False,
            constrained_layout=True,
        )

        for ax, key in zip(axes.flat, keys, strict=False):
            slice_rows = rows_by_slice[key]
            topk_values, error_values, z = grid_matrix(slice_rows)
            vmin, vmax = finite_value_limits(
                z,
                context=(
                    f"L={key.max_length}, stride={key.stride}, "
                    f"budget={key.bits_budget:g}"
                ),
            )

            smooth_topk, smooth_error, smooth_z = interpolate_heatmap_surface(
                topk_values,
                error_values,
                z,
            )
            mesh = ax.pcolormesh(
                smooth_topk,
                smooth_error,
                np.ma.masked_invalid(smooth_z),
                cmap=OFFLINE_SEARCH_CMAP,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            add_missing_cell_overlay(ax, topk_values, error_values, z)

            best = min(slice_rows, key=lambda row: row.avg_ppl)
            near_rows = near_optimal_rows(slice_rows, best, near_optimal_rel)
            scatter_near_optimal(ax, near_rows, size=94, zorder=3)
            scatter_empirical_optimum(ax, best, size=190, zorder=4)

            ax.set_title(f"{key.bits_budget:g} bpw")
            ax.set_xlabel("Top-k/outlier ratio")
            ax.set_ylabel("Error-correction ratio")
            ax.set_xticks(topk_values)
            ax.set_yticks(error_values)
            ax.set_xticklabels([f"{value:.2f}" for value in topk_values], fontsize=7)
            ax.set_yticklabels([f"{value:.2f}" for value in error_values], fontsize=7)
            ax.set_xlim(cell_edges(topk_values)[0], cell_edges(topk_values)[-1])
            ax.set_ylim(cell_edges(error_values)[0], cell_edges(error_values)[-1])
            ax.set_aspect("equal")
            ax.grid(color="white", alpha=0.22, linewidth=0.7)
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.74, pad=0.015)
            cbar.set_label("Avg PPL", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        for ax in axes.flat[len(keys) :]:
            ax.set_visible(False)

        strides = sorted({key.stride for key in keys})
        stride_text = ", ".join(str(stride) for stride in strides)
        fig.suptitle(
            f"Offline search heatmaps: L={max_length}, stride={stride_text}",
            fontsize=13,
        )
        fig.legend(
            handles=offline_search_legend_handles(near_optimal_rel),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.065),
            ncol=3,
            frameon=True,
            fontsize=8,
        )

        output_path = output_dir / f"offline_search_heatmaps_L{max_length}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def render_fitted_heuristic_policy(
    output_path: Path,
    fit_points: list[PolicyFitPoint],
) -> None:
    max_lengths = sorted({point.max_length for point in fit_points})
    ncols = min(3, len(max_lengths))
    nrows = int(math.ceil(len(max_lengths) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.55 * ncols, 3.65 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
        dpi=220,
    )
    components = [
        ("svd", "SVD share"),
        ("topk", "Top-k/outlier share"),
        ("error", "Error-correction share"),
    ]

    for ax, max_length in zip(axes.flat, max_lengths, strict=False):
        points = sorted(
            [point for point in fit_points if point.max_length == max_length],
            key=lambda point: point.bits_budget,
        )
        for share_name, label in components:
            budgets, fitted, best, near_low, near_high = values_for_share(
                points, share_name
            )
            color = POLICY_COLORS[share_name]

            ax.fill_between(
                budgets,
                near_low,
                near_high,
                color=color,
                alpha=0.13,
                linewidth=0,
            )
            ax.plot(
                budgets,
                fitted,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=label,
            )
            ax.scatter(
                budgets,
                best,
                color=color,
                linewidth=1.1,
                marker="x",
                s=34,
                zorder=4,
        )

        ax.set_title(f"L={max_length}")
        ax.set_ylim(0.0, 1.02)
        ax.grid(color="#E6E6E6", linewidth=0.8)

    for ax in axes.flat[len(max_lengths) :]:
        ax.set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Allocation share")

    handles = [
        Line2D(
            [0],
            [0],
            color=POLICY_COLORS[share_name],
            linewidth=2.0,
            marker="o",
            label=label,
        )
        for share_name, label in components
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="x",
            linestyle="None",
            label="Exact optimum",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.145),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    fig.supxlabel("Bits per weight", y=-0.06)
    fig.suptitle("Fitted heuristic policy from near-optimal offline search", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def closest_row_to_fitted_policy(
    point: PolicyFitPoint,
    rows: list[BenchmarkRow],
) -> BenchmarkRow:
    return min(
        rows,
        key=lambda row: (
            (row.topk_ratio - point.fitted_topk_ratio) ** 2
            + (row.error_ratio - point.fitted_error_ratio) ** 2,
            row.avg_ppl,
            row.topk_ratio,
            row.error_ratio,
        ),
    )


def render_fitted_policy_ppl_by_budget(
    output_path: Path,
    fit_points: list[PolicyFitPoint],
    rows_by_slice: dict[SliceKey, list[BenchmarkRow]],
) -> None:
    max_lengths = sorted({point.max_length for point in fit_points})
    ncols = min(3, len(max_lengths))
    nrows = int(math.ceil(len(max_lengths) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.55 * ncols, 3.65 * nrows),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
        dpi=220,
    )

    for ax, max_length in zip(axes.flat, max_lengths, strict=False):
        points = sorted(
            [point for point in fit_points if point.max_length == max_length],
            key=lambda point: point.bits_budget,
        )
        budgets = np.array([point.bits_budget for point in points], dtype=float)
        exact_ppl = np.array([point.best_avg_ppl for point in points], dtype=float)
        fitted_ppl = []

        for point in points:
            key = SliceKey(point.max_length, point.stride, point.bits_budget)
            try:
                slice_rows = rows_by_slice[key]
            except KeyError as exc:
                raise KeyError(
                    "Missing benchmark rows for "
                    f"L={point.max_length}, stride={point.stride}, "
                    f"budget={point.bits_budget:g}"
                ) from exc
            fitted_ppl.append(closest_row_to_fitted_policy(point, slice_rows).avg_ppl)

        fitted_ppl_array = np.array(fitted_ppl, dtype=float)
        ax.plot(
            budgets,
            fitted_ppl_array,
            color="#B279A2",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Fitted near-optimal policy",
        )
        ax.plot(
            budgets,
            exact_ppl,
            color="#333333",
            linewidth=1.7,
            linestyle="--",
            marker="x",
            markersize=5,
            label="Exact optimum",
        )

        y_values = np.concatenate([fitted_ppl_array, exact_ppl])
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        y_pad = max((y_max - y_min) * 0.08, abs(y_max) * 0.005, 1e-3)
        ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
        ax.set_title(f"L={max_length}")
        ax.grid(color="#E6E6E6", linewidth=0.8)

    for ax in axes.flat[len(max_lengths) :]:
        ax.set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Average PPL")

    handles = [
        Line2D(
            [0],
            [0],
            color="#B279A2",
            linewidth=2.0,
            marker="o",
            label="Fitted near-optimal policy",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=1.7,
            linestyle="--",
            marker="x",
            label="Exact optimum",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.125),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    fig.supxlabel("Bits per weight", y=-0.045)
    fig.suptitle("PPL from fitted policy vs exact offline optimum", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def interpolate_policy_surface(
    fit_points: list[PolicyFitPoint],
    share_name: str,
    budget_grid: np.ndarray,
    seq_grid: np.ndarray,
) -> np.ndarray:
    max_lengths = sorted({point.max_length for point in fit_points})
    budgets_by_length: dict[int, np.ndarray] = {}
    values_by_length: dict[int, np.ndarray] = {}

    for max_length in max_lengths:
        points = sorted(
            [point for point in fit_points if point.max_length == max_length],
            key=lambda point: point.bits_budget,
        )
        budgets, fitted, _, _, _ = values_for_share(points, share_name)
        budgets_by_length[max_length] = budgets
        values_by_length[max_length] = fitted

    surface = np.empty_like(budget_grid, dtype=float)
    log_lengths = np.log2(np.array(max_lengths, dtype=float))

    for index in np.ndindex(budget_grid.shape):
        budget = float(budget_grid[index])
        seq_len = float(seq_grid[index])
        per_length_values = np.array(
            [
                np.interp(
                    budget,
                    budgets_by_length[max_length],
                    values_by_length[max_length],
                    left=values_by_length[max_length][0],
                    right=values_by_length[max_length][-1],
                )
                for max_length in max_lengths
            ],
            dtype=float,
        )

        if len(max_lengths) == 1:
            surface[index] = per_length_values[0]
        else:
            surface[index] = float(
                np.interp(
                    math.log2(seq_len),
                    log_lengths,
                    per_length_values,
                    left=per_length_values[0],
                    right=per_length_values[-1],
                )
            )

    return surface


def render_policy_surface_by_sequence(
    output_path: Path,
    fit_points: list[PolicyFitPoint],
) -> None:
    budgets = sorted({point.bits_budget for point in fit_points})
    max_lengths = sorted({point.max_length for point in fit_points})
    budget_axis = np.linspace(min(budgets), max(budgets), 120)
    seq_axis = np.geomspace(min(max_lengths), max(max_lengths), 100)
    budget_grid, seq_grid = np.meshgrid(budget_axis, seq_axis)

    panels = [
        ("svd", "SVD share", (0.0, 1.0)),
        ("topk", "Top-k/outlier share", None),
        ("error", "Error-correction share", None),
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.1),
        constrained_layout=True,
        dpi=220,
    )

    for ax, (share_name, title, fixed_lims) in zip(axes, panels, strict=True):
        surface = interpolate_policy_surface(
            fit_points, share_name, budget_grid, seq_grid
        )
        if fixed_lims is None:
            vmin = 0.0
            vmax = max(0.20, math.ceil((float(np.nanmax(surface)) + 0.02) * 20) / 20)
        else:
            vmin, vmax = fixed_lims
        image = ax.pcolormesh(
            budget_grid,
            seq_grid,
            surface,
            shading="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )

        for max_length in max_lengths:
            points = sorted(
                [point for point in fit_points if point.max_length == max_length],
                key=lambda point: point.bits_budget,
            )
            budgets_for_points, fitted, _, _, _ = values_for_share(points, share_name)
            ax.scatter(
                budgets_for_points,
                np.full_like(budgets_for_points, max_length, dtype=float),
                c=fitted,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                edgecolors="white",
                linewidths=0.6,
                s=34,
                zorder=3,
            )

        ax.set_title(title)
        ax.set_xlabel("Bits per weight")
        ax.set_ylabel("Sequence length")
        ax.set_yscale("log", base=2)
        ax.set_yticks(max_lengths)
        ax.set_yticklabels([str(max_length) for max_length in max_lengths])
        ax.grid(color="white", alpha=0.20, linewidth=0.7)
        fig.colorbar(image, ax=ax, shrink=0.88)

    fig.suptitle("Policy surface by sequence length", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_policy_fit_csv(
    output_path: Path,
    fit_points: list[PolicyFitPoint],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "max_length",
        "stride",
        "bits_budget",
        "best_topk_ratio",
        "best_error_ratio",
        "best_svd_ratio",
        "best_avg_ppl",
        "fitted_topk_ratio",
        "fitted_error_ratio",
        "fitted_svd_ratio",
        "near_min_topk_ratio",
        "near_max_topk_ratio",
        "near_min_error_ratio",
        "near_max_error_ratio",
        "near_min_svd_ratio",
        "near_max_svd_ratio",
        "near_count",
        "total_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in fit_points:
            writer.writerow(
                {
                    "max_length": point.max_length,
                    "stride": point.stride,
                    "bits_budget": f"{point.bits_budget:.12g}",
                    "best_topk_ratio": f"{point.best_topk_ratio:.12g}",
                    "best_error_ratio": f"{point.best_error_ratio:.12g}",
                    "best_svd_ratio": f"{point.best_svd_ratio:.12g}",
                    "best_avg_ppl": f"{point.best_avg_ppl:.12g}",
                    "fitted_topk_ratio": f"{point.fitted_topk_ratio:.12g}",
                    "fitted_error_ratio": f"{point.fitted_error_ratio:.12g}",
                    "fitted_svd_ratio": f"{point.fitted_svd_ratio:.12g}",
                    "near_min_topk_ratio": f"{point.near_min_topk_ratio:.12g}",
                    "near_max_topk_ratio": f"{point.near_max_topk_ratio:.12g}",
                    "near_min_error_ratio": f"{point.near_min_error_ratio:.12g}",
                    "near_max_error_ratio": f"{point.near_max_error_ratio:.12g}",
                    "near_min_svd_ratio": f"{point.near_min_svd_ratio:.12g}",
                    "near_max_svd_ratio": f"{point.near_max_svd_ratio:.12g}",
                    "near_count": point.near_count,
                    "total_count": point.total_count,
                }
            )


def run_all_mode(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or args.benchmark_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all_rows(args.benchmark_dir)
    rows_by_slice = group_by_slice(rows)
    complete_count = 0
    for key, slice_rows in sorted(
        rows_by_slice.items(),
        key=lambda item: (
            item[0].max_length,
            item[0].stride,
            item[0].bits_budget,
        ),
    ):
        is_complete = validate_grid(
            slice_rows,
            allow_incomplete=args.allow_incomplete,
            context=(
                f"L={key.max_length}, stride={key.stride}, "
                f"budget={key.bits_budget:g}"
            ),
        )
        if is_complete:
            complete_count += 1

    fit_points = compute_policy_fit_points(
        rows_by_slice,
        near_optimal_rel=args.near_optimal_rel,
        temperature=args.temperature,
    )

    written_paths = []
    written_paths.extend(
        render_sequence_heatmaps(
            output_dir=output_dir,
            rows_by_slice=rows_by_slice,
            near_optimal_rel=args.near_optimal_rel,
        )
    )
    fitted_policy_path = output_dir / "fitted_heuristic_policy.png"
    render_fitted_heuristic_policy(fitted_policy_path, fit_points)
    written_paths.append(fitted_policy_path)

    fitted_ppl_path = output_dir / "fitted_policy_ppl_by_budget.png"
    render_fitted_policy_ppl_by_budget(
        fitted_ppl_path,
        fit_points,
        rows_by_slice,
    )
    written_paths.append(fitted_ppl_path)

    surface_path = output_dir / "policy_surface_by_sequence.png"
    render_policy_surface_by_sequence(surface_path, fit_points)
    written_paths.append(surface_path)

    csv_path = output_dir / "heuristic_policy_fit_points.csv"
    write_policy_fit_csv(csv_path, fit_points)
    written_paths.append(csv_path)

    max_lengths = sorted({row.max_length for row in rows})
    budgets = sorted({row.bits_budget for row in rows})
    print(
        f"Loaded {len(rows)} runs across {len(rows_by_slice)} slices "
        f"({complete_count} complete)."
    )
    print(
        "Sequence lengths: "
        + ", ".join(str(max_length) for max_length in max_lengths)
    )
    print(
        "Bits per weight budgets: "
        + ", ".join(f"{budget:g}" for budget in budgets)
    )
    print(
        f"Near-optimal threshold: {args.near_optimal_rel:g}; "
        f"soft-fit temperature: {args.temperature:g}"
    )
    for path in written_paths:
        print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    run_all_mode(args)


if __name__ == "__main__":
    main()
