from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "svd": "#4C78A8",
    "topk": "#F58518",
    "error": "#54A24B",
    "dark": "#2F2F2F",
    "accent": "#B279A2",
}


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def optimal_ratios(bits_per_weight: np.ndarray | float, seq_len: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mock fitted policy: low budgets favor SVD, extra budget gradually funds top-k and error repair."""
    b = np.asarray(bits_per_weight, dtype=float)
    s = np.asarray(seq_len, dtype=float)
    seq_shift = np.log2(s / 4096.0)

    topk = 0.025 + 0.125 * sigmoid((b - 1.25) / 0.42) * (1.0 + 0.06 * seq_shift)
    error = 0.018 + 0.095 * sigmoid((b - 1.65) / 0.38) * (1.0 + 0.08 * seq_shift)

    topk = np.clip(topk, 0.015, 0.18)
    error = np.clip(error, 0.012, 0.14)
    svd = np.clip(1.0 - topk - error, 0.68, 0.98)
    total = svd + topk + error
    return svd / total, topk / total, error / total


def perplexity_degradation(topk_ratio: np.ndarray, error_ratio: np.ndarray, bits_per_weight: float, seq_len: int) -> np.ndarray:
    """Synthetic objective surface. Lower is better."""
    _, topk_star, error_star = optimal_ratios(bits_per_weight, seq_len)
    top_delta = topk_ratio - topk_star
    err_delta = error_ratio - error_star

    base = 0.18 + 1.55 * np.exp(-1.05 * bits_per_weight) + 0.10 * np.log2(4096 / seq_len)
    curvature = 26.0 / (bits_per_weight + 0.35)
    ridge = curvature * (2.2 * top_delta**2 + 3.1 * err_delta**2)
    interaction = 19.0 * top_delta * err_delta + 7.5 * (top_delta + err_delta) ** 2
    svd_star, _, _ = optimal_ratios(bits_per_weight, seq_len)
    svd_ratio = 1.0 - topk_ratio - error_ratio
    star_svd_penalty = 2.8 * np.maximum(svd_star - svd_ratio, 0.0) ** 2
    low_correction_penalty = 0.45 * np.maximum(topk_star - topk_ratio, 0.0) * np.maximum(error_star - error_ratio, 0.0)
    infeasible = np.where(svd_ratio < 0.55, np.nan, 0.0)
    return base + ridge + interaction + star_svd_penalty + low_correction_penalty + infeasible


def search_optimum(bits_per_weight: float, seq_len: int, grid_size: int = 81) -> tuple[float, float, float]:
    ratios = np.linspace(0.0, 0.30, grid_size)
    top_grid, err_grid = np.meshgrid(ratios, ratios)
    surface = perplexity_degradation(top_grid, err_grid, bits_per_weight, seq_len)
    min_idx = np.nanargmin(surface)
    row, col = np.unravel_index(min_idx, surface.shape)
    return top_grid[row, col], err_grid[row, col], surface[row, col]


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def plot_problem_formulation() -> None:
    budgets = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
    svd, topk, error = optimal_ratios(budgets, 4096)

    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    width = 0.16
    ax.bar(budgets, svd, width=width, color=COLORS["svd"], label="SVD rank budget")
    ax.bar(budgets, topk, bottom=svd, width=width, color=COLORS["topk"], label="Top-k/outlier budget")
    ax.bar(budgets, error, bottom=svd + topk, width=width, color=COLORS["error"], label="Error-correction budget")

    ax.set_title("Problem formulation: allocate one controlled activation-compression budget", pad=12)
    ax.set_xlabel("Activation budget (bits per weight)")
    ax.set_ylabel("Share of total budget")
    ax.set_ylim(0, 1.06)
    ax.set_xticks(budgets)
    ax.grid(axis="y", color="#E6E6E6")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.annotate(
        "Low budgets spend nearly everything on rank information",
        xy=(0.75, svd[1] * 0.55),
        xytext=(1.35, 0.52),
        arrowprops={"arrowstyle": "->", "color": COLORS["dark"], "lw": 1.1},
        fontsize=9,
        color=COLORS["dark"],
    )
    ax.annotate(
        "Extra budget can repair outliers and residual error",
        xy=(2.5, svd[-2] + topk[-2] + error[-2] * 0.55),
        xytext=(1.75, 0.93),
        arrowprops={"arrowstyle": "->", "color": COLORS["dark"], "lw": 1.1},
        fontsize=9,
        color=COLORS["dark"],
    )
    save(fig, "allocation_problem_formulation.png")


def plot_offline_search_heatmaps() -> None:
    budgets = [0.75, 1.50, 2.50]
    seq_len = 4096
    ratios = np.linspace(0.0, 0.30, 90)
    top_grid, err_grid = np.meshgrid(ratios, ratios)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    vmin, vmax = 0.22, 1.20
    for ax, budget in zip(axes, budgets, strict=True):
        z = perplexity_degradation(top_grid, err_grid, budget, seq_len)
        top_star, err_star, _ = search_optimum(budget, seq_len)
        image = ax.imshow(
            z,
            origin="lower",
            extent=[ratios.min(), ratios.max(), ratios.min(), ratios.max()],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.scatter([top_star], [err_star], s=70, color="white", edgecolor="black", linewidth=1.0, zorder=3)
        ax.set_title(f"Budget = {budget:.2f} bpw")
        ax.set_xlabel("Top-k/outlier ratio")
        ax.set_ylabel("Error-correction ratio")
        ax.set_xlim(0, 0.30)
        ax.set_ylim(0, 0.30)
        ax.grid(color="white", alpha=0.16)

    cbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    cbar.set_label("Perplexity degradation (mock; lower is better)")
    fig.suptitle("Offline search: sweep allocation ratios and mark the empirical optimum", fontsize=13)
    save(fig, "offline_search_heatmaps.png")


def plot_interaction_residual() -> None:
    budget = 1.5
    seq_len = 4096
    ratios = np.linspace(0.0, 0.30, 90)
    top_grid, err_grid = np.meshgrid(ratios, ratios)
    observed = perplexity_degradation(top_grid, err_grid, budget, seq_len)
    top_star, err_star, _ = search_optimum(budget, seq_len)

    top_only = perplexity_degradation(top_grid, np.full_like(err_grid, err_star), budget, seq_len)
    err_only = perplexity_degradation(np.full_like(top_grid, top_star), err_grid, budget, seq_len)
    anchor = perplexity_degradation(np.array([[top_star]]), np.array([[err_star]]), budget, seq_len)[0, 0]
    additive = top_only + err_only - anchor
    residual = observed - additive

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    panels = [
        (observed, "Observed surface", "viridis", None),
        (additive, "Additive approximation", "viridis", None),
        (residual, "Interaction residual", "coolwarm", (-0.07, 0.07)),
    ]
    for ax, (z, title, cmap, lims) in zip(axes, panels, strict=True):
        if lims:
            vmin, vmax = lims
        else:
            vmin, vmax = 0.24, 1.05
        image = ax.imshow(
            z,
            origin="lower",
            extent=[ratios.min(), ratios.max(), ratios.min(), ratios.max()],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.scatter([top_star], [err_star], s=62, color="white", edgecolor="black", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Top-k/outlier ratio")
        ax.set_ylabel("Error-correction ratio")
        fig.colorbar(image, ax=ax, shrink=0.88)
    fig.suptitle("Why a grid sweep is justified: components interact instead of adding independently", fontsize=13)
    save(fig, "interaction_surface_residual.png")


def plot_fitted_heuristic_policy() -> None:
    budgets = np.arange(0.5, 3.01, 0.25)
    seq_lens = [2048, 4096, 8192]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), sharex=True, constrained_layout=True)
    line_styles = {2048: "-", 4096: "--", 8192: "-."}
    for seq_len in seq_lens:
        svd, topk, error = optimal_ratios(budgets, seq_len)
        axes[0].plot(budgets, svd, marker="o", ms=3.8, lw=1.8, ls=line_styles[seq_len], label=f"L={seq_len}")
        axes[1].plot(budgets, topk, marker="o", ms=3.8, lw=1.8, ls=line_styles[seq_len], label=f"L={seq_len}")
        axes[2].plot(budgets, error, marker="o", ms=3.8, lw=1.8, ls=line_styles[seq_len], label=f"L={seq_len}")

    for ax, title, color in zip(
        axes,
        ["SVD rank share", "Top-k/outlier share", "Error-correction share"],
        [COLORS["svd"], COLORS["topk"], COLORS["error"]],
        strict=True,
    ):
        for line in ax.lines:
            line.set_color(color)
        ax.set_title(title)
        ax.set_xlabel("Bits per weight")
        ax.grid(color="#E6E6E6")
        ax.set_ylim(0, 1.0 if "SVD" in title else 0.20)
    axes[0].set_ylabel("Fitted budget share")
    axes[2].legend(loc="upper left", frameon=False)
    fig.suptitle("Fitted heuristic: deployment policy from offline optima", fontsize=13)
    save(fig, "fitted_heuristic_policy.png")


def plot_policy_surfaces() -> None:
    budgets = np.linspace(0.5, 3.0, 80)
    seq_lens = np.array([1024, 2048, 4096, 8192, 16384])
    budget_grid, seq_grid = np.meshgrid(budgets, seq_lens)
    svd, topk, error = optimal_ratios(budget_grid, seq_grid)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    panels = [
        (svd, "SVD rank share", COLORS["svd"], (0.70, 0.96)),
        (topk, "Top-k/outlier share", COLORS["topk"], (0.02, 0.18)),
        (error, "Error-correction share", COLORS["error"], (0.01, 0.14)),
    ]
    for ax, (z, title, _, lims) in zip(axes, panels, strict=True):
        image = ax.imshow(
            z,
            origin="lower",
            extent=[budgets.min(), budgets.max(), seq_lens.min(), seq_lens.max()],
            aspect="auto",
            cmap="magma",
            vmin=lims[0],
            vmax=lims[1],
        )
        ax.set_title(title)
        ax.set_xlabel("Bits per weight")
        ax.set_ylabel("Sequence length")
        ax.set_yticks(seq_lens)
        fig.colorbar(image, ax=ax, shrink=0.88)
    fig.suptitle("Heuristic as a small function of budget and sequence length", fontsize=13)
    save(fig, "policy_surface_by_sequence.png")


def plot_bpw_latency_proxy() -> None:
    budgets = np.linspace(0.5, 3.0, 12)
    seq_len = 4096
    payload = budgets * seq_len
    networks = {
        "Constrained link": {"rtt": 40, "bandwidth": 230},
        "Median link": {"rtt": 20, "bandwidth": 430},
        "Fast link": {"rtt": 10, "bandwidth": 760},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), constrained_layout=True)
    for label, cfg in networks.items():
        svd, topk, error = optimal_ratios(budgets, seq_len)
        comm = cfg["rtt"] + payload / cfg["bandwidth"]
        svd_overhead = 2.0 + 2.4 * budgets**1.15 * svd
        other_overhead = 0.8 + 1.0 * (topk + error)
        codec_time = comm + svd_overhead + other_overhead
        corr = np.corrcoef(budgets, codec_time)[0, 1]
        axes[0].plot(budgets, codec_time, marker="o", lw=1.9, label=f"{label} (r={corr:.2f})")

    axes[0].set_title("Bits per weight is a controlled proxy for codec latency")
    axes[0].set_xlabel("Bits per weight")
    axes[0].set_ylabel("Mock codec time (ms)")
    axes[0].grid(color="#E6E6E6")
    axes[0].legend(frameon=False, fontsize=8)

    budget = 1.5
    svd, topk, error = optimal_ratios(budget, seq_len)
    categories = ["Communication", "SVD codec", "Top-k + error codec"]
    values = np.array([payload[4] / networks["Median link"]["bandwidth"] + 20, 2.0 + 2.4 * budget**1.15 * svd, 0.8 + 1.0 * (topk + error)])
    axes[1].bar(categories, values, color=["#72B7B2", COLORS["svd"], "#E45756"])
    axes[1].set_title("Search uses bpw; final experiments report end-to-end time")
    axes[1].set_ylabel("Mock latency component (ms)")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", color="#E6E6E6")
    for i, value in enumerate(values):
        axes[1].text(i, value + 0.8, f"{float(value):.1f}", ha="center", fontsize=9)
    save(fig, "bpw_latency_proxy.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    plot_problem_formulation()
    plot_offline_search_heatmaps()
    plot_interaction_residual()
    plot_fitted_heuristic_policy()
    plot_policy_surfaces()
    plot_bpw_latency_proxy()


if __name__ == "__main__":
    main()
