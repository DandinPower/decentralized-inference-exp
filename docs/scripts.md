# Scripts: Pre-Built Visualizations

This project includes plotting scripts in `scripts/` to visualize PPL vs communication trade-offs from CSV summaries.

All scripts are offline analytics: they read CSV files and save figures. They do not run model inference.

## Input CSV Expectations

The plotting scripts expect columns like:

- `mode`
- `rank`
- `uv_format`
- `s_format`
- `avg_ppl`
- `bytes_per_token`
- `eval_time_s`
- `status`

Default CSV path used by scripts:

- `results/onlinesvd_bitsqueeze/onlinesvd_bitsqueeze_ppl_20260226_221024.csv`

## 1) Rank Comparison Pareto

Script: `scripts/plot_onlinesvd_pareto.py`

Purpose:

- compare Pareto frontiers for multiple ranks (default: 512 vs 1024)
- fix one `mode` and one `s_format`
- x-axis: `bytes_per_token`, y-axis: `avg_ppl`

Example:

```bash
python -u scripts/plot_onlinesvd_pareto.py \
  --csv results/onlinesvd_bitsqueeze/onlinesvd_bitsqueeze_ppl_20260226_221024.csv \
  --mode trunc_slice \
  --s-format fp32 \
  --ranks 512 1024 \
  --output results/onlinesvd_bitsqueeze/pareto_rank_512_vs_1024.png
```

Useful option:

- `--avg-ppl-range "8.2~10"` to remove outlier points from analysis/plotting

## 2) s_format Comparison Pareto

Script: `scripts/plot_onlinesvd_sformat_pareto.py`

Purpose:

- compare Pareto frontiers for different singular-value formats (`s_format`)
- fix one `mode` and one `rank`

Example:

```bash
python -u scripts/plot_onlinesvd_sformat_pareto.py \
  --csv results/onlinesvd_bitsqueeze/onlinesvd_bitsqueeze_ppl_20260226_221024.csv \
  --mode trunc_slice \
  --rank 512 \
  --s-formats fp32 fp16 fp8 \
  --output results/onlinesvd_bitsqueeze/pareto_sformat_comparison.png
```

## 3) Mode Comparison Pareto

Script: `scripts/plot_onlinesvd_mode_pareto_time.py`

Purpose:

- compare Pareto frontiers across experiment mode/family labels
- fix one `rank` and one `s_format`
- default compares `trunc_approx` and `trunc_slice`

Example:

```bash
python -u scripts/plot_onlinesvd_mode_pareto_time.py \
  --csv results/onlinesvd_bitsqueeze/onlinesvd_bitsqueeze_ppl_20260226_221024.csv \
  --rank 512 \
  --s-format fp32 \
  --modes trunc_approx trunc_slice \
  --output results/onlinesvd_bitsqueeze/pareto_mode_comparison.png
```

Even though the script also loads `eval_time_s`, the plotted frontier in this script is still `bytes_per_token` vs `avg_ppl`.

## Output

All scripts save the figure to `--output` and print:

- output path
- filtered point counts
- number of Pareto points retained per series
