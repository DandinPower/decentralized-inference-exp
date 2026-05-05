# Decentralized Inference Experiments

This repository contains experiments for activation compression in decentralized LLM inference.
The core workflow evaluates two things at the same time:

- language quality through WikiText2 perplexity (PPL)
- communication cost through simulated traffic metrics (bytes, transfers, bytes per token)

## Installation

The following setup steps are preserved from the original README:

```bash
sudo apt install python3-dev
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install huggingface-hub numpy packaging psutil pyyaml safetensors transformers datasets accelerate bitsandbytes torchao matplotlib

git clone https://github.com/DandinPower/PyBitSqueeze.git
cd PyBitSqueeze
uv pip install -r requirements.txt
uv pip install . --no-build-isolation
cd ..

git clone https://github.com/DandinPower/PyBitSqueeze-LLM.git
cd PyBitSqueeze-LLM
uv pip install -r requirements.txt
uv pip install . --no-build-isolation
cd ..
```

## Project Layout

- `src/`: evaluation pipeline and compressor implementations
- `example/`: runnable examples for each compressor family
- `scripts/`: pre-built visualization scripts for Pareto plots
- `scripts/plot_final_benchmark_pareto.py`: aggregate latest final benchmark runs and render family-wise Pareto frontiers
- `final_benchmark.py`: main benchmark sweep entry point
- `final_benchmark.sh`: convenient benchmark launcher with model presets
- `docs/`: detailed project documentation
- `results/`: generated JSON results, CSV summaries, and plot artifacts

## Quick Start

Run the benchmark wrapper:

```bash
source .venv/bin/activate
bash final_benchmark.sh
```

Or run the Python benchmark directly:

```bash
source .venv/bin/activate
python -u final_benchmark.py \
  --model-name "Qwen/Qwen3.5-0.8B" \
  --model-dir "/mnt/ssd/liaw/Qwen/Qwen3.5-0.8B" \
  --result-name "Qwen/Qwen3.5-0.8B"
```

## Benchmark Entry Points

- `final_benchmark.py` runs four experiment groups:
  - BitSqueeze only
  - BitSqueeze + outlier separation (top-k)
  - OnlineSVD + BitSqueeze + outlier separation
  - OnlineSVD error-correction sweeps
- `final_benchmark.sh` defines model presets and launches `final_benchmark.py` with those values.
- `scripts/plot_final_benchmark_pareto.py` reads the newest `run_*.json` from each approach under those four groups, exports a consolidated CSV, and saves Pareto plots.

Results are written under `results/<result-name>/.../run_YYYYMMDD_HHMMSS.json`.

### Plot Final Benchmark Pareto

After benchmark runs are available, generate a consolidated Pareto view:

```bash
python -u scripts/plot_final_benchmark_pareto.py \
  --results-dir results/Qwen/Qwen3.5-9B
```

By default this writes:

- `results/<result-name>/pareto_final_benchmark_data.csv`
- six figure variants under `results/<result-name>/` (label by approach name / avg_ppl / bytes_per_token, each with full-scatter and frontier-only versions)

Useful option:

- `--avg-ppl-range "7.6~8.2"` to constrain analysis/plotting to a PPL window

## Documentation

- `docs/README.md`: documentation index
- `docs/examples.md`: how to run example scripts
- `docs/compressors.md`: compressor definitions and configuration options
- `docs/scripts.md`: pre-built visualization scripts
- `docs/evaluation.md`: `eval_ppl` pipeline and Qwen3 residual-path variant differences
