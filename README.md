# Decentralized Inference Experiments

This repository contains experiments for activation compression in decentralized LLM inference.
The core workflow evaluates two things at the same time:

- language quality through WikiText2 perplexity (PPL)
- communication cost through simulated traffic metrics (bytes, transfers, bytes per token)

## Installation

The following setup steps are preserved from the original README:

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install huggingface-hub numpy packaging psutil pyyaml safetensors transformers datasets accelerate bitsandbytes torchao

git clone https://github.com/DandinPower/PyBitSqueeze.git
cd PyBitSqueeze
uv pip install -r requirements.txt
uv pip install . --no-build-isolation
cd ..
```

## Project Layout

- `src/`: evaluation pipeline and compressor implementations
- `example/`: runnable examples for each compressor family
- `scripts/`: pre-built visualization scripts for Pareto plots
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

Results are written under `results/<result-name>/.../run_YYYYMMDD_HHMMSS.json`.

## Documentation

- `docs/README.md`: documentation index
- `docs/examples.md`: how to run example scripts
- `docs/compressors.md`: compressor definitions and configuration options
- `docs/scripts.md`: pre-built visualization scripts
- `docs/evaluation.md`: `eval_ppl` pipeline and Qwen3 residual-path variant differences
