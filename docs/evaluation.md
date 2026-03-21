# Evaluation Pipeline

This document explains what `src/eval_ppl.py` does, and how `src/eval_ppl_residual_path_for_qwen3.py` differs.

## What `eval_ppl.py` Is

`src/eval_ppl.py` is the main evaluation pipeline for:

- WikiText2 perplexity evaluation
- simulated inter-node activation traffic accounting

It evaluates a model with optional activation compression at layer-boundary transitions.

## How `eval_ppl.py` Works

1. Load tokenizer/model

- Loads from `--model_dir` (downloads from `--model_name` only if local path is missing)
- Supports `fp32/fp16/bf16` plus optional bitsandbytes loading (`load_in_8bit`, `load_in_4bit`)

2. Build a default node plan

- Splits model layers into a 4-node plan (`node0..node3`) using `make_default_plan(...)`
- Finds inter-node boundaries with `find_boundaries(...)`

3. Install boundary hooks

- At each boundary module output, run:
  - `compress(hidden_states)`
  - update traffic meter with `Payload.nbytes`
  - `decompress(...)` for continued forward pass

4. Compute PPL on WikiText2

- Uses sliding windows over test text (`max_length`, `stride`)
- Handles batched windows (`batch_windows`)
- Computes NLL manually from shifted logits/labels

5. Save result JSON

- Includes args, PPL metrics, timing, total traffic, and per-link traffic

Typical JSON fields:

- `results.avg_ppl`
- `traffic_totals.bytes_per_token`
- `traffic_per_link` (per boundary key)

## CLI Example (`eval_ppl.py`)

```bash
python -u src/eval_ppl.py \
  --model_name "Qwen/Qwen3-8B" \
  --model_dir "/mnt/ssd/liaw/Qwen/Qwen3-8B" \
  --dtype bf16 \
  --max_length 2048 \
  --stride 512 \
  --batch_windows 2 \
  --result_dir "results/manual_eval"
```

Note: from CLI, built-in compressor names are currently minimal (`none`). Most custom compressors are passed as Python objects from scripts such as `example/*.py` and `final_benchmark.py`.

## What `eval_ppl_residual_path_for_qwen3.py` Changes

`src/eval_ppl_residual_path_for_qwen3.py` is a Qwen-specific variant for split-path compression experiments.

Main differences versus `eval_ppl.py`:

1. Qwen residual patching

- Monkey-patches Qwen decoder layer forward logic to replace one residual `+` with a module `ResidualAdd`
- This enables attaching hooks directly on the residual merge path

2. Two compressor paths

- Uses separate compressors:
  - `norm_compressor`
  - `residual_compressor`
- Keeps `--compressor` as legacy shorthand (applies to both unless overridden)

3. Different hook locations

- Installs pre-hooks on the next layer's:
  - `input_layernorm` input (norm path)
  - `attn_residual_add` input (residual path)
- This is different from output hooks in `eval_ppl.py`

4. Traffic accounting nuance

- Residual path hook records bytes with `ntokens=0` to avoid duplicate token counting.
- Total bytes still include residual traffic, but bytes-per-token denominator is controlled by non-duplicated token accounting.

## CLI Example (`eval_ppl_residual_path_for_qwen3.py`)

```bash
python -u src/eval_ppl_residual_path_for_qwen3.py \
  --model_name "Qwen/Qwen3-8B" \
  --model_dir "/mnt/ssd/liaw/Qwen/Qwen3-8B" \
  --compressor none \
  --result_dir "results/manual_eval_residual"
```

## Which File To Use

- Use `src/eval_ppl.py` for standard boundary compression experiments.
- Use `src/eval_ppl_residual_path_for_qwen3.py` when you specifically need Qwen residual-path vs norm-path split compression behavior.
