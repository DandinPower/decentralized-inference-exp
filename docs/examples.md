# Examples: How To Run

All example scripts are in `example/` and call `run_ppl_eval(...)` to evaluate WikiText2 PPL plus traffic metrics.

Run from repo root:

```bash
source .venv/bin/activate
```

Common optional flags for all examples:

- `--dtype {fp32,fp16,bf16}`
- `--first-k-tokens <int>` (0 means full sequence)
- `--batch-windows <int>` (how many sliding windows per forward pass)

## 1) BitSqueeze Example

Script: `example/bitsqz_example.py`

```bash
python -u example/bitsqz_example.py \
  --model-name "Qwen/Qwen3.5-0.8B" \
  --model-dir "/mnt/ssd/liaw/Qwen/Qwen3.5-0.8B" \
  --result-name "Qwen/Qwen3.5-0.8B"
```

What it runs by default:

- `BitSqueezeCompressor(method="FP16")`
- `OutlierSeparationBitSqueezeCompressor(outlier_ratio=0.001, method="FP16")`

Output directories:

- `results/<result-name>/bitsqz/bitsqz_FP16/`
- `results/<result-name>/bitsqz/bitsqz_outlier_0.001_FP16/`

## 2) Top-k Sparse Example

Script: `example/topk_example.py`

```bash
python -u example/topk_example.py \
  --model-name "Qwen/Qwen3.5-0.8B" \
  --model-dir "/mnt/ssd/liaw/Qwen/Qwen3.5-0.8B" \
  --result-name "Qwen/Qwen3.5-0.8B"
```

What it runs by default:

- `TopkCompressor(outlier_ratio=0.01)`
- `TopkCompressor(outlier_ratio=0.02)`

Output directories:

- `results/<result-name>/topk/topk_outlier_0.01/`
- `results/<result-name>/topk/topk_outlier_0.02/`

## 3) OnlineSVD Example

Script: `example/onlinesvd_example.py`

```bash
python -u example/onlinesvd_example.py \
  --model-name "Qwen/Qwen3.5-0.8B" \
  --model-dir "/mnt/ssd/liaw/Qwen/Qwen3.5-0.8B" \
  --result-name "Qwen/Qwen3.5-0.8B"
```

What it runs by default:

- plain OnlineSVD (`rank=512`, `mode="trunc_approx"`)
- OnlineSVD with outlier separation (`outlier_ratio=0.001`)
- OnlineSVD + BitSqueeze (`fmt_uv="q8_0"`, `fmt_s="fp32"`)

Output directory:

- `results/<result-name>/onlinesvd/`

## 4) TorchAO FP8 Example

Script: `example/torchao_fp8_example.py`

```bash
python -u example/torchao_fp8_example.py \
  --model-name "Qwen/Qwen3.5-0.8B" \
  --model-dir "/mnt/ssd/liaw/Qwen/Qwen3.5-0.8B" \
  --result-name "Qwen/Qwen3.5-0.8B"
```

What it runs by default:

- `TorchAO_Float8Tensor_Compressor()`

Output directory:

- `results/<result-name>/torchao_fp8/`

## Result File Format

Each run produces a timestamped JSON file:

- `run_YYYYMMDD_HHMMSS.json`

and includes:

- experiment args (`model`, `dtype`, compressor info)
- quality metrics (`avg_ppl`, `total_nll`)
- traffic metrics (`total_bytes`, `bytes_per_token`, per-link breakdown)
