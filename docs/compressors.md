# Compressors: Meaning And Configs

This project defines compressors that transform activation tensors before simulated transmission across node boundaries.

All compressors follow the `Compressor` interface in `src/compressor.py`:

- `compress(x) -> Payload(data, meta, nbytes)`
- `decompress(payload, device, dtype) -> reconstructed_tensor`

`nbytes` is the simulated communication cost used by traffic metrics.

## Baseline Compressor

### `NoneCompressor`

File: `src/compressor.py`

- Purpose: no compression baseline
- Behavior: returns input tensor as payload, cost = raw tensor size

## BitSqueeze Family

File: `src/compressors/bitsqz_compressor.py`

### `BitSqueezeCompressor(method, per_feature=False)`

- Purpose: quantize full activation tensor with BitSqueeze
- Key configs:
  - `method`: quantization format string (for example `FP16`, `BF16`, `Q8_0`, `FP8`, `Q4_0`, `NF4`, `NF4_DQ`, `Q2_K`)
  - `per_feature`: if `True`, transposes tensor to compress feature-wise blocks
- Behavior:
  - cast activation to CPU fp32
  - optionally transpose for per-feature path
  - compress with `bitsqueeze.BitSqueezeBuffer.compress`

### `OutlierSeparationBitSqueezeCompressor(outlier_ratio, method, per_feature=False)`

- Purpose: preserve top-k outlier values sparsely, compress residual densely with BitSqueeze
- Key configs:
  - `outlier_ratio`: fraction of features kept per token row as sparse outliers
  - `method`: BitSqueeze format for residual
  - `per_feature`: same meaning as above
- Behavior:
  - split tensor into sparse top-k and dense residual
  - BitSqueeze the residual
  - total `nbytes` = compressed residual + sparse outlier overhead

## Top-k Sparse Compressor

File: `src/compressors/topk_compressor.py`

### `TopkCompressor(outlier_ratio)`

- Purpose: top-k only sparse transmission baseline
- Key config:
  - `outlier_ratio`: keep this fraction of largest-magnitude feature values per token row
- Behavior:
  - keeps sparse top-k activations
  - drops dense residual entirely
  - decompress reconstructs only sparse part

This is a strong compression baseline but typically has larger quality loss than BitSqueeze/OnlineSVD methods.

## OnlineSVD Family

File: `src/compressors/onlinesvd.py`

### `OnlineSVDCompressor(...)`

Main low-rank compressor with optional outlier path, optional error correction path, and optional quantization of SVD factors.

Important configs:

- `rank`:
  - target low-rank dimension
- `mode`:
  - `full`: full SVD
  - `trunc_slice`: full SVD then truncate to `rank`
  - `trunc_approx`: approximate low-rank SVD via `torch.svd_lowrank`
- `outlier_ratio`:
  - top-k sparse separation before SVD (`0.0` disables)
- `svd_error_correction_ratio`:
  - top-k sparse correction from SVD reconstruction error (`0.0` disables)
- `fmt_uv` / `fmt_s`:
  - quantization format for `U,Vh` and singular values `S`
  - `fp32` means no BitSqueeze on that factor
- `niter`, `q_oversample`:
  - only used by `trunc_approx`
- `svd_device`:
  - `auto`, `cpu`, or `cuda`

Optional residual stabilization configs:

- scaling path:
  - `residual_scale_alpha`, `residual_scale_eps`, `residual_scale_q`, `scale_factor_format`
- centering path:
  - `residual_center` (`none` or `center`), `center_factor_format`

Note: residual scaling and residual centering are mutually exclusive by design.

### `OutlierSeparationOnlineSVDCompressor(...)`

- Subclass convenience wrapper around `OnlineSVDCompressor`
- Same behavior, but constructor emphasizes outlier-separation use

### `OnlineSVDBitSqueezeCompressor(...)`

- Subclass convenience wrapper for OnlineSVD + BitSqueeze factors
- Same base behavior, constructor emphasizes `fmt_uv`/`fmt_s` usage

## TorchAO FP8 Compressor

File: `src/compressors/torchao_fp8_compressor.py`

### `TorchAO_Float8Tensor_Compressor()`

- Purpose: use TorchAO float8 tensor representation for payload
- Behavior:
  - `compress`: converts high-precision tensor to `Float8Tensor`
  - `decompress`: dequantizes back to requested dtype/device

## Format Names Used In OnlineSVD

`src/compressors/onlinesvd.py` maps these lowercase names to BitSqueeze methods:

- `fp32`, `fp16`, `bf16`, `q8_0`, `mxfp8`, `fp8`, `q4_0`, `nf4`, `mxfp4`, `nf4_dq`, `q2_k`

When setting configs from scripts, use names exactly as expected by the constructor in that file.
