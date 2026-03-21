import argparse
import gc

import torch

from src.eval_ppl import run_ppl_eval
from src.compressors.onlinesvd import (
    OutlierSeparationOnlineSVDCompressor,
)
from src.compressors.bitsqz_compressor import (
    BitSqueezeCompressor,
    OutlierSeparationBitSqueezeCompressor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-dir", type=str, default="/mnt/ssd/liaw/Qwen/Qwen3.5-0.8B")
    parser.add_argument("--result-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--first-k-tokens", type=int, default=0)
    parser.add_argument("--batch-windows", type=int, default=2)
    args = parser.parse_args()

    model_name = args.model_name
    model_dir = args.model_dir
    dtype = args.dtype
    first_k_tokens = args.first_k_tokens
    batch_windows = args.batch_windows
    result_name = args.result_name

    ## Bitsqz | different uv_formats (10)
    ## Bitsqz + Topk (0.001) | different uv_formats (10)
    ## SVD (approx + niter=6) + Bitsqz + Topk (0.001) | different uv_formats (10)
    ## ErrorCorrection (approx + niter=6) + Bitsqz + Topk (0.001) | different uv_formats (10) + error_correction_ratios (5) -> (50)

    # UV_FORMATS = ["FP16", "BF16", "Q8_0", "MXFP8", "FP8", "Q4_0", "NF4", "MXFP4", "NF4_DQ", "Q2_K"]
    UV_FORMATS = ["FP16"]
    RANK=512
    OUTLIER_RATIO=0.001
    FMT_S="fp32"
    NITER=6 # for approx
    Q_OVERSAMPLE=0  # for approx

    result_dir = f"results/{result_name}/bitsqz"
    cases = []
    for uv_format in UV_FORMATS:
            cases.append(
                (
                    f"bitsqz_{uv_format}",
                    BitSqueezeCompressor(method=uv_format),
                )
            )

    for name, comp in cases:
        print(f"Running {result_dir}/{name}...")
        run_ppl_eval(
            model_name=model_name,
            model_dir=model_dir,
            dtype=dtype,
            load_in_8bit=True,
            compressor=comp,
            first_k_tokens=first_k_tokens,
            batch_windows=batch_windows,
            result_dir=f"{result_dir}/{name}",
            wandb=False,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cases.clear()

    result_dir = f"results/{result_name}/bitsqz_topk"
    cases = []
    for uv_format in UV_FORMATS:
            cases.append(
                (
                    f"bitsqz_topk_{uv_format}",
                    OutlierSeparationBitSqueezeCompressor(
                         outlier_ratio=OUTLIER_RATIO, 
                         method=uv_format),
                )
            )

    for name, comp in cases:
        print(f"Running {result_dir}/{name}...")
        run_ppl_eval(
            model_name=model_name,
            model_dir=model_dir,
            dtype=dtype,
            load_in_8bit=True,
            compressor=comp,
            first_k_tokens=first_k_tokens,
            batch_windows=batch_windows,
            result_dir=f"{result_dir}/{name}",
            wandb=False,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cases.clear()

    result_dir = f"results/{result_name}/svd_bitsqz_topk"
    cases = []
    for uv_format in UV_FORMATS:
            cases.append(
                (
                    f"svd_bitsqz_topk_{uv_format}",
                    OutlierSeparationOnlineSVDCompressor(
                        rank=RANK,
                        mode="trunc_approx",
                        outlier_ratio=OUTLIER_RATIO,
                        fmt_uv=uv_format,
                        fmt_s=FMT_S,
                        niter=NITER,
                        q_oversample=Q_OVERSAMPLE,
                        svd_device="cuda",
                    ),
                )
            )

    for name, comp in cases:
        print(f"Running {result_dir}/{name}...")
        run_ppl_eval(
            model_name=model_name,
            model_dir=model_dir,
            dtype=dtype,
            load_in_8bit=True,
            compressor=comp,
            first_k_tokens=first_k_tokens,
            batch_windows=batch_windows,
            result_dir=f"{result_dir}/{name}",
            wandb=False,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cases.clear()

    result_dir = f"results/{result_name}/error_correction"
    uv_format = "nf4_dq"
    cases = []
    error_correction_ratios = [0.001, 0.005, 0.01, 0.015, 0.02]
    for error_correction_ratio in error_correction_ratios:
        for uv_format in UV_FORMATS:
            cases.append(
                (
                    f"error_correction_{error_correction_ratio}_{uv_format}",
                    OutlierSeparationOnlineSVDCompressor(
                        rank=RANK,
                        mode="trunc_approx",
                        outlier_ratio=OUTLIER_RATIO,
                        fmt_uv=uv_format,
                        fmt_s=FMT_S,
                        niter=NITER,
                        q_oversample=Q_OVERSAMPLE,
                        svd_error_correction_ratio=error_correction_ratio,
                        svd_device="cuda",
                    ),
                )
            )

    for name, comp in cases:
        print(f"Running {result_dir}/{name}...")
        run_ppl_eval(
            model_name=model_name,
            model_dir=model_dir,
            dtype=dtype,
            load_in_8bit=True,
            compressor=comp,
            first_k_tokens=first_k_tokens,
            batch_windows=batch_windows,
            result_dir=f"{result_dir}/{name}",
            wandb=False,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cases.clear()


if __name__ == "__main__":
    main()
