import argparse
import gc

import torch

from major_entry.eval_ppl import run_ppl_eval
from onlinesvd import (
    OnlineSVDCompressor,
    OnlineSVDBitSqueezeCompressor,
    OutlierSeparationOnlineSVDCompressor,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple Qwen/Qwen3-8B OnlineSVD comparison"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--model-dir", type=str, default="/mnt/ssd/liaw/Qwen/Qwen3-8B")
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--first-k-tokens", type=int, default=0)
    parser.add_argument("--batch-windows", type=int, default=2)
    parser.add_argument("--svd-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--result-dir", type=str, default="results/onlinesvd_qwen3_8b_compare"
    )
    args = parser.parse_args()

    model_name = args.model_name
    model_dir = args.model_dir
    dtype = args.dtype
    first_k_tokens = args.first_k_tokens
    batch_windows = args.batch_windows
    svd_device = args.svd_device


    UV_FORMATS = ["fp32", "fp16", "bf16", "q8_0", "mxfp8", "fp8", "q4_0", "nf4", "mxfp4", "nf4_dq", "q2_k"]
    RANK=512
    OUTLIER_RATIO=0.001
    FMT_S="fp32"
    NITER=6 # for approx
    Q_OVERSAMPLE=0  # for approx

    result_dir = "results/show_the_better_onlinesvd_niters_is_comparable_to_fullsvd"
    cases = []
    for uv_format in UV_FORMATS:
        cases.append(
            (
                f"trunc_approx_{uv_format}",
                OutlierSeparationOnlineSVDCompressor(
                    rank=RANK,
                    mode="trunc_approx",
                    outlier_ratio=OUTLIER_RATIO,
                    fmt_uv=uv_format,
                    fmt_s=FMT_S,
                    niter=NITER,
                    q_oversample=Q_OVERSAMPLE,
                    svd_device=svd_device,
                ),
            )
        )

        cases.append(
            (
                f"trunc_slice_{uv_format}",
                OutlierSeparationOnlineSVDCompressor(
                    rank=RANK,
                    mode="trunc_slice",
                    outlier_ratio=OUTLIER_RATIO,
                    fmt_uv=uv_format,
                    fmt_s=FMT_S,
                    svd_device=svd_device,
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

    result_dir = "results/show_the_none_and_center_difference"
    cases = []
    for uv_format in UV_FORMATS:
        cases.append(
            (
                f"center_none_{uv_format}",
                OutlierSeparationOnlineSVDCompressor(
                    rank=RANK,
                    mode="trunc_approx",
                    outlier_ratio=OUTLIER_RATIO,
                    fmt_uv=uv_format,
                    fmt_s=FMT_S,
                    niter=NITER,
                    q_oversample=Q_OVERSAMPLE,
                    residual_center="none",
                    svd_device=svd_device,
                ),
            )
        )

        cases.append(
            (
                f"center_fp32_{uv_format}",
                OutlierSeparationOnlineSVDCompressor(
                    rank=RANK,
                    mode="trunc_approx",
                    outlier_ratio=OUTLIER_RATIO,
                    fmt_uv=uv_format,
                    fmt_s=FMT_S,
                    niter=NITER,
                    q_oversample=Q_OVERSAMPLE,
                    residual_center="center",
                    center_factor_format=torch.float32,
                    svd_device=svd_device,
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

    result_dir = "results/show_which_error_correction_is_better_and_consider_the_rank_again"
    uv_format = "nf4_dq"
    cases = []
    ranks = [64, 128, 256, 384, 512]
    error_correction_ratios = [0, 0.001, 0.005, 0.01, 0.015, 0.02]
    for error_correction_ratio in error_correction_ratios:
        for rank in ranks:
            cases.append(
                (
                    f"error_correction_{error_correction_ratio}_{uv_format}_rank_{rank}",
                    OutlierSeparationOnlineSVDCompressor(
                        rank=rank,
                        mode="trunc_approx",
                        outlier_ratio=OUTLIER_RATIO,
                        fmt_uv=uv_format,
                        fmt_s=FMT_S,
                        niter=NITER,
                        q_oversample=Q_OVERSAMPLE,
                        svd_error_correction_ratio=error_correction_ratio,
                        svd_device=svd_device,
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
