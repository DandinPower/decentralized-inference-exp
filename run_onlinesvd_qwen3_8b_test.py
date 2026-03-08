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
    parser.add_argument("--niter", type=int, default=2)
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
    niter = args.niter
    svd_device = args.svd_device

    cases = []

    # cases.append(
    #     (
    #         "onlinesvd_only",
    #         OnlineSVDCompressor(
    #             rank=512,
    #             mode="trunc_approx",
    #             fmt_uv="fp32",
    #             fmt_s="fp32",
    #             niter=niter,
    #             svd_device=svd_device,
    #         ),
    #     )
    # )

    # cases.append(
    #     (
    #         "onlinesvd_bitsqz",
    #         OnlineSVDBitSqueezeCompressor(
    #             rank=512,
    #             mode="trunc_approx",
    #             fmt_uv="nf4_dq",
    #             fmt_s="fp32",
    #             niter=niter,
    #             svd_device=svd_device,
    #         ),
    #     )
    # )

    # uv_formats = ["fp32", "fp16", "bf16", "q8_0", "mxfp8", "fp8", "q4_0", "nf4", "mxfp4", "nf4_dq", "q2_k"]

    uv_formats = ["fp32"]

    for uv_format in uv_formats:
        for ratio in [0.005, 0.001, 0]:
            cases.append(
                (
                    f"test_{ratio:.3f}_{uv_format}",
                    OutlierSeparationOnlineSVDCompressor(
                        rank=512,
                        mode="trunc_approx",
                        outlier_ratio=ratio,
                        fmt_uv=uv_format,
                        fmt_s="fp32",
                        niter=niter,
                        svd_device=svd_device,
                    ),
                )
            )

    for name, comp in cases:
        print(f"Running {name}...")
        run_ppl_eval(
            model_name=model_name,
            model_dir=model_dir,
            dtype=dtype,
            load_in_8bit=True,
            compressor=comp,
            first_k_tokens=first_k_tokens,
            batch_windows=batch_windows,
            result_dir=f"{args.result_dir}/{name}",
            wandb=False,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
