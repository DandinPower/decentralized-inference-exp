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

    cases = []
    
    # uv_formats = ["fp32", "fp16", "bf16", "q8_0", "mxfp8", "fp8", "q4_0", "nf4", "mxfp4", "nf4_dq", "q2_k"]
    
    # uv_formats = ["nf4_dq"]
    # ratios = [0.001]
    # niters = [2, 4, 6]
    # oversamples = [0, 2, 4, 6, 8, 10]

    # uv_formats = ["nf4_dq"]
    # ratios = [0.001]
    # niters = [6]
    # oversamples = [0]
    # scale_alphas = [0, 0.25, 0.5, 0.75, 1]

    uv_formats = ["nf4_dq"]
    ratios = [0.001]
    niters = [6]
    oversamples = [0]
    scale_alphas = [0]
    residual_center = "center"
    center_factor_format = torch.float32

    for uv_format in uv_formats:
        for ratio in ratios:
            for oversample in oversamples:
                for niter in niters:
                    for scale_alpha in scale_alphas:
                        cases.append(
                            (
                                f"test_{ratio:.3f}_{uv_format}_{oversample}_{niter}_{scale_alpha}_{residual_center}",
                                OutlierSeparationOnlineSVDCompressor(
                                    rank=512,
                                    mode="trunc_approx",
                                    outlier_ratio=ratio,
                                    fmt_uv=uv_format,
                                    fmt_s="fp32",
                                    niter=niter,
                                    q_oversample=oversample,
                                    residual_center=residual_center,
                                    center_factor_format=center_factor_format,
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
