import argparse
import gc

import torch

from major_entry.compressor import NoneCompressor
from major_entry.qwen3_two_path_compression_eval_ppl import run_ppl_eval
from onlinesvd import OutlierSeparationOnlineSVDCompressor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen/Qwen3-8B two-path test: norm=none, residual=OnlineSVD+topk+bitsqz"
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
        "--svd-error-correction-ratio",
        type=float,
        default=0.0,
        help="Secondary per-row top-k ratio for SVD reconstruction error.",
    )
    parser.add_argument(
        "--result-dir", type=str, default="results/onlinesvd_qwen3_8b_two_path_compare"
    )
    args = parser.parse_args()

    uv_formats = [
        "fp32",
        "fp16",
        "bf16",
        "q8_0",
        "mxfp8",
        "fp8",
        "q4_0",
        "nf4",
        "mxfp4",
        "nf4_dq",
        "q2_k",
    ]
    topk_ratios = [0.005, 0.001]

    cases = []
    for uv_format in uv_formats:
        for ratio in topk_ratios:
            cases.append(
                (
                    f"twopath_norm_none_residual_onlinesvd_bitsqz_topk_{ratio:.3f}_{uv_format}",
                    NoneCompressor(),
                    OutlierSeparationOnlineSVDCompressor(
                        rank=512,
                        mode="trunc_approx",
                        outlier_ratio=ratio,
                        svd_error_correction_ratio=args.svd_error_correction_ratio,
                        fmt_uv=uv_format,
                        fmt_s="fp32",
                        niter=args.niter,
                        svd_device=args.svd_device,
                    ),
                )
            )

    for name, norm_comp, residual_comp in cases:
        print(f"Running {name}...")
        run_ppl_eval(
            model_name=args.model_name,
            model_dir=args.model_dir,
            dtype=args.dtype,
            load_in_8bit=True,
            norm_compressor=norm_comp,
            residual_compressor=residual_comp,
            first_k_tokens=args.first_k_tokens,
            batch_windows=args.batch_windows,
            result_dir=f"{args.result_dir}/{name}",
            wandb=False,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
