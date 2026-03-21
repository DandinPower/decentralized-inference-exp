import argparse
import gc

import torch

from src.eval_ppl import run_ppl_eval
from src.compressors.onlinesvd import (
    OnlineSVDCompressor,
    OutlierSeparationOnlineSVDCompressor,
    OnlineSVDBitSqueezeCompressor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-dir", type=str, default="Qwen/Qwen3.5-0.8B")
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

    rank = 512
    mode = "trunc_approx"
    outlier_ratio = 0.001

    result_dir = f"results/{result_name}/onlinesvd"
    cases = [
        (
            "onlinesvd_plain",
            OnlineSVDCompressor(
                rank=rank,
                mode=mode,
                niter=6,
                q_oversample=0,
                svd_device="auto",
            ),
        ),
        (
            f"onlinesvd_outlier_{outlier_ratio}",
            OutlierSeparationOnlineSVDCompressor(
                rank=rank,
                mode=mode,
                outlier_ratio=outlier_ratio,
                niter=6,
                q_oversample=0,
                svd_device="auto",
            ),
        ),
        (
            "onlinesvd_bitsqz_q8_0",
            OnlineSVDBitSqueezeCompressor(
                rank=rank,
                mode=mode,
                fmt_uv="q8_0",
                fmt_s="fp32",
                niter=6,
                q_oversample=0,
                svd_device="auto",
            ),
        ),
    ]

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


if __name__ == "__main__":
    main()
