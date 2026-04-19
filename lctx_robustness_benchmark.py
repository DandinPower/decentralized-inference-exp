import argparse
import gc

import torch

from src.eval_ppl import run_ppl_eval
from src.compressors.onlinesvd import (
    OutlierSeparationOnlineSVDCompressor,
)
from src.compressor import NoneCompressor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-dir", type=str, default="/mnt/ssd/liaw/Qwen/Qwen3.5-0.8B")
    parser.add_argument("--result-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"]
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

    OUTLIER_RATIO=0.001
    FMT_S="fp32"
    NITER=2 # for approx
    Q_OVERSAMPLE=0  # for approx
    ERROR_CORRECTION_RATIO=0.001  # for error correction
    RANK_RATIO=[0.4, 0.35,0.30,0.25, 0.25, 0.2, 0.15, 0.1]
    CTX_LEN_STRIDE_PAIRS=[(8192, 512), (4096, 512), (2048, 512), (1024, 512), (512, 256)]

    result_dir = f"results/{result_name}/lctx_robustness_baseline"
    cases = []

    for ctx_len, stride in CTX_LEN_STRIDE_PAIRS:
        cases.append(
            (
                f"{ctx_len}_{stride}_none",
                NoneCompressor(),
                ctx_len,
                stride,
            )
        )

    for name, comp, ctx_len, stride in cases:
        print(f"Running {result_dir}/{name}...")
        run_ppl_eval(
            model_name=model_name,
            model_dir=model_dir,
            dtype=dtype,
            load_in_8bit=True,
            compressor=comp,
            max_length=ctx_len,
            stride=stride,
            first_k_tokens=first_k_tokens,
            batch_windows=batch_windows,
            result_dir=f"{result_dir}/{name}",
            wandb=False,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cases.clear()

    result_dir = f"results/{result_name}/lctx_robustness"
    uv_format = "nf4_dq"
    cases = []

    for ctx_len, stride in CTX_LEN_STRIDE_PAIRS:
        for rank_ratio in RANK_RATIO:
            rank = int(ctx_len * rank_ratio)
            cases.append(
                (
                    f"error_correction_{ctx_len}_{stride}_{rank}",
                    OutlierSeparationOnlineSVDCompressor(
                        rank=rank,
                        mode="trunc_approx",
                        outlier_ratio=OUTLIER_RATIO,
                        fmt_uv=uv_format,
                        fmt_s=FMT_S,
                        niter=NITER,
                        q_oversample=Q_OVERSAMPLE,
                        svd_error_correction_ratio=ERROR_CORRECTION_RATIO,
                        svd_device="cuda",
                    ),
                    ctx_len,
                    stride,
                )
            )

    for name, comp, ctx_len, stride in cases:
        print(f"Running {result_dir}/{name}...")
        run_ppl_eval(
            model_name=model_name,
            model_dir=model_dir,
            dtype=dtype,
            load_in_8bit=True,
            compressor=comp,
            max_length=ctx_len,
            stride=stride,
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
