import argparse
import gc

import torch

from src.eval_ppl import run_ppl_eval

from src.compressors.bitsqz_compressor import (
    BitSqueezeCompressor,
    OutlierSeparationBitSqueezeCompressor,
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

    UV_FORMATS = ["FP16"]
    OUTLIER_RATIOS = [0.001]

    result_dir = f"results/{result_name}/bitsqz"
    cases = []
    for uv_format in UV_FORMATS:
        cases.append(
            (
                f"bitsqz_{uv_format}",
                BitSqueezeCompressor(method=uv_format),
            )
        )

    for outlier_ratio in OUTLIER_RATIOS:
        for uv_format in UV_FORMATS:
            cases.append(
                (
                    f"bitsqz_outlier_{outlier_ratio}_{uv_format}",
                    OutlierSeparationBitSqueezeCompressor(
                        outlier_ratio=outlier_ratio,
                        method=uv_format,
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
