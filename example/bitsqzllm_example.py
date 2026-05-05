import argparse
import gc

import torch

from src.eval_ppl import run_ppl_eval
from src.compressors.bitsqzllm_compressor import BitSqueezeLLMCompressor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-dir", type=str, default="/mnt/ssd/liaw/hf_cache/Qwen/Qwen3.5-0.8B")
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

    result_dir = f"results/{result_name}/bitsqzllm"
    cases = [
        (
            "bitsqzllm_topk_0.001_error_topk_0.01_lowrank_0.2_NF4_DQ_NONE",
            BitSqueezeLLMCompressor(
                topk_ratio=0.001,
                error_topk_ratio=0.01,
                lowrank_ratio=0.2,
                svd_uv_format="NF4_DQ",
                svd_s_format="NONE",
            ),
        )
    ]

    for name, comp in cases:
        try:
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
        finally:
            comp.release()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cases.clear()


if __name__ == "__main__":
    main()
