import argparse
import gc

import torch

from src.eval_ppl import run_ppl_eval

from src.compressors.bitsqzllm_compressor import BitSqueezeLLMCompressor, get_topk_ratio_error_topk_ratio_low_rank_ratio_under_svd_nf4dq_scenario

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-dir", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--result-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--first-k-tokens", type=int, default=0)
    parser.add_argument("--batch-windows", type=int, default=1)
    args = parser.parse_args()

    model_name = args.model_name
    model_dir = args.model_dir
    dtype = args.dtype
    first_k_tokens = args.first_k_tokens
    batch_windows = args.batch_windows
    result_name = args.result_name


    # CTX_LEN_STRIDE_PAIRS=[(16384, 2048), (8192, 1024), (4096, 512), (2048, 512), (1024, 512), (512, 256)]
    
    CTX_LEN_STRIDE_PAIRS=[(8192, 1024), (2048, 512), (512, 256)]    # test small set first
    TOPK_SEPARATION_BITS_PORTION_RANGE = [0.0, 0.05, 0.1, 0.15, 0.2]
    ERROR_CORRECTION_BITS_PORTION_RANGE = [0.0, 0.05, 0.1, 0.15, 0.2]
    BITS_PER_WEIGHT_BUDGETS=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    
    FMT_UV="NF4_DQ"    
    FMT_S="NONE"

    MODEL_NAME_TO_DIM_PAIRS={
        "Qwen/Qwen3.5-0.8B": 1024,
        "Qwen/Qwen3-8B": 4096,
    }
    DIM = MODEL_NAME_TO_DIM_PAIRS[model_name]

    result_dir = f"results/{result_name}/heuristic_benchmark"
    cases = []
    for ctx_len, stride in CTX_LEN_STRIDE_PAIRS:
        for bits_per_weight_budget in BITS_PER_WEIGHT_BUDGETS:
            for topk_portion in TOPK_SEPARATION_BITS_PORTION_RANGE:
                for error_topk_portion in ERROR_CORRECTION_BITS_PORTION_RANGE:
                    topk_ratio, error_topk_ratio, low_rank_ratio = get_topk_ratio_error_topk_ratio_low_rank_ratio_under_svd_nf4dq_scenario(
                        rows=ctx_len,
                        cols=DIM,
                        bits_budget=bits_per_weight_budget,
                        topk_portion=topk_portion,
                        error_topk_portion=error_topk_portion,
                    )
                    cases.append(
                        (
                            f"error_correction_{ctx_len}_{stride}_bits_budget_{bits_per_weight_budget}_topk_portion_{topk_portion}_error_topk_portion_{error_topk_portion}",
                            BitSqueezeLLMCompressor(
                                topk_ratio=topk_ratio,
                                error_topk_ratio=error_topk_ratio,
                                lowrank_ratio=low_rank_ratio,
                                svd_uv_format=FMT_UV,
                                svd_s_format=FMT_S,
                            ),
                            ctx_len,
                            stride,
                        )
                    )

    for index, (name, comp, ctx_len, stride) in enumerate(cases):
        print(f"Running {index+1}/{len(cases)}: {result_dir}/{name}...")
        print(f"Compressor config: topk_ratio={comp.topk_ratio}, error_topk_ratio={comp.error_topk_ratio}, lowrank_ratio={comp.lowrank_ratio}, lowrank={ctx_len*comp.lowrank_ratio}")
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
