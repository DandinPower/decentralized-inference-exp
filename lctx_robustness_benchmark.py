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

    OUTLIER_RATIO=0.001
    FMT_S="fp32"
    NITER=2 # for approx
    Q_OVERSAMPLE=0  # for approx
    ERROR_CORRECTION_RATIO=0.001  # for error correction
    CTX_LEN_STRIDE_PAIRS=[(512, 256), (1024, 512), (2048, 512), (4096, 512), (8192, 512)]
    # ctx -> list of (target_bits_per_weight, low_rank_ratio)
    CTX_LEN_LOW_RANK_RATIO_PAIRS={
        "Qwen/Qwen3-32B": [(512, [(0.5, 0.09595249911144904, 49), (0.75, 0.15097113392305975, 77), (1.0, 0.20598976873467043, 105), (1.25, 0.26100840354628113, 133), (1.5, 0.3160270383578918, 161), (1.75, 0.3710456731695025, 189), (2.0, 0.4260643079811132, 218), (2.25, 0.4810829427927239, 246)]), (1024, [(0.5, 0.08796654043333609, 90), (0.75, 0.13840607049832238, 141), (1.0, 0.18884560056330868, 193), (1.25, 0.23928513062829498, 245), (1.5, 0.2897246606932813, 296), (1.75, 0.34016419075826754, 348), (2.0, 0.39060372082325384, 399), (2.25, 0.44104325088824015, 451)]), (2048, [(0.5, 0.07541347687262386, 154), (0.75, 0.1186551493913302, 243), (1.0, 0.16189682191003657, 331), (1.25, 0.2051384944287429, 420), (1.5, 0.24838016694744924, 508), (1.75, 0.2916218394661556, 597), (2.0, 0.334863511984862, 685), (2.25, 0.3781051845035683, 774)]), (4096, [(0.5, 0.058669020604129436, 240), (0.75, 0.0923095140698, 378), (1.0, 0.12595000753547053, 515), (1.25, 0.1595905010011411, 653), (1.5, 0.19323099446681166, 791), (1.75, 0.2268714879324822, 929), (2.0, 0.26051198139815274, 1067), (2.25, 0.2941524748638233, 1204)]), (8192, [(0.5, 0.040627527572672954, 332), (0.75, 0.06392312824507718, 523), (1.0, 0.08721872891748139, 714), (1.25, 0.11051432958988562, 905), (1.5, 0.13380993026228985, 1096), (1.75, 0.15710553093469407, 1287), (2.0, 0.18040113160709828, 1477), (2.25, 0.2036967322795025, 1668)])],
        "Qwen/Qwen3-8B": [(512, [(0.5, 0.09379155311336597, 48), (0.75, 0.14757111338479142, 75), (1.0, 0.20135067365621687, 103), (1.25, 0.25513023392764234, 130), (1.5, 0.3089097941990678, 158), (1.75, 0.3626893544704932, 185), (2.0, 0.41646891474191866, 213), (2.25, 0.4702484750133441, 240)]), (1024, [(0.5, 0.0844265866292298, 86), (0.75, 0.1328363266689258, 136), (1.0, 0.18124606670862178, 185), (1.25, 0.2296558067483178, 235), (1.5, 0.27806554678801376, 284), (1.75, 0.32647528682770977, 334), (2.0, 0.37488502686740577, 383), (2.25, 0.42329476690710177, 433)]), (2048, [(0.5, 0.07037323234666887, 144), (0.75, 0.11072485639865791, 226), (1.0, 0.15107648045064695, 309), (1.25, 0.19142810450263598, 392), (1.5, 0.231779728554625, 474), (1.75, 0.27213135260661403, 557), (2.0, 0.3124829766586031, 639), (2.25, 0.35283460071059214, 722)]), (4096, [(0.5, 0.0527965682230655, 216), (0.75, 0.08306982981885994, 340), (1.0, 0.11334309141465439, 464), (1.25, 0.1436163530104488, 588), (1.5, 0.17388961460624328, 712), (1.75, 0.2041628762020377, 836), (2.0, 0.23443613779783215, 960), (2.25, 0.2647093993936266, 1084)]), (8192, [(0.5, 0.035208815124447214, 288), (0.75, 0.05539735590681373, 453), (1.0, 0.07558589668918025, 619), (1.25, 0.09577443747154678, 784), (1.5, 0.1159629782539133, 949), (1.75, 0.1361515190362798, 1115), (2.0, 0.15634005981864635, 1280), (2.25, 0.17652860060101286, 1446)])]
    }

    result_dir = f"results/{result_name}/lctx_robustness_error_0.001_aligned"
    uv_format = "nf4_dq"
    cases = []

    for index, (ctx_len, stride) in enumerate(CTX_LEN_STRIDE_PAIRS):
        for target_bits, rank_ratio, low_rank in CTX_LEN_LOW_RANK_RATIO_PAIRS[model_name][index][1]:
            cases.append(
                (
                    f"error_correction_{ctx_len}_{stride}_{low_rank}_{target_bits}",
                    OutlierSeparationOnlineSVDCompressor(
                        rank=low_rank,
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
