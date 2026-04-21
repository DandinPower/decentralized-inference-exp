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
    ERROR_CORRECTION_RATIO=0.01  # for error correction
    CTX_LEN_STRIDE_PAIRS=[(512, 256), (1024, 512), (2048, 512), (4096, 512), (8192, 512)]
    # ctx -> list of (target_bits_per_weight, low_rank_ratio)
    CTX_LEN_LOW_RANK_RATIO_PAIRS={
        "Qwen/Qwen3-32B": [(512, [(0.5, 0.032571031808473525, 16), (0.75, 0.08758966662008422, 44), (1.0, 0.14260830143169492, 73), (1.25, 0.19762693624330563, 101), (1.5, 0.2526455710549163, 129), (1.75, 0.30766420586652704, 157), (2.0, 0.3626828406781377, 185), (2.25, 0.4177014754897484, 213)]), (1024, [(0.5, 0.02986020179847188, 30), (0.75, 0.08029973186345817, 82), (1.0, 0.13073926192844446, 133), (1.25, 0.1811787919934308, 185), (1.5, 0.23161832205841706, 237), (1.75, 0.2820578521234034, 288), (2.0, 0.33249738218838965, 340), (2.25, 0.38293691225337595, 392)]), (2048, [(0.5, 0.02559907013107415, 52), (0.75, 0.0688407426497805, 140), (1.0, 0.11208241516848685, 229), (1.25, 0.15532408768719322, 318), (1.5, 0.19856576020589956, 406), (1.75, 0.24180743272460592, 495), (2.0, 0.2850491052433123, 583), (2.25, 0.3282907777620186, 672)]), (4096, [(0.5, 0.019915172131676963, 81), (0.75, 0.053555665597347514, 219), (1.0, 0.08719615906301807, 357), (1.25, 0.12083665252868864, 494), (1.5, 0.1544771459943592, 632), (1.75, 0.18811763946002974, 770), (2.0, 0.2217581329257003, 908), (2.25, 0.25539862639137084, 1046)]), (8192, [(0.5, 0.013790995598063294, 112), (0.75, 0.037086596270467516, 303), (1.0, 0.06038219694287174, 494), (1.25, 0.08367779761527597, 685), (1.5, 0.10697339828768018, 876), (1.75, 0.1302689989600844, 1067), (2.0, 0.1535645996324886, 1258), (2.25, 0.17686020030489286, 1448)])],
        "Qwen/Qwen3-8B": [(512, [(0.5, 0.031837499680683855, 16), (0.75, 0.0856170599521093, 43), (1.0, 0.13939662022353475, 71), (1.25, 0.1931761804949602, 98), (1.5, 0.24695574076638563, 126), (1.75, 0.3007353010378111, 153), (2.0, 0.35451486130923654, 181), (2.25, 0.408294421580662, 209)]), (1024, [(0.5, 0.02865856610350002, 29), (0.75, 0.07706830614319601, 78), (1.0, 0.125478046182892, 128), (1.25, 0.173887786222588, 178), (1.5, 0.222297526262284, 227), (1.75, 0.27070726630198, 277), (2.0, 0.319117006341676, 326), (2.25, 0.367526746381372, 376)]), (2048, [(0.5, 0.023888161438777502, 48), (0.75, 0.06423978549076655, 131), (1.0, 0.10459140954275557, 214), (1.25, 0.1449430335947446, 296), (1.5, 0.18529465764673364, 379), (1.75, 0.22564628169872267, 462), (2.0, 0.2659979057507117, 544), (2.25, 0.30634952980270075, 627)]), (4096, [(0.5, 0.017921770864710303, 73), (0.75, 0.048195032460504744, 197), (1.0, 0.07846829405629918, 321), (1.25, 0.10874155565209362, 445), (1.5, 0.13901481724788806, 569), (1.75, 0.1692880788436825, 693), (2.0, 0.19956134043947693, 817), (2.25, 0.2298346020352714, 941)]), (8192, [(0.5, 0.011951616143160978, 97), (0.75, 0.0321401569255275, 263), (1.0, 0.05232869770789402, 428), (1.25, 0.07251723849026054, 594), (1.5, 0.09270577927262706, 759), (1.75, 0.11289432005499359, 924), (2.0, 0.13308286083736012, 1090), (2.25, 0.15327140161972663, 1255)])]
    }

    result_dir = f"results/{result_name}/lctx_robustness_same_bits"
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
