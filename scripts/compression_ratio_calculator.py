from dataclasses import dataclass

@dataclass
class CompressionConfig:
    context_length: int
    feature_dimension: int
    quantization_bits: int
    outlier_separation_topk_ratio: float
    error_correction_topk_ratio: float
    context_length_to_low_rank_ratio: float

@dataclass
class SolveLowRankRatioConfig:
    context_length: int
    feature_dimension: int
    quantization_bits: int
    outlier_separation_topk_ratio: float
    error_correction_topk_ratio: float
    target_bits_per_weight: float

def get_compression_bits_per_weight(compression_config: CompressionConfig) -> float:
    original_size = compression_config.context_length * compression_config.feature_dimension * 32
    outlier_separation_overhead = compression_config.context_length * compression_config.feature_dimension * 32 * compression_config.outlier_separation_topk_ratio
    error_correction_overhead = compression_config.context_length * compression_config.feature_dimension * 32 * compression_config.error_correction_topk_ratio
    # M x K (quantized)
    low_rank_svd_storage = compression_config.context_length * compression_config.context_length * compression_config.context_length_to_low_rank_ratio * compression_config.quantization_bits
    # K x D (quantized)
    low_rank_svd_storage += compression_config.context_length * compression_config.context_length_to_low_rank_ratio * compression_config.feature_dimension * compression_config.quantization_bits
    # K (fp32)
    low_rank_svd_storage += compression_config.context_length * compression_config.context_length_to_low_rank_ratio * 32
    compressed_size = outlier_separation_overhead + error_correction_overhead + low_rank_svd_storage
    compression_ratio = compressed_size / original_size
    bits_per_weight = compression_ratio * 32
    return bits_per_weight

def get_low_rank_ratio_for_target_bits_per_weight(config: SolveLowRankRatioConfig) -> float:
    target_compressed_size = config.target_bits_per_weight * config.context_length * config.feature_dimension
    outlier_separation_overhead = config.context_length * config.feature_dimension * 32 * config.outlier_separation_topk_ratio
    error_correction_overhead = config.context_length * config.feature_dimension * 32 * config.error_correction_topk_ratio
    low_rank_svd_storage = target_compressed_size - outlier_separation_overhead - error_correction_overhead
    low_rank_ratio = low_rank_svd_storage / (config.quantization_bits * (config.context_length * config.context_length + config.context_length * config.feature_dimension) + 32 * config.context_length)
    return low_rank_ratio

if __name__ == "__main__":
    TARGET_BITS_PER_WEIGHT = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    CONTEXT_LENGTH = [512, 1024, 2048, 4096, 8192]
    FEATURE_DIMENSION = 5120 # qwen3-32b
    QUANTIZATION_BITS = 4.12515 # nf4_dq
    OUTLIER_SEPARATION_TOPK_RATIO = 0.001
    ERROR_CORRECTION_TOPK_RATIO = 0.01

    answers = []
    for ctx_length in CONTEXT_LENGTH:
        ctx_answers = []
        for target_bits in TARGET_BITS_PER_WEIGHT:
            solve_config = SolveLowRankRatioConfig(
                context_length=ctx_length,
                feature_dimension=FEATURE_DIMENSION,
                quantization_bits=QUANTIZATION_BITS,
                outlier_separation_topk_ratio=OUTLIER_SEPARATION_TOPK_RATIO,
                error_correction_topk_ratio=ERROR_CORRECTION_TOPK_RATIO,
                target_bits_per_weight=target_bits
            )
            low_rank_ratio = get_low_rank_ratio_for_target_bits_per_weight(solve_config)
            low_rank = int(ctx_length * low_rank_ratio)
            print(f"Context Length: {ctx_length}, Target Bits/Weight: {target_bits}, Required Low-Rank Ratio: {low_rank_ratio:.4f}, Required Low-Rank: {low_rank}")
            ctx_answers.append((target_bits, low_rank_ratio, low_rank))
        answers.append((ctx_length, ctx_answers))
    print(answers)

    FEATURE_DIMENSION = 4096 # for qwen3-8b

    answers = []
    for ctx_length in CONTEXT_LENGTH:
        ctx_answers = []
        for target_bits in TARGET_BITS_PER_WEIGHT:
            solve_config = SolveLowRankRatioConfig(
                context_length=ctx_length,
                feature_dimension=FEATURE_DIMENSION,
                quantization_bits=QUANTIZATION_BITS,
                outlier_separation_topk_ratio=OUTLIER_SEPARATION_TOPK_RATIO,
                error_correction_topk_ratio=ERROR_CORRECTION_TOPK_RATIO,
                target_bits_per_weight=target_bits
            )
            low_rank_ratio = get_low_rank_ratio_for_target_bits_per_weight(solve_config)
            low_rank = int(ctx_length * low_rank_ratio)
            print(f"Context Length: {ctx_length}, Target Bits/Weight: {target_bits}, Required Low-Rank Ratio: {low_rank_ratio:.4f}, Required Low-Rank: {low_rank}")
            ctx_answers.append((target_bits, low_rank_ratio, low_rank))
        answers.append((ctx_length, ctx_answers))
    print(answers)