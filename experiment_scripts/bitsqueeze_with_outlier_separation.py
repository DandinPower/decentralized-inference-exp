import torch
import sys
import gc
from pathlib import Path

import bitsqueeze
from major_entry.compressor import Compressor, Payload
from eval_ppl import run_ppl_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

class BitSqueezeCompressor(Compressor):
    name = "BitSqueeze"
    def __init__(self, method:str, per_feature: bool=False):
        self.method = method
        self.per_feature = per_feature
        if (self.per_feature):
            print(f"[WARNING] The block-wise selection will operate on tokens per feature. Please ensure that the token dimension size is larger than the quantization method block size. For example, the q4_0 block size is 32.")

    def compress(self, x: torch.Tensor) -> Payload:
        x = x.to(dtype=torch.float32, device="cpu")
        if (self.per_feature):
            x = x.transpose(-2, -1).contiguous()
        buffer = bitsqueeze.BitSqueezeBuffer.compress(x, self.method)
        return Payload(data=buffer, meta={"orig_dtype": str(x.dtype)}, nbytes=buffer.size)

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        decompressed = p.data.decompress()
        if (self.per_feature):
            decompressed = decompressed.transpose(-2, -1).contiguous()
        return decompressed.to(device=device, dtype=dtype)

class OutlierSeparationBitSqueezeCompressor(Compressor):
    name = "OutlierSeparationBitSqueeze"
    def __init__(self, outlier_ratio: float, method:str, per_feature: bool=False):
        self.outlier_ratio = outlier_ratio
        self.method = method
        self.per_feature = per_feature
        if (self.per_feature):
            print(f"[WARNING] The block-wise selection will operate on tokens per feature. Please ensure that the token dimension size is larger than the quantization method block size. For example, the q4_0 block size is 32.")

    def compress(self, x: torch.Tensor) -> Payload:
        x = x.to(dtype=torch.float32, device="cpu")
        topk_activation, residual, k = self.separate_topk_activation_and_residual(x, self.outlier_ratio)
        additional_overhead = self.get_sparse_matrix_size(topk_activation)
        if (self.per_feature):
            residual = residual.transpose(-2, -1).contiguous()
        buffer = bitsqueeze.BitSqueezeBuffer.compress(residual, self.method)
        return Payload(data=buffer, meta={"orig_dtype": str(x.dtype), "topk_activation": topk_activation, "k": k}, nbytes=buffer.size + additional_overhead)

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        decompressed = p.data.decompress()
        topk_activation = p.meta["topk_activation"].to_dense()
        if (self.per_feature):
            decompressed = decompressed.transpose(-2, -1).contiguous()
        decompressed = decompressed + topk_activation
        return decompressed.to(device=device, dtype=dtype)

    def separate_topk_activation_and_residual(
        self,
        input_activation: torch.Tensor, 
        topk_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Separates the input activation into a sparse top-k tensor and a residual dense tensor.
        
        Args:
            input_activation: at least 2D Tensor [tokens, features], fp32, cpu.
            topk_ratio: Float between 0 and 1.
            
        Returns:
            tuple containing:
            - topk_activation: Sparse COO Tensor containing only top-k values.
            - residual: Dense Tensor with top-k values set to zero.
            - k: The number of top-k values.
        """
        # 1. Assertions
        assert input_activation.dtype == torch.float32, "input_activation must be fp32"
        assert input_activation.device.type == 'cpu', "input_activation must be on cpu"
        assert input_activation.dim() >= 2, "input_activation must be at least 2D"
        assert 0.0 < topk_ratio < 1.0, "topk_ratio must be smaller than 1 but larger than 0"

        original_shape = input_activation.shape
        feature_dim = original_shape[-1]
        
        flat_input = input_activation.view(-1, feature_dim)
        total_rows, cols = flat_input.shape
        
        k = int(cols * topk_ratio)
        
        # Edge Case: k=0
        if k == 0:
            ndim = input_activation.dim()
            empty_indices = torch.empty((ndim, 0), dtype=torch.long)
            empty_values = torch.empty((0,), dtype=torch.float32)
            topk_sparse = torch.sparse_coo_tensor(empty_indices, empty_values, size=original_shape)
            return topk_sparse, input_activation.clone(), 0

        _, topk_indices = torch.topk(flat_input.abs(), k, dim=1)
        topk_values = torch.gather(flat_input, 1, topk_indices)

        residual_flat = flat_input.clone()
        residual_flat.scatter_(1, topk_indices, 0.0)
        residual = residual_flat.view(original_shape)

        feat_indices = topk_indices.flatten()
        flat_row_indices = torch.arange(total_rows, device=input_activation.device)
        flat_row_indices = flat_row_indices.unsqueeze(1).expand(-1, k).flatten()
        
        spatial_dims = original_shape[:-1] # e.g. (Batch, Seq)
        indices_list = []
        
        current_flat = flat_row_indices
        spatial_indices_reversed = []
        
        for dim_size in reversed(spatial_dims):
            coord = current_flat % dim_size
            spatial_indices_reversed.append(coord)
            current_flat = current_flat // dim_size
            
        indices_list.extend(reversed(spatial_indices_reversed))
        indices_list.append(feat_indices)
        all_indices = torch.stack(indices_list)
        all_values = topk_values.flatten()
        
        topk_activation = torch.sparse_coo_tensor(
            all_indices, 
            all_values, 
            size=original_shape
        )

        return topk_activation, residual, k

    def get_sparse_matrix_size(self, topk_activation: torch.Tensor) -> int:
        """
        Returns the size of the sparse matrix in bytes.
        Since each row has same k non-zero elements, we can calculate the optimal size as:
        - Column indices: k * rows * sizeof(int16)
        - Values: k * rows * sizeof(float32)
        """
        assert topk_activation.dim() >= 2, "topk_activation must be at least two dims [token dimension, feature dimension] or maybe with batch dimension -> [batch, token, feature]"
        # reshape to 2D if needed
        
        total_rows = topk_activation.numel() // topk_activation.shape[-1]
        
        k = topk_activation._nnz() // total_rows
        
        return (k * total_rows * 2) + (k * total_rows * 4)  # int16 + float32


if __name__ == "__main__":
    # List of 1D quantization methods to test
    # QUANTIZATION_METHODS = [
    #     "Q8_0", "Q4_0", "Q2_K", "Q2_K_FAST", 
    #     "BF16", "FP16", "FP8", "FP4", 
    #     "MXFP8", "MXFP4", "NVFP4", 
    #     "NF4", "NF4_DQ", 
    #     "IQ2_XXS", "IQ2_XS", "IQ2_S"
    # ]
    QUANTIZATION_METHODS = [
        "Q2_K_FAST"
    ]
    MODEL_NAME = "Qwen/Qwen3-8B"
    MODEL_DIR = "/mnt/ssd/liaw/Qwen/Qwen3-8B"
    TEST_NAME = "8B_test"
    
    for method in QUANTIZATION_METHODS:
        run_ppl_eval(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,
            wandb=False,
            dtype = "bf16",
            load_in_8bit = True,
            compressor=BitSqueezeCompressor(method=method),
            batch_windows=2,
            first_k_tokens=0,
            result_dir = f"results/{TEST_NAME}/bitsqueeze_{method}",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for method in QUANTIZATION_METHODS:
        run_ppl_eval(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,
            wandb=False,
            dtype = "bf16",
            load_in_8bit = True,
            compressor=BitSqueezeCompressor(method=method, per_feature=True),
            batch_windows=2,
            first_k_tokens=0,
            result_dir = f"results/{TEST_NAME}/bitsqueeze_{method}_perfeature",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTLIER_RATIO = 0.001

    for method in QUANTIZATION_METHODS:
        run_ppl_eval(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,
            wandb=False,
            dtype = "bf16",
            load_in_8bit = True,
            compressor=OutlierSeparationBitSqueezeCompressor(method=method, outlier_ratio=OUTLIER_RATIO),
            batch_windows=2,
            first_k_tokens=0,
            result_dir = f"results/{TEST_NAME}/outlier_separation_bitsqueeze_{method}_{OUTLIER_RATIO}",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for method in QUANTIZATION_METHODS:
        run_ppl_eval(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,
            wandb=False,
            dtype = "bf16",
            load_in_8bit = True,
            compressor=OutlierSeparationBitSqueezeCompressor(method=method, outlier_ratio=OUTLIER_RATIO, per_feature=True),
            batch_windows=2,
            first_k_tokens=0,
            result_dir = f"results/{TEST_NAME}/outlier_separation_bitsqueeze_{method}_{OUTLIER_RATIO}_perfeature",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()