import torch
import sys
import gc
from pathlib import Path

from major_entry.compressor import Compressor, Payload
from major_entry.eval_ppl import run_ppl_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

class OnlineSVDCompressor(Compressor):
    name = "OnlineSVD"
    def __init__(self, rank: int, mode: str):
        """
        Args:
            rank: Target rank for truncated SVD.
            mode: 'full', 'trunc_slice', or 'trunc_approx'.
        """
        self.rank = rank
        self.mode = mode

    def compress(self, x: torch.Tensor) -> Payload:
        if x.dim() < 2:
            raise ValueError(f"OnlineSVDCompressor expects at least 2D input, got shape={tuple(x.shape)}")

        # SVD requires float32 for stability. We decompose over (..., seq, feature),
        # so each leading-batch item gets its own U, S, Vh.
        x = x.to(dtype=torch.float32)

        if torch.cuda.is_available():
            x = x.to("cuda")

        seq_dim = x.shape[-2]
        feature_dim = x.shape[-1]
        max_rank = min(seq_dim, feature_dim)

        # Perform SVD on the last two dims (batched over leading dims).
        if self.mode == "full":
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
            compressed_data = (U, S, Vh)
        elif self.mode == "trunc_slice":
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
            k = min(self.rank, max_rank)
            compressed_data = (U[..., :k], S[..., :k], Vh[..., :k, :])
        elif self.mode == "trunc_approx":
            # torch.svd_lowrank is best handled per matrix for robust batched behavior.
            # It returns U, S, V where A ~= U @ diag(S) @ V.T. We store Vh = V.T.
            k = min(self.rank, max_rank)
            flat_x = x.reshape(-1, seq_dim, feature_dim)
            U_list = []
            S_list = []
            Vh_list = []
            for i in range(flat_x.shape[0]):
                U_i, S_i, V_i = torch.svd_lowrank(flat_x[i], q=k, niter=2)
                U_list.append(U_i)
                S_list.append(S_i)
                Vh_list.append(V_i.mT)

            leading_shape = x.shape[:-2]
            U = torch.stack(U_list, dim=0).reshape(*leading_shape, seq_dim, k)
            S = torch.stack(S_list, dim=0).reshape(*leading_shape, k)
            Vh = torch.stack(Vh_list, dim=0).reshape(*leading_shape, k, feature_dim)
            compressed_data = (U, S, Vh)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Calculate traffic bytes based on actual payload tensor storage.
        nbytes = sum(t.numel() * t.element_size() for t in compressed_data)
        
        # Store on CPU to save GPU memory during inference/eval loop
        compressed_data_cpu = tuple(t.cpu() for t in compressed_data)

        return Payload(
            data=compressed_data_cpu,
            meta={
                "orig_dtype": str(x.dtype),
                "shape": tuple(x.shape),
                "seq_dim": seq_dim,
                "feature_dim": feature_dim,
                "effective_rank": compressed_data[1].shape[-1],
            },
            nbytes=nbytes
        )

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        U, S, Vh = p.data
        U = U.to(device=device, dtype=torch.float32)
        S = S.to(device=device, dtype=torch.float32)
        Vh = Vh.to(device=device, dtype=torch.float32)
        
        # Reconstruction: U @ diag(S) @ Vh
        decompressed = U @ (torch.diag_embed(S) @ Vh)
            
        return decompressed.to(device=device, dtype=dtype)

class OutlierSeparationOnlineSVDCompressor(Compressor):
    name = "OutlierSeparationOnlineSVD"
    def __init__(self, rank: int, mode: str, outlier_ratio: float):
        self.outlier_ratio = outlier_ratio
        self.svd = OnlineSVDCompressor(rank, mode)

    def compress(self, x: torch.Tensor) -> Payload:
        x = x.to(dtype=torch.float32)
        if torch.cuda.is_available():
            x = x.to("cuda")
            
        topk_activation, residual, k = self.separate_topk_activation_and_residual(x, self.outlier_ratio)
        
        # Compress residual using SVD
        svd_payload = self.svd.compress(residual)
        
        # Calculate overhead for sparse part
        sparse_overhead = self.get_sparse_matrix_size(topk_activation)
        
        # Payload data: (svd_components, sparse_topk)
        return Payload(
            data=(svd_payload.data, topk_activation.cpu()),
            meta={**svd_payload.meta, "k": k},
            nbytes=svd_payload.nbytes + sparse_overhead
        )

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        svd_data, topk_activation = p.data
        
        # Decompress SVD part
        svd_p = Payload(data=svd_data, meta=p.meta, nbytes=0)
        decompressed_residual = self.svd.decompress(svd_p, device, torch.float32)
        
        # Add TopK
        # topk_activation is SparseTensor on CPU, move to device and densify
        topk_dense = topk_activation.to(device=device, dtype=torch.float32).to_dense()
        
        final = decompressed_residual + topk_dense
        return final.to(dtype=dtype)

    def separate_topk_activation_and_residual(
        self,
        input_activation: torch.Tensor, 
        topk_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Separates the input activation into a sparse top-k tensor and a residual dense tensor.
        """
        assert input_activation.dtype == torch.float32, "input_activation must be fp32"
        assert input_activation.dim() >= 2, "input_activation must be at least 2D"
        assert 0.0 < topk_ratio < 1.0, "topk_ratio must be between 0 and 1"

        original_shape = input_activation.shape
        feature_dim = original_shape[-1]
        
        flat_input = input_activation.view(-1, feature_dim)
        total_rows, cols = flat_input.shape
        
        k = int(cols * topk_ratio)
        
        if k == 0:
            ndim = input_activation.dim()
            empty_indices = torch.empty((ndim, 0), dtype=torch.long, device=input_activation.device)
            empty_values = torch.empty((0,), dtype=torch.float32, device=input_activation.device)
            topk_sparse = torch.sparse_coo_tensor(empty_indices, empty_values, size=original_shape, device=input_activation.device)
            return topk_sparse, input_activation.clone(), 0

        _, topk_indices = torch.topk(flat_input.abs(), k, dim=1)
        topk_values = torch.gather(flat_input, 1, topk_indices)

        residual_flat = flat_input.clone()
        residual_flat.scatter_(1, topk_indices, 0.0)
        residual = residual_flat.view(original_shape)

        # Construct Sparse Tensor Indices
        feat_indices = topk_indices.flatten()
        flat_row_indices = torch.arange(total_rows, device=input_activation.device)
        flat_row_indices = flat_row_indices.unsqueeze(1).expand(-1, k).flatten()
        
        spatial_dims = original_shape[:-1] 
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
            size=original_shape,
            device=input_activation.device
        )

        return topk_activation, residual, k

    def get_sparse_matrix_size(self, topk_activation: torch.Tensor) -> int:
        """
        Returns the size of the sparse matrix in bytes.
        """
        nnz = topk_activation._nnz()
        ndim = topk_activation.dim()
        
        # Values: float32 (4 bytes)
        values_size = nnz * 4
        
        # Indices: int64 (8 bytes) usually in PyTorch, but we can assume efficient storage
        # If we serialize, we might use smaller types, but let's stick to PyTorch defaults or estimation.
        # PyTorch Sparse COO uses 1 set of values and 'ndim' sets of indices.
        indices_size = nnz * ndim * 8 
        
        return values_size + indices_size


if __name__ == "__main__":
    ranks = [64, 128, 256, 512, 1024, 2048]
    
    configs = []
    # Trunc (Approx) and Trunc (Slice) for each rank
    for r in ranks:
        configs.append(("trunc_approx", r))
        configs.append(("trunc_slice", r))
    # Full 2048 (assuming Full SVD)
    configs.append(("full", 2048))

    MODEL_NAME = "Qwen/Qwen3-8B"
    MODEL_DIR = "/mnt/ssd/liaw/Qwen/Qwen3-8B"
    TEST_NAME = "8B_test"

    # 1. OnlineSVD (No Outlier Separation)
    print("=== Running OnlineSVD Experiments ===")
    for mode, rank in configs:
        method_name = f"{mode}_{rank}"
        print(f"Running {method_name}...")
        
        run_ppl_eval(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,
            wandb=False,
            dtype = "bf16",
            load_in_8bit = True,
            compressor=OnlineSVDCompressor(rank=rank, mode=mode),
            batch_windows=2,
            first_k_tokens=0,
            result_dir = f"results/{TEST_NAME}/onlinesvd_{method_name}",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. OutlierSeparation + OnlineSVD
    OUTLIER_RATIO = 0.001
    print(f"\n=== Running OutlierSeparation ({OUTLIER_RATIO}) + OnlineSVD Experiments ===")
    for mode, rank in configs:
        method_name = f"{mode}_{rank}"
        print(f"Running {method_name} with Outlier Separation...")
        
        run_ppl_eval(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,
            wandb=False,
            dtype = "bf16",
            load_in_8bit = True,
            compressor=OutlierSeparationOnlineSVDCompressor(rank=rank, mode=mode, outlier_ratio=OUTLIER_RATIO),
            batch_windows=2,
            first_k_tokens=0,
            result_dir = f"results/{TEST_NAME}/outlier_separation_onlinesvd_{method_name}_{OUTLIER_RATIO}",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. OutlierSeparation (0.005) + OnlineSVD
    OUTLIER_RATIO = 0.005
    print(f"\n=== Running OutlierSeparation ({OUTLIER_RATIO}) + OnlineSVD Experiments ===")
    for mode, rank in configs:
        method_name = f"{mode}_{rank}"
        print(f"Running {method_name} with Outlier Separation...")
        
        run_ppl_eval(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,
            wandb=False,
            dtype = "bf16",
            load_in_8bit = True,
            compressor=OutlierSeparationOnlineSVDCompressor(rank=rank, mode=mode, outlier_ratio=OUTLIER_RATIO),
            batch_windows=2,
            first_k_tokens=0,
            result_dir = f"results/{TEST_NAME}/outlier_separation_onlinesvd_{method_name}_{OUTLIER_RATIO}",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()