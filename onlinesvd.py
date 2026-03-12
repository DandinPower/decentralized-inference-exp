#!/usr/bin/env python3
"""Unified OnlineSVD compressions.

Supports:
- Plain OnlineSVD with truncation options
- Optional top-k outlier separation
- Optional per-component BitSqueeze quantization (U/Vh and S can use different formats)

This module is intended to be imported by evaluation scripts and provides the
same entry points as the previous split implementations.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any

import torch

from major_entry.compressor import Compressor, Payload

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FORMAT_TO_METHOD = {
    "fp32": None,
    "fp16": "FP16",
    "bf16": "BF16",
    "q8_0": "Q8_0",
    "mxfp8": "MXFP8",
    "fp8": "FP8",
    "q4_0": "Q4_0",
    "nf4": "NF4",
    "mxfp4": "MXFP4",
    "nf4_dq": "NF4_DQ",
    "q2_k": "Q2_K",
}

FACTOR_DTYPE_TO_NAME = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

SUPPORTED_MODES = ("full", "trunc_slice", "trunc_approx")

_BITSQUEEZE = None


def _import_bitsqueeze():
    """Import bitsqueeze with a safe fallback for environments.

    Some environments require torch lib path to be present in LD_LIBRARY_PATH.
    """
    global _BITSQUEEZE
    if _BITSQUEEZE is not None:
        return _BITSQUEEZE

    try:
        _BITSQUEEZE = importlib.import_module("bitsqueeze")
    except ImportError as err:
        if "libc10.so" not in str(err):
            raise
        torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{torch_lib_dir}:{prev}" if prev else str(torch_lib_dir)
        )
        _BITSQUEEZE = importlib.import_module("bitsqueeze")

    return _BITSQUEEZE


def _separate_topk_activation_and_residual(
    input_activation: torch.Tensor,
    topk_ratio: float,
) -> tuple[torch.Tensor | None, torch.Tensor, int]:
    """Split out top-k values per flattened token row.

    Returns tuple(sparse_topk, residual, k).
    """
    if input_activation.dtype != torch.float32:
        raise ValueError("input_activation must be fp32")
    if input_activation.dim() < 2:
        raise ValueError("input_activation must be at least 2D")
    if not (0.0 <= topk_ratio < 1.0):
        raise ValueError("topk_ratio must be in [0, 1)")

    if topk_ratio == 0.0:
        return None, input_activation.clone(), 0

    original_shape = input_activation.shape
    feature_dim = original_shape[-1]

    flat_input = input_activation.view(-1, feature_dim)
    total_rows, cols = flat_input.shape
    k = int(cols * topk_ratio)

    if k <= 0:
        return None, input_activation.clone(), 0

    _, topk_indices = torch.topk(flat_input.abs(), k, dim=1)
    topk_values = torch.gather(flat_input, 1, topk_indices)

    residual_flat = flat_input.clone()
    residual_flat.scatter_(1, topk_indices, 0.0)
    residual = residual_flat.view(original_shape)

    feat_indices = topk_indices.reshape(-1)
    flat_row_indices = torch.arange(total_rows, device=input_activation.device)
    flat_row_indices = flat_row_indices.unsqueeze(1).expand(-1, k).reshape(-1)

    # Reconstruct full sparse indices from flattened row index + feature index.
    spatial_dims = original_shape[:-1]
    indices_list: list[torch.Tensor] = []
    current = flat_row_indices
    spatial_indices_reversed: list[torch.Tensor] = []

    for dim_size in reversed(spatial_dims):
        coord = current % dim_size
        spatial_indices_reversed.append(coord)
        current = current // dim_size

    indices_list.extend(reversed(spatial_indices_reversed))
    indices_list.append(feat_indices)
    all_indices = torch.stack(indices_list)

    return (
        torch.sparse_coo_tensor(
            all_indices,
            topk_values.reshape(-1),
            size=original_shape,
            device=input_activation.device,
        ),
        residual,
        k,
    )


def _sparse_matrix_bytes(topk_activation: torch.Tensor | None) -> int:
    # if topk_activation is None:
    #     return 0
    # nnz = topk_activation._nnz()
    # ndim = topk_activation.dim()
    # return nnz * 4 + nnz * ndim * 8
    """
    Returns the size of the sparse matrix in bytes.
    Since each row has same k non-zero elements, we can calculate the optimal size as:
    - Column indices: k * rows * sizeof(int16)
    - Values: k * rows * sizeof(float32)
    """
    topk = topk_activation
    if topk is None:
        return 0

    assert topk.dim() >= 2, (
        "topk_activation must be at least two dims [token dimension, feature dimension] or maybe with batch dimension -> [batch, token, feature]"
    )
    # reshape to 2D if needed

    total_rows = topk.numel() // topk.shape[-1]

    k = topk._nnz() // total_rows

    return (k * total_rows * 2) + (k * total_rows * 4)  # int16 + float32


class OnlineSVDCompressor(Compressor):
    name = "OnlineSVD"

    def __init__(
        self,
        rank: int,
        mode: str,
        *,
        outlier_ratio: float = 0.0,
        fmt_uv: str = "fp32",
        fmt_s: str | None = None,
        niter: int = 2,
        q_oversample: int = 0,
        svd_device: str = "auto",
        residual_scale_alpha: float = 0.0,
        residual_scale_eps: float = 1e-6,
        residual_scale_q: float = 0.99,
        scale_factor_format: torch.dtype = torch.float32,
        residual_center: str = "none",
        center_factor_format: torch.dtype = torch.float32,
    ):
        self.rank = int(rank)
        self.mode = mode
        self.outlier_ratio = float(outlier_ratio)
        self.fmt_uv = fmt_uv.lower()
        self.fmt_s = fmt_s.lower() if fmt_s is not None else self.fmt_uv
        self.niter = int(niter)
        self.q_oversample = int(q_oversample)
        self.svd_device = svd_device
        self.residual_scale_alpha = float(residual_scale_alpha)
        self.residual_scale_eps = float(residual_scale_eps)
        self.residual_scale_q = float(residual_scale_q)
        self.scale_factor_dtype = self._validate_factor_dtype(
            scale_factor_format, "scale_factor_format"
        )
        self.residual_center = str(residual_center).lower()
        self.center_factor_dtype = self._validate_factor_dtype(
            center_factor_format, "center_factor_format"
        )

        if self.rank <= 0:
            raise ValueError(f"rank must be >0, got {self.rank}")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown mode: {self.mode}")
        if not (0.0 <= self.outlier_ratio < 1.0):
            raise ValueError(
                f"outlier_ratio must be in [0,1), got {self.outlier_ratio}"
            )
        if self.fmt_uv not in FORMAT_TO_METHOD:
            raise ValueError(f"Unknown fmt_uv: {self.fmt_uv}")
        if self.fmt_s not in FORMAT_TO_METHOD:
            raise ValueError(f"Unknown fmt_s: {self.fmt_s}")
        if self.niter < 0:
            raise ValueError(f"niter must be >=0, got {self.niter}")
        if self.q_oversample < 0:
            raise ValueError(f"q_oversample must be >=0, got {self.q_oversample}")
        if self.svd_device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"svd_device must be auto/cpu/cuda, got {self.svd_device}")
        if not (0.0 <= self.residual_scale_alpha <= 1.0):
            raise ValueError(
                "residual_scale_alpha must be in [0,1], "
                f"got {self.residual_scale_alpha}"
            )
        if self.residual_scale_eps <= 0.0:
            raise ValueError(
                f"residual_scale_eps must be >0, got {self.residual_scale_eps}"
            )
        if not (0.0 < self.residual_scale_q < 1.0):
            raise ValueError(
                f"residual_scale_q must be in (0,1), got {self.residual_scale_q}"
            )
        if self.residual_center not in ("none", "center"):
            raise ValueError(
                f"residual_center must be one of none/center, got {self.residual_center}"
            )
        if self._use_residual_scaling and self._use_residual_centering:
            raise ValueError(
                "residual scaling and residual centering cannot be enabled together"
            )

        self.name = self._build_name("OnlineSVD")

    @property
    def _use_bitsqueeze(self) -> bool:
        return self.fmt_uv != "fp32" or self.fmt_s != "fp32"

    @property
    def _use_residual_scaling(self) -> bool:
        return self.residual_scale_alpha > 0.0

    @property
    def _use_residual_centering(self) -> bool:
        return self.residual_center == "center"

    @staticmethod
    def _validate_factor_dtype(
        factor_dtype: torch.dtype,
        arg_name: str,
    ) -> torch.dtype:
        if not isinstance(factor_dtype, torch.dtype):
            raise ValueError(
                f"{arg_name} must be a torch.dtype and one of "
                "torch.float32/torch.float16/torch.bfloat16"
            )
        if factor_dtype not in FACTOR_DTYPE_TO_NAME:
            raise ValueError(
                f"{arg_name} must be one of torch.float32/torch.float16/torch.bfloat16"
            )
        return factor_dtype

    @property
    def _scale_factor_format_name(self) -> str:
        return FACTOR_DTYPE_TO_NAME[self.scale_factor_dtype]

    @property
    def _center_factor_format_name(self) -> str:
        return FACTOR_DTYPE_TO_NAME[self.center_factor_dtype]

    def _build_name(self, prefix: str) -> str:
        parts = [prefix, self.mode, str(self.rank)]
        if self.mode == "trunc_approx":
            parts.append(f"niter{self.niter}")
            parts.append(f"qov{self.q_oversample}")
        if self.outlier_ratio > 0:
            parts.append(f"out{self.outlier_ratio:g}")
        if self._use_bitsqueeze:
            parts.append(f"uv-{self.fmt_uv}")
            if self.fmt_s != self.fmt_uv:
                parts.append(f"s-{self.fmt_s}")
            else:
                parts.append(f"{self.fmt_s}")
        if self._use_residual_scaling:
            parts.append(f"rscale-a{self.residual_scale_alpha:g}")
            parts.append(f"q{self.residual_scale_q:g}")
            parts.append(f"eps{self.residual_scale_eps:g}")
            parts.append(f"sf-{self._scale_factor_format_name}")
        if self._use_residual_centering:
            parts.append("rcenter")
            parts.append(f"cf-{self._center_factor_format_name}")
        return "_".join(parts)

    def _pick_svd_device(self, x: torch.Tensor) -> torch.device:
        if self.svd_device == "auto":
            return x.device
        if self.svd_device == "cpu":
            return torch.device("cpu")
        if self.svd_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "svd_device='cuda' requested but CUDA is unavailable"
                )
            return torch.device("cuda")
        raise ValueError(f"Unknown svd_device: {self.svd_device}")

    def _svd_decompose(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_dim = x.shape[-2]
        feature_dim = x.shape[-1]
        max_rank = min(seq_dim, feature_dim)
        k = min(self.rank, max_rank)
        q = min(max_rank, k + self.q_oversample)

        if self.mode == "full":
            return torch.linalg.svd(x, full_matrices=False)

        if self.mode == "trunc_slice":
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
            return U[..., :k], S[..., :k], Vh[..., :k, :]

        flat_x = x.reshape(-1, seq_dim, feature_dim)
        U_list = []
        S_list = []
        Vh_list = []
        for i in range(flat_x.shape[0]):
            U_i, S_i, V_i = torch.svd_lowrank(flat_x[i], q=q, niter=self.niter)
            U_list.append(U_i[:, :k])
            S_list.append(S_i[:k])
            Vh_list.append(V_i[:, :k].mT)

        lead_shape = x.shape[:-2]
        U = torch.stack(U_list, dim=0).reshape(*lead_shape, seq_dim, k)
        S = torch.stack(S_list, dim=0).reshape(*lead_shape, k)
        Vh = torch.stack(Vh_list, dim=0).reshape(*lead_shape, k, feature_dim)
        return U, S, Vh

    def _quantize(self, x: torch.Tensor, fmt: str) -> tuple[Any, int]:
        x_cpu = x.detach().to(dtype=torch.float32, device="cpu").contiguous()
        method = FORMAT_TO_METHOD[fmt]
        if method is None:
            return x_cpu, int(x_cpu.numel() * x_cpu.element_size())
        bitsqueeze = _import_bitsqueeze()
        buffer = bitsqueeze.BitSqueezeBuffer.compress(x_cpu, method)
        return buffer, int(buffer.size)

    def _dequantize(self, data: Any, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(data):
            out = data
        else:
            out = data.decompress()
        return out.to(device=device, dtype=torch.float32)

    def _compute_feature_scale(self, residual: torch.Tensor) -> torch.Tensor:
        p_abs = torch.quantile(residual.abs(), self.residual_scale_q, dim=-2)
        return torch.clamp(p_abs, min=self.residual_scale_eps).pow(
            self.residual_scale_alpha
        )

    def compress(self, x: torch.Tensor) -> Payload:
        if x.dim() < 2:
            raise ValueError(
                f"OnlineSVDCompressor expects at least 2D input, got shape={tuple(x.shape)}"
            )

        x_fp32 = x.to(dtype=torch.float32)
        work_device = self._pick_svd_device(x_fp32)
        x_work = x_fp32.to(work_device)

        if self.outlier_ratio > 0:
            topk_activation, residual, k = _separate_topk_activation_and_residual(
                x_work, self.outlier_ratio
            )
        else:
            topk_activation = None
            residual = x_work
            k = 0

        if self._use_residual_scaling:
            scale_factor = self._compute_feature_scale(residual)
            residual_for_svd = residual / scale_factor.unsqueeze(-2)
            scale_factor_payload = (
                scale_factor.detach()
                .to(device="cpu", dtype=self.scale_factor_dtype)
                .contiguous()
            )
            scale_factor_bytes = int(
                scale_factor_payload.numel() * scale_factor_payload.element_size()
            )
            center_factor_payload = None
            center_factor_bytes = 0
        elif self._use_residual_centering:
            center_factor = residual.mean(dim=-2)
            residual_for_svd = residual - center_factor.unsqueeze(-2)
            center_factor_payload = (
                center_factor.detach()
                .to(device="cpu", dtype=self.center_factor_dtype)
                .contiguous()
            )
            center_factor_bytes = int(
                center_factor_payload.numel() * center_factor_payload.element_size()
            )
            scale_factor_payload = None
            scale_factor_bytes = 0
        else:
            scale_factor_payload = None
            residual_for_svd = residual
            scale_factor_bytes = 0
            center_factor_payload = None
            center_factor_bytes = 0

        U, S, Vh = self._svd_decompose(residual_for_svd)

        qU, u_bytes = self._quantize(U, self.fmt_uv)
        qS, s_bytes = self._quantize(S, self.fmt_s)
        qVh, vh_bytes = self._quantize(Vh, self.fmt_uv)

        nbytes = (
            u_bytes
            + s_bytes
            + vh_bytes
            + _sparse_matrix_bytes(topk_activation)
            + scale_factor_bytes
            + center_factor_bytes
        )

        if topk_activation is not None:
            topk_cpu = topk_activation.to("cpu")
        else:
            topk_cpu = None

        return Payload(
            data=(
                qU,
                qS,
                qVh,
                topk_cpu,
                scale_factor_payload,
                center_factor_payload,
            ),
            meta={
                "orig_dtype": str(x.dtype),
                "shape": tuple(x.shape),
                "seq_dim": int(x.shape[-2]),
                "feature_dim": int(x.shape[-1]),
                "effective_rank": int(S.shape[-1]),
                "mode": self.mode,
                "outlier_ratio": self.outlier_ratio,
                "topk_k": k,
                "fmt_uv": self.fmt_uv,
                "fmt_s": self.fmt_s,
                "niter": self.niter,
                "q_oversample": self.q_oversample,
                "residual_scale_alpha": self.residual_scale_alpha,
                "residual_scale_eps": self.residual_scale_eps,
                "residual_scale_q": self.residual_scale_q,
                "residual_center": self.residual_center,
                "scale_factor_format": self._scale_factor_format_name,
                "center_factor_format": self._center_factor_format_name,
            },
            nbytes=nbytes,
        )

    def decompress(
        self, p: Payload, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if len(p.data) == 4:
            qU, qS, qVh, topk_activation = p.data
            scale_factor = None
            center_factor = None
        elif len(p.data) == 5:
            qU, qS, qVh, topk_activation, scale_factor = p.data
            center_factor = None
        elif len(p.data) == 6:
            qU, qS, qVh, topk_activation, scale_factor, center_factor = p.data
        else:
            raise ValueError(f"Unexpected payload format with {len(p.data)} elements")
        U = self._dequantize(qU, device)
        S = self._dequantize(qS, device)
        Vh = self._dequantize(qVh, device)

        k_eff = min(self.rank, U.shape[-1], S.shape[-1], Vh.shape[-2])
        U = U[..., :k_eff]
        S = S[..., :k_eff]
        Vh = Vh[..., :k_eff, :]

        decompressed = (U * S.unsqueeze(-2)) @ Vh
        if scale_factor is not None:
            scale_factor_f32 = scale_factor.to(device=device, dtype=torch.float32)
            decompressed = decompressed * scale_factor_f32.unsqueeze(-2)
        if center_factor is not None:
            center_factor_f32 = center_factor.to(device=device, dtype=torch.float32)
            decompressed = decompressed + center_factor_f32.unsqueeze(-2)
        if topk_activation is not None:
            topk_dense = topk_activation.to(
                device=device, dtype=torch.float32
            ).to_dense()
            decompressed = decompressed + topk_dense

        return decompressed.to(device=device, dtype=dtype)


class OutlierSeparationOnlineSVDCompressor(OnlineSVDCompressor):
    name = "OutlierSeparationOnlineSVD"

    def __init__(
        self,
        rank: int,
        mode: str,
        outlier_ratio: float,
        fmt_uv: str = "fp32",
        fmt_s: str | None = None,
        niter: int = 2,
        q_oversample: int = 0,
        svd_device: str = "auto",
        residual_scale_alpha: float = 0.0,
        residual_scale_eps: float = 1e-6,
        residual_scale_q: float = 0.99,
        scale_factor_format: torch.dtype = torch.float32,
        residual_center: str = "none",
        center_factor_format: torch.dtype = torch.float32,
    ):
        super().__init__(
            rank=rank,
            mode=mode,
            outlier_ratio=outlier_ratio,
            fmt_uv=fmt_uv,
            fmt_s=fmt_s,
            niter=niter,
            q_oversample=q_oversample,
            svd_device=svd_device,
            residual_scale_alpha=residual_scale_alpha,
            residual_scale_eps=residual_scale_eps,
            residual_scale_q=residual_scale_q,
            scale_factor_format=scale_factor_format,
            residual_center=residual_center,
            center_factor_format=center_factor_format,
        )
        self.name = self._build_name("OutlierSeparationOnlineSVD")


class OnlineSVDBitSqueezeCompressor(OnlineSVDCompressor):
    name = "OnlineSVDBitSqueeze"

    def __init__(
        self,
        rank: int,
        mode: str,
        fmt_uv: str,
        fmt_s: str | None = None,
        outlier_ratio: float = 0.0,
        niter: int = 2,
        q_oversample: int = 0,
        svd_device: str = "auto",
        residual_scale_alpha: float = 0.0,
        residual_scale_eps: float = 1e-6,
        residual_scale_q: float = 0.99,
        scale_factor_format: torch.dtype = torch.float32,
        residual_center: str = "none",
        center_factor_format: torch.dtype = torch.float32,
    ):
        super().__init__(
            rank=rank,
            mode=mode,
            outlier_ratio=outlier_ratio,
            fmt_uv=fmt_uv,
            fmt_s=fmt_s,
            niter=niter,
            q_oversample=q_oversample,
            svd_device=svd_device,
            residual_scale_alpha=residual_scale_alpha,
            residual_scale_eps=residual_scale_eps,
            residual_scale_q=residual_scale_q,
            scale_factor_format=scale_factor_format,
            residual_center=residual_center,
            center_factor_format=center_factor_format,
        )
        self.name = self._build_name("OnlineSVDBitSqueeze")
