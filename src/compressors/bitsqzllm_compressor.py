import threading
import weakref
from typing import Any, Optional, Tuple

import torch

from src.compressor import Compressor, Payload


class BitSqueezeLLMCompressor(Compressor):
    name = "BitSqueezeLLM"

    _active_lock = threading.RLock()
    _active_ref: Optional[weakref.ReferenceType] = None

    def __init__(
        self,
        topk_ratio: float = 0.0,
        error_topk_ratio: float = 0.0,
        lowrank_ratio: float = 0.0,
        svd_niters: int = 2,
        svd_uv_format: str = "NF4_DQ",
        svd_s_format: str = "NONE",
        quantization_only_format: str = "NF4",
    ):
        self.topk_ratio = topk_ratio
        self.error_topk_ratio = error_topk_ratio
        self.lowrank_ratio = lowrank_ratio
        self.svd_niters = svd_niters
        self.svd_uv_format = svd_uv_format
        self.svd_s_format = svd_s_format
        self.quantization_only_format = quantization_only_format

        self._codec: Any = None
        self._codec_key: Optional[Tuple[int, int, int]] = None
        self._bitsqueeze_llm: Any = None
        self._printed_info = False

    def compress(self, x: torch.Tensor) -> Payload:
        if x.device.type != "cuda":
            raise RuntimeError("BitSqueezeLLMCompressor requires CUDA tensors")
        if x.dim() < 2:
            raise ValueError("BitSqueezeLLMCompressor requires tensors with at least 2 dimensions")
        if x.numel() == 0 or x.shape[-1] == 0:
            raise ValueError("BitSqueezeLLMCompressor does not support empty tensors")

        orig_shape = tuple(int(dim) for dim in x.shape)
        matrix_shape = (int(x.numel() // x.shape[-1]), int(x.shape[-1]))
        device_index = self._device_index(x)
        matrix = x.to(dtype=torch.float32).reshape(matrix_shape).contiguous()

        with self._active_lock:
            codec = self._ensure_codec_locked(matrix_shape, device_index)
            packed = codec.compress(matrix)
            self._print_info_once(codec, matrix, packed)

        return Payload(
            data=packed,
            meta={
                "orig_shape": orig_shape,
                "matrix_shape": matrix_shape,
                "device_index": device_index,
            },
            nbytes=int(packed.packed_size),
        )

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        orig_shape = tuple(int(dim) for dim in p.meta["orig_shape"])
        matrix_shape = tuple(int(dim) for dim in p.meta["matrix_shape"])
        if len(matrix_shape) != 2:
            raise ValueError(f"Invalid BitSqueezeLLM matrix shape metadata: {matrix_shape}")

        device_index = int(p.meta["device_index"])
        with self._active_lock:
            codec = self._ensure_codec_locked(matrix_shape, device_index)
            restored = codec.decompress(p.data)

        return restored.reshape(orig_shape).to(device=device, dtype=dtype)

    def release(self) -> None:
        with self._active_lock:
            self._release_codec_locked()
            active = self._active_instance_locked()
            if active is self:
                self.__class__._active_ref = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def _ensure_codec_locked(self, matrix_shape: Tuple[int, int], device_index: int):
        active = self._active_instance_locked()
        if active is not None and active is not self:
            active._release_codec_locked()
            self.__class__._active_ref = None

        codec_key = (int(matrix_shape[0]), int(matrix_shape[1]), int(device_index))
        if self._codec is not None and self._codec_key != codec_key:
            self._release_codec_locked()

        if self._codec is None:
            bitsqueeze_llm = self._load_bitsqueeze_llm()
            self._codec = bitsqueeze_llm.BitSqueezeLLM(
                matrix_shape[0],
                matrix_shape[1],
                topk_ratio=self.topk_ratio,
                error_topk_ratio=self.error_topk_ratio,
                lowrank_ratio=self.lowrank_ratio,
                svd_niters=self.svd_niters,
                svd_uv_format=self.svd_uv_format,
                svd_s_format=self.svd_s_format,
                quantization_only_format=self.quantization_only_format,
            )
            self._codec.warmup(device_index)
            self._codec_key = codec_key

        self.__class__._active_ref = weakref.ref(self)
        return self._codec

    def _release_codec_locked(self) -> None:
        codec = self._codec
        self._codec = None
        self._codec_key = None
        if codec is not None:
            codec.release()

    def _load_bitsqueeze_llm(self):
        if self._bitsqueeze_llm is None:
            try:
                import bitsqueeze_llm
            except ImportError as exc:
                raise ImportError(
                    "bitsqueeze_llm is required for BitSqueezeLLMCompressor. "
                    "Install PyBitSqueeze-LLM in the active virtual environment."
                ) from exc
            self._bitsqueeze_llm = bitsqueeze_llm
        return self._bitsqueeze_llm

    def _print_info_once(self, codec, matrix, packed) -> None:
        if self._printed_info:
            return

        bitsqueeze_llm = self._load_bitsqueeze_llm()
        print(f"[bitsqz_llm setting] shape: {matrix.shape}")
        print(f"[bitsqz_llm setting] topk_ratio: {self.topk_ratio}")
        print(f"[bitsqz_llm setting] error_topk_ratio: {self.error_topk_ratio}")
        print(f"[bitsqz_llm setting] lowrank_ratio: {self.lowrank_ratio}")
        print(f"[bitsqz_llm setting] svd_niters: {self.svd_niters}")
        print(f"[bitsqz_llm setting] svd_uv_format: {self.svd_uv_format}")
        print(f"[bitsqz_llm setting] svd_s_format: {self.svd_s_format}")
        print(f"[bitsqz_llm setting] quantization_only_format: {self.quantization_only_format}")
        print(f"[bitsqz_llm setting] Bits per weight: {bitsqueeze_llm.get_bits_per_weight(packed):.4f}")

        latency_info = codec.get_latency_info(10)
        print("[bitsqz_llm setting] Latency info (ms):")
        for key, value in latency_info.items():
            print(f"[bitsqz_llm setting]  {key}: {value:.4f}")

        self._printed_info = True

    @classmethod
    def _active_instance_locked(cls):
        if cls._active_ref is None:
            return None
        return cls._active_ref()

    @staticmethod
    def _device_index(x: torch.Tensor) -> int:
        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return int(device_index)
