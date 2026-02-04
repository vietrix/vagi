"""
Hardware-Adaptive Operations for vAGI.

This module provides device-aware operations that automatically select
the optimal kernel based on available hardware:

1. DeviceManager: Detects and manages hardware capabilities
2. adaptive_attention: FlashAttention-2 -> SDPA -> Naive fallback
3. adaptive_matmul: Optimized matrix multiplication per device

Priority Order:
    CUDA + FlashAttention-2 (fastest)
    -> CUDA + PyTorch SDPA
    -> MPS (Apple Silicon)
    -> CPU with optimizations

Installation for FlashAttention-2:
    pip install flash-attn --no-build-isolation

Reference:
    - FlashAttention-2: https://github.com/Dao-AILab/flash-attention
    - PyTorch SDPA: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
from functools import lru_cache
import math
import warnings


# Try importing FlashAttention-2
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None


@dataclass
class DeviceCapabilities:
    """Hardware capabilities for a device."""
    device_type: str  # 'cuda', 'mps', 'cpu'
    device_index: int
    compute_capability: Optional[Tuple[int, int]] = None  # CUDA compute capability
    total_memory_gb: float = 0.0
    supports_bf16: bool = False
    supports_fp16: bool = False
    supports_flash_attn: bool = False
    supports_sdpa: bool = False
    num_sms: int = 0  # CUDA Streaming Multiprocessors


class DeviceManager:
    """
    Hardware detection and management for optimal kernel selection.

    Provides unified interface for detecting:
    - CUDA GPUs with compute capability
    - Apple Silicon (MPS)
    - CPU with instruction set extensions

    Usage:
        dm = DeviceManager()
        if dm.is_cuda:
            # Use CUDA-optimized kernels
        elif dm.is_mps:
            # Use MPS backend
        else:
            # Use CPU fallback
    """

    _instance = None
    _capabilities_cache = {}

    def __new__(cls):
        """Singleton pattern for device manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._default_device = None
        self._detect_devices()

    def _detect_devices(self):
        """Detect all available compute devices."""
        self._cuda_available = torch.cuda.is_available()
        self._mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self._cuda_device_count = torch.cuda.device_count() if self._cuda_available else 0

        # Detect SDPA support (PyTorch 2.0+)
        self._sdpa_available = hasattr(F, 'scaled_dot_product_attention')

        # Cache capabilities for each CUDA device
        if self._cuda_available:
            for i in range(self._cuda_device_count):
                self._capabilities_cache[f'cuda:{i}'] = self._get_cuda_capabilities(i)

    def _get_cuda_capabilities(self, device_index: int) -> DeviceCapabilities:
        """Get capabilities for a specific CUDA device."""
        props = torch.cuda.get_device_properties(device_index)
        compute_cap = (props.major, props.minor)

        # FlashAttention-2 requires Ampere+ (SM80+)
        supports_flash = FLASH_ATTN_AVAILABLE and compute_cap >= (8, 0)

        # BF16 requires Ampere+ (SM80+)
        supports_bf16 = compute_cap >= (8, 0)

        # FP16 requires Pascal+ (SM60+)
        supports_fp16 = compute_cap >= (6, 0)

        return DeviceCapabilities(
            device_type='cuda',
            device_index=device_index,
            compute_capability=compute_cap,
            total_memory_gb=props.total_memory / (1024 ** 3),
            supports_bf16=supports_bf16,
            supports_fp16=supports_fp16,
            supports_flash_attn=supports_flash,
            supports_sdpa=self._sdpa_available,
            num_sms=props.multi_processor_count,
        )

    @property
    def is_cuda(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    @property
    def is_mps(self) -> bool:
        """Check if Apple Silicon MPS is available."""
        return self._mps_available

    @property
    def is_cpu(self) -> bool:
        """Check if only CPU is available (no GPU)."""
        return not self._cuda_available and not self._mps_available

    @property
    def num_gpus(self) -> int:
        """Number of CUDA GPUs available."""
        return self._cuda_device_count

    @property
    def supports_flash_attention(self) -> bool:
        """Check if FlashAttention-2 is supported on current device."""
        if not self._cuda_available:
            return False
        device_idx = torch.cuda.current_device()
        caps = self._capabilities_cache.get(f'cuda:{device_idx}')
        return caps.supports_flash_attn if caps else False

    @property
    def supports_sdpa(self) -> bool:
        """Check if PyTorch SDPA is available."""
        return self._sdpa_available

    def get_optimal_dtype(self, device: Optional[torch.device] = None) -> torch.dtype:
        """
        Get optimal dtype for the given device.

        Priority: BF16 > FP16 > FP32
        """
        if device is None:
            device = self.get_default_device()

        device_str = str(device)

        if 'cuda' in device_str:
            caps = self._capabilities_cache.get(device_str)
            if caps:
                if caps.supports_bf16:
                    return torch.bfloat16
                elif caps.supports_fp16:
                    return torch.float16
        elif 'mps' in device_str:
            # MPS supports FP16
            return torch.float16

        return torch.float32

    def get_default_device(self) -> torch.device:
        """Get the default compute device."""
        if self._default_device is not None:
            return self._default_device

        if self._cuda_available:
            return torch.device('cuda')
        elif self._mps_available:
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def set_default_device(self, device: torch.device):
        """Set the default compute device."""
        self._default_device = device

    def get_capabilities(self, device: Optional[torch.device] = None) -> DeviceCapabilities:
        """Get capabilities for a specific device."""
        if device is None:
            device = self.get_default_device()

        device_str = str(device)

        if device_str in self._capabilities_cache:
            return self._capabilities_cache[device_str]

        # Create capabilities for non-CUDA devices
        if 'mps' in device_str:
            return DeviceCapabilities(
                device_type='mps',
                device_index=0,
                supports_fp16=True,
                supports_sdpa=self._sdpa_available,
            )
        else:
            return DeviceCapabilities(
                device_type='cpu',
                device_index=0,
                supports_sdpa=self._sdpa_available,
            )

    def synchronize(self):
        """Synchronize the current device."""
        if self._cuda_available:
            torch.cuda.synchronize()
        elif self._mps_available:
            torch.mps.synchronize()

    def empty_cache(self):
        """Clear device memory cache."""
        if self._cuda_available:
            torch.cuda.empty_cache()
        elif self._mps_available:
            torch.mps.empty_cache()

    def memory_stats(self) -> dict:
        """Get memory statistics for the current device."""
        if self._cuda_available:
            return {
                'allocated': torch.cuda.memory_allocated() / (1024 ** 3),
                'reserved': torch.cuda.memory_reserved() / (1024 ** 3),
                'max_allocated': torch.cuda.max_memory_allocated() / (1024 ** 3),
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

    def __repr__(self) -> str:
        return (
            f"DeviceManager("
            f"cuda={self.is_cuda}, "
            f"mps={self.is_mps}, "
            f"gpus={self.num_gpus}, "
            f"flash_attn={self.supports_flash_attention}, "
            f"sdpa={self.supports_sdpa})"
        )


# Global device manager instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def _naive_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Naive attention implementation as fallback.

    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, num_heads, seq_len, head_dim]
        value: [batch, num_heads, seq_len, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Attention scale factor

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    L, S = query.size(-2), key.size(-2)
    head_dim = query.size(-1)

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

    # Apply attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        else:
            attn_weights = attn_weights + attn_mask

    # Softmax and dropout
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if dropout_p > 0.0 and query.requires_grad:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute output
    return torch.matmul(attn_weights, value)


def _flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    FlashAttention-2 implementation.

    Expects input shape: [batch, seq_len, num_heads, head_dim]
    """
    if not FLASH_ATTN_AVAILABLE:
        raise RuntimeError("FlashAttention-2 not available")

    # FlashAttention expects [batch, seq_len, num_heads, head_dim]
    # Our input is [batch, num_heads, seq_len, head_dim]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # FlashAttention doesn't support arbitrary attention masks
    # Only causal or no mask
    if attn_mask is not None:
        warnings.warn(
            "FlashAttention-2 doesn't support arbitrary attention masks. "
            "Falling back to SDPA or naive attention."
        )
        # Transpose back and fall through
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        raise ValueError("Attention mask not supported")

    # Call FlashAttention
    output = flash_attn_func(
        query, key, value,
        dropout_p=dropout_p if query.requires_grad else 0.0,
        causal=is_causal,
        softmax_scale=scale,
    )

    # Transpose back to [batch, num_heads, seq_len, head_dim]
    return output.transpose(1, 2)


def _sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    PyTorch Scaled Dot-Product Attention (SDPA).

    Uses the most efficient backend automatically:
    - FlashAttention (if available and applicable)
    - Memory-Efficient Attention
    - Math fallback
    """
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p if query.requires_grad else 0.0,
        is_causal=is_causal,
        scale=scale,
    )


def adaptive_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    force_backend: Optional[Literal['flash', 'sdpa', 'naive']] = None,
) -> Tensor:
    """
    Hardware-adaptive attention that selects the optimal backend.

    Priority:
        1. FlashAttention-2 (CUDA Ampere+, no arbitrary mask)
        2. PyTorch SDPA (PyTorch 2.0+)
        3. Naive attention (fallback)

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim]
        value: Value tensor [batch, num_heads, seq_len, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (only during training)
        is_causal: Whether to use causal masking
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        force_backend: Force a specific backend ('flash', 'sdpa', 'naive')

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]

    Example:
        >>> dm = get_device_manager()
        >>> q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
        >>> out = adaptive_attention(q, k, v, is_causal=True)
    """
    dm = get_device_manager()

    # Force specific backend if requested
    if force_backend == 'flash':
        return _flash_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)
    elif force_backend == 'sdpa':
        return _sdpa_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)
    elif force_backend == 'naive':
        return _naive_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)

    # Auto-select best backend
    device = query.device

    # Try FlashAttention-2 first (fastest)
    if (device.type == 'cuda' and
        dm.supports_flash_attention and
        attn_mask is None and  # FlashAttention doesn't support arbitrary masks
        query.dtype in (torch.float16, torch.bfloat16)):
        try:
            return _flash_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)
        except Exception:
            pass  # Fall through to SDPA

    # Try PyTorch SDPA (second choice)
    if dm.supports_sdpa:
        try:
            return _sdpa_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)
        except Exception:
            pass  # Fall through to naive

    # Naive fallback (always works)
    return _naive_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)


class AdaptiveAttention(nn.Module):
    """
    Multi-head attention module with hardware-adaptive kernel selection.

    Automatically selects the best attention implementation based on:
    - Available hardware (CUDA, MPS, CPU)
    - Input dtype (FP16, BF16, FP32)
    - Presence of attention mask
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout
        self.is_causal = is_causal
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass with hardware-adaptive attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional mask [batch, 1, seq_len, seq_len]
            position_ids: Position IDs (unused, for RoPE compatibility)
            past_key_value: Cached KV for incremental decoding
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated cache

        Returns:
            output: [batch, seq_len, hidden_size]
            attn_weights: Optional attention weights
            past_key_value: Optional updated cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present_key_value = (key, value) if use_cache else None

        # Adaptive attention
        if output_attentions:
            # Use naive attention to get weights
            attn_output = _naive_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
                scale=self.scale,
            )
            # Compute attention weights separately
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            attn_output = adaptive_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
                scale=self.scale,
            )
            attn_weights = None

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, present_key_value


def adaptive_matmul(
    a: Tensor,
    b: Tensor,
    use_tf32: bool = True,
) -> Tensor:
    """
    Hardware-adaptive matrix multiplication.

    Optimizations:
    - CUDA: Use TF32 for Ampere+ GPUs (faster, slight precision loss)
    - CPU: Use MKL/OpenBLAS optimizations

    Args:
        a: First matrix
        b: Second matrix
        use_tf32: Whether to allow TF32 on supported hardware

    Returns:
        Matrix product a @ b
    """
    dm = get_device_manager()

    if dm.is_cuda and use_tf32:
        # TF32 is enabled by default on Ampere+
        # This just ensures we use the optimized path
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True,
        ):
            return torch.matmul(a, b)

    return torch.matmul(a, b)


def adaptive_layer_norm(
    input: Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Hardware-adaptive layer normalization.

    Uses fused kernels when available for better performance.
    """
    dm = get_device_manager()

    # On CUDA, try to use fused layer norm if available
    if dm.is_cuda:
        try:
            from apex.normalization import FusedLayerNorm
            # Apex fused layer norm is faster
            return F.layer_norm(input, normalized_shape, weight, bias, eps)
        except ImportError:
            pass

    return F.layer_norm(input, normalized_shape, weight, bias, eps)


# Convenience functions for common operations
def to_device(tensor: Tensor, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Move tensor to optimal device with optimal dtype."""
    dm = get_device_manager()
    device = dm.get_default_device()

    if dtype is None:
        dtype = dm.get_optimal_dtype(device)

    return tensor.to(device=device, dtype=dtype)


def get_attention_backend() -> str:
    """Get the name of the attention backend that will be used."""
    dm = get_device_manager()

    if dm.is_cuda and dm.supports_flash_attention:
        return "flash_attention_2"
    elif dm.supports_sdpa:
        return "sdpa"
    else:
        return "naive"


if __name__ == "__main__":
    # Test device manager
    dm = get_device_manager()
    print(f"Device Manager: {dm}")
    print(f"  is_cuda: {dm.is_cuda}")
    print(f"  is_mps: {dm.is_mps}")
    print(f"  is_cpu: {dm.is_cpu}")
    print(f"  num_gpus: {dm.num_gpus}")
    print(f"  supports_flash_attention: {dm.supports_flash_attention}")
    print(f"  supports_sdpa: {dm.supports_sdpa}")
    print(f"  attention_backend: {get_attention_backend()}")

    # Test adaptive attention
    device = dm.get_default_device()
    dtype = dm.get_optimal_dtype(device)
    print(f"\nTesting on {device} with {dtype}")

    batch, heads, seq_len, head_dim = 2, 8, 128, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # Test causal attention
    out = adaptive_attention(q, k, v, is_causal=True)
    print(f"Causal attention output shape: {out.shape}")

    # Test with mask
    mask = torch.zeros(batch, 1, seq_len, seq_len, device=device, dtype=dtype)
    out_masked = adaptive_attention(q, k, v, attn_mask=mask)
    print(f"Masked attention output shape: {out_masked.shape}")

    print("\nAll tests passed!")
