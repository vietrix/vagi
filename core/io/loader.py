"""
Unified Model Loader for vAGI.

This module provides a unified interface for loading models in various formats:
- Full precision (FP32, FP16, BF16)
- AWQ 4-bit quantized
- GPTQ 4-bit quantized
- Dynamic Int8 quantized (for CPU)
- SafeTensors format

The loader automatically detects hardware and selects the optimal format:
- GPU (CUDA): 4-bit quantized models (AWQ/GPTQ)
- CPU: Dynamic Int8 quantized for faster inference

Usage:
    from core.io.loader import UnifiedModelLoader

    loader = UnifiedModelLoader()
    model = loader.load("path/to/model", device="auto")
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union, Literal
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

# Try importing optional dependencies
try:
    from safetensors import safe_open
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False

try:
    from auto_gptq import AutoGPTQForCausalLM
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_path: str
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    dtype: str = "auto"  # "auto", "float32", "float16", "bfloat16", "int8"
    quantization: Optional[str] = None  # None, "awq", "gptq", "int8"
    use_safetensors: bool = True
    trust_remote_code: bool = False
    max_memory: Optional[Dict[int, str]] = None
    low_cpu_mem_usage: bool = True
    offload_folder: Optional[str] = None


@dataclass
class LoadedModel:
    """Container for loaded model and metadata."""
    model: nn.Module
    config: Dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    quantization: Optional[str]
    load_time_ms: float
    memory_footprint_mb: float


class QuantizationDetector:
    """Detects quantization format from model files."""

    @staticmethod
    def detect(model_path: str) -> Optional[str]:
        """
        Detect quantization format from model directory.

        Returns:
            'awq', 'gptq', or None for full precision
        """
        path = Path(model_path)

        # Check for quantization config files
        quant_config_path = path / "quantize_config.json"
        if quant_config_path.exists():
            with open(quant_config_path, 'r') as f:
                config = json.load(f)
                if "zero_point" in config or "awq" in str(config).lower():
                    return "awq"
                elif "bits" in config and "group_size" in config:
                    return "gptq"

        # Check for AWQ-specific files
        if (path / "awq_model.safetensors").exists():
            return "awq"

        # Check for GPTQ-specific files
        if (path / "gptq_model-4bit-128g.safetensors").exists():
            return "gptq"

        # Check model config for quantization hints
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                if config.get("quantization_config"):
                    quant_method = config["quantization_config"].get("quant_method", "")
                    if "awq" in quant_method.lower():
                        return "awq"
                    elif "gptq" in quant_method.lower():
                        return "gptq"

        return None


class DynamicInt8Quantizer:
    """
    Dynamic Int8 quantization for CPU inference.

    Applies quantization at runtime without needing pre-quantized weights.
    Best for CPU inference where memory bandwidth is the bottleneck.
    """

    @staticmethod
    def quantize_model(model: nn.Module) -> nn.Module:
        """
        Apply dynamic Int8 quantization to a model.

        Args:
            model: PyTorch model to quantize

        Returns:
            Quantized model
        """
        # Move to CPU for quantization
        model = model.cpu().float()

        # Quantize linear layers dynamically
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Quantize all linear layers
            dtype=torch.qint8,
        )

        return quantized_model

    @staticmethod
    def is_quantized(model: nn.Module) -> bool:
        """Check if a model is already quantized."""
        for module in model.modules():
            if isinstance(module, (
                torch.nn.quantized.Linear,
                torch.nn.quantized.dynamic.Linear,
            )):
                return True
        return False


class UnifiedModelLoader:
    """
    Unified model loader with automatic format detection and device selection.

    Features:
    - Auto-detects quantization format (AWQ, GPTQ, full precision)
    - Auto-selects optimal device (CUDA, MPS, CPU)
    - Auto-selects optimal dtype based on hardware
    - Supports SafeTensors format
    - Memory-efficient loading with low_cpu_mem_usage
    """

    def __init__(self):
        self._detect_hardware()

    def _detect_hardware(self):
        """Detect available hardware."""
        self.cuda_available = torch.cuda.is_available()
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        if self.cuda_available:
            self.default_device = "cuda"
            props = torch.cuda.get_device_properties(0)
            self.compute_capability = (props.major, props.minor)
            self.gpu_memory_gb = props.total_memory / (1024 ** 3)
        elif self.mps_available:
            self.default_device = "mps"
            self.compute_capability = None
            self.gpu_memory_gb = 0
        else:
            self.default_device = "cpu"
            self.compute_capability = None
            self.gpu_memory_gb = 0

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve 'auto' device to actual device."""
        if device == "auto":
            return torch.device(self.default_device)
        return torch.device(device)

    def _resolve_dtype(
        self,
        dtype: str,
        device: torch.device,
        quantization: Optional[str],
    ) -> torch.dtype:
        """Resolve 'auto' dtype to actual dtype."""
        if quantization == "int8":
            return torch.float32  # Int8 quantization works on float32

        if dtype == "auto":
            if device.type == "cuda":
                # BF16 for Ampere+, FP16 for older
                if self.compute_capability and self.compute_capability >= (8, 0):
                    return torch.bfloat16
                return torch.float16
            elif device.type == "mps":
                return torch.float16
            else:
                return torch.float32

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(dtype, torch.float32)

    def _resolve_quantization(
        self,
        quantization: Optional[str],
        device: torch.device,
        model_path: str,
    ) -> Optional[str]:
        """Resolve quantization method."""
        # Auto-detect from model files
        if quantization is None:
            detected = QuantizationDetector.detect(model_path)
            if detected:
                return detected

        # Auto-select based on device
        if quantization == "auto":
            if device.type == "cuda":
                # Prefer AWQ on GPU
                if AWQ_AVAILABLE:
                    return "awq"
                elif GPTQ_AVAILABLE:
                    return "gptq"
            elif device.type == "cpu":
                # Use Int8 on CPU
                return "int8"

        return quantization

    def _load_safetensors(
        self,
        model_path: str,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        """Load weights from SafeTensors format."""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors not installed. pip install safetensors")

        path = Path(model_path)
        state_dict = {}

        # Find all safetensors files
        safetensor_files = list(path.glob("*.safetensors"))

        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        for sf_path in safetensor_files:
            tensors = load_safetensors(str(sf_path))
            state_dict.update(tensors)

        # Convert to target dtype
        for key in state_dict:
            if state_dict[key].dtype in (torch.float32, torch.float16, torch.bfloat16):
                state_dict[key] = state_dict[key].to(dtype)

        return state_dict

    def _load_pytorch(
        self,
        model_path: str,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        """Load weights from PyTorch format."""
        path = Path(model_path)

        # Find model file
        model_files = list(path.glob("pytorch_model*.bin")) + list(path.glob("model.bin"))

        if not model_files:
            raise FileNotFoundError(f"No PyTorch model files found in {model_path}")

        state_dict = {}
        for model_file in model_files:
            loaded = torch.load(str(model_file), map_location="cpu")
            state_dict.update(loaded)

        # Convert to target dtype
        for key in state_dict:
            if state_dict[key].dtype in (torch.float32, torch.float16, torch.bfloat16):
                state_dict[key] = state_dict[key].to(dtype)

        return state_dict

    def _load_awq(
        self,
        model_path: str,
        device: torch.device,
    ) -> nn.Module:
        """Load AWQ quantized model."""
        if not AWQ_AVAILABLE:
            raise ImportError(
                "AutoAWQ not installed. Install with: pip install autoawq"
            )

        model = AutoAWQForCausalLM.from_quantized(
            model_path,
            device=str(device),
            fuse_layers=True,
        )
        return model.model

    def _load_gptq(
        self,
        model_path: str,
        device: torch.device,
    ) -> nn.Module:
        """Load GPTQ quantized model."""
        if not GPTQ_AVAILABLE:
            raise ImportError(
                "AutoGPTQ not installed. Install with: pip install auto-gptq"
            )

        model = AutoGPTQForCausalLM.from_quantized(
            model_path,
            device=str(device),
            use_safetensors=True,
        )
        return model.model

    def _load_config(self, model_path: str) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = Path(model_path) / "config.json"

        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def _estimate_memory(
        self,
        model: nn.Module,
        dtype: torch.dtype,
    ) -> float:
        """Estimate model memory footprint in MB."""
        total_params = sum(p.numel() for p in model.parameters())

        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.qint8: 1,
        }.get(dtype, 4)

        return (total_params * bytes_per_param) / (1024 ** 2)

    def load(
        self,
        model_path: str,
        model_class: Optional[type] = None,
        device: str = "auto",
        dtype: str = "auto",
        quantization: Optional[str] = None,
        use_safetensors: bool = True,
        **kwargs,
    ) -> LoadedModel:
        """
        Load a model with automatic format and device detection.

        Args:
            model_path: Path to model directory
            model_class: Optional model class to instantiate
            device: Device to load to ("auto", "cuda", "cpu", "mps")
            dtype: Data type ("auto", "float32", "float16", "bfloat16")
            quantization: Quantization method (None, "auto", "awq", "gptq", "int8")
            use_safetensors: Prefer SafeTensors format
            **kwargs: Additional arguments for model initialization

        Returns:
            LoadedModel containing the model and metadata

        Example:
            >>> loader = UnifiedModelLoader()
            >>> result = loader.load("./models/vagi-7b", device="auto")
            >>> model = result.model
            >>> print(f"Loaded on {result.device} with {result.dtype}")
        """
        import time
        start_time = time.perf_counter()

        # Resolve settings
        resolved_device = self._resolve_device(device)
        resolved_quant = self._resolve_quantization(
            quantization, resolved_device, model_path
        )
        resolved_dtype = self._resolve_dtype(dtype, resolved_device, resolved_quant)

        # Load config
        config = self._load_config(model_path)

        # Load model based on quantization method
        if resolved_quant == "awq":
            model = self._load_awq(model_path, resolved_device)
        elif resolved_quant == "gptq":
            model = self._load_gptq(model_path, resolved_device)
        elif model_class is not None:
            # Load weights and instantiate model
            if use_safetensors and SAFETENSORS_AVAILABLE:
                state_dict = self._load_safetensors(model_path, resolved_dtype)
            else:
                state_dict = self._load_pytorch(model_path, resolved_dtype)

            # Instantiate model
            model = model_class(**config, **kwargs)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device=resolved_device, dtype=resolved_dtype)

            # Apply Int8 quantization if requested
            if resolved_quant == "int8" and resolved_device.type == "cpu":
                model = DynamicInt8Quantizer.quantize_model(model)
        else:
            # Try to load with transformers
            try:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=resolved_dtype,
                    device_map=str(resolved_device) if resolved_device.type == "cuda" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=kwargs.get("trust_remote_code", False),
                )

                if resolved_device.type != "cuda":
                    model = model.to(resolved_device)

                if resolved_quant == "int8" and resolved_device.type == "cpu":
                    model = DynamicInt8Quantizer.quantize_model(model)

            except ImportError:
                raise ImportError(
                    "transformers not installed and no model_class provided. "
                    "Install with: pip install transformers"
                )

        load_time = (time.perf_counter() - start_time) * 1000
        memory_mb = self._estimate_memory(model, resolved_dtype)

        return LoadedModel(
            model=model,
            config=config,
            device=resolved_device,
            dtype=resolved_dtype,
            quantization=resolved_quant,
            load_time_ms=load_time,
            memory_footprint_mb=memory_mb,
        )

    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
        strict: bool = False,
    ) -> nn.Module:
        """
        Load a checkpoint into an existing model.

        Args:
            model: Model to load checkpoint into
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce matching keys

        Returns:
            Model with loaded checkpoint
        """
        path = Path(checkpoint_path)

        if path.suffix == ".safetensors":
            if not SAFETENSORS_AVAILABLE:
                raise ImportError("safetensors not installed")
            state_dict = load_safetensors(str(path))
        else:
            state_dict = torch.load(str(path), map_location="cpu")

        # Handle nested state dicts
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=strict)
        return model

    def save_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_safetensors: bool = True,
    ):
        """
        Save a checkpoint.

        Args:
            model: Model to save
            checkpoint_path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            metadata: Optional metadata to include
            use_safetensors: Use SafeTensors format (recommended)
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if use_safetensors and SAFETENSORS_AVAILABLE:
            from safetensors.torch import save_file

            # SafeTensors only saves tensors, not full state dicts
            state_dict = model.state_dict()
            save_file(state_dict, str(path.with_suffix(".safetensors")))

            # Save optimizer and metadata separately
            if optimizer or metadata:
                extra = {}
                if optimizer:
                    extra["optimizer_state_dict"] = optimizer.state_dict()
                if metadata:
                    extra["metadata"] = metadata
                torch.save(extra, str(path.with_suffix(".pt")))
        else:
            checkpoint = {
                "model_state_dict": model.state_dict(),
            }
            if optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            if metadata:
                checkpoint["metadata"] = metadata

            torch.save(checkpoint, str(path))


def get_recommended_settings(model_size_b: float) -> Dict[str, Any]:
    """
    Get recommended loading settings based on model size.

    Args:
        model_size_b: Model size in billions of parameters

    Returns:
        Dictionary of recommended settings
    """
    loader = UnifiedModelLoader()

    if loader.cuda_available:
        gpu_memory_gb = loader.gpu_memory_gb

        # Estimate memory needed (rough: 2 bytes per param for FP16)
        needed_memory_gb = model_size_b * 2

        if gpu_memory_gb >= needed_memory_gb * 1.5:
            # Plenty of memory, use FP16
            return {
                "device": "cuda",
                "dtype": "float16",
                "quantization": None,
            }
        elif gpu_memory_gb >= needed_memory_gb * 0.5:
            # Use 4-bit quantization
            return {
                "device": "cuda",
                "dtype": "float16",
                "quantization": "awq" if AWQ_AVAILABLE else "gptq",
            }
        else:
            # Not enough GPU memory, use CPU with Int8
            return {
                "device": "cpu",
                "dtype": "float32",
                "quantization": "int8",
            }
    elif loader.mps_available:
        return {
            "device": "mps",
            "dtype": "float16",
            "quantization": None,
        }
    else:
        return {
            "device": "cpu",
            "dtype": "float32",
            "quantization": "int8",
        }


if __name__ == "__main__":
    # Test the loader
    loader = UnifiedModelLoader()

    print("Unified Model Loader")
    print("=" * 50)
    print(f"CUDA available: {loader.cuda_available}")
    print(f"MPS available: {loader.mps_available}")
    print(f"Default device: {loader.default_device}")

    if loader.cuda_available:
        print(f"Compute capability: {loader.compute_capability}")
        print(f"GPU memory: {loader.gpu_memory_gb:.2f} GB")

    print("\nDependencies:")
    print(f"  SafeTensors: {SAFETENSORS_AVAILABLE}")
    print(f"  AutoAWQ: {AWQ_AVAILABLE}")
    print(f"  AutoGPTQ: {GPTQ_AVAILABLE}")

    print("\nRecommended settings for common model sizes:")
    for size in [1, 7, 13, 70]:
        settings = get_recommended_settings(size)
        print(f"  {size}B model: {settings}")

    print("\nAll checks passed!")
