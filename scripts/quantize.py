#!/usr/bin/env python3
"""
Model Quantization Script for vAGI Edge Deployment.

Implements 4-bit quantization using AWQ (Activation-aware Weight Quantization)
or GPTQ for efficient inference on edge devices.

AWQ Advantages:
- Preserves salient weights based on activation patterns
- Better accuracy than naive quantization
- Fast inference with optimized kernels
- Compatible with Transformers/vLLM/TGI

Usage:
    # Quantize with AWQ (recommended)
    python scripts/quantize.py --model checkpoints/model.pt \
        --output checkpoints/model_4bit --method awq

    # Quantize with GPTQ
    python scripts/quantize.py --model gpt2 \
        --output checkpoints/gpt2_4bit --method gptq

    # Custom calibration data
    python scripts/quantize.py --model checkpoints/model.pt \
        --output checkpoints/model_4bit --calibration data/vietnamese_samples.jsonl

Output Formats:
    - .safetensors: Recommended for production (safe, fast loading)
    - .pt: PyTorch native format
    - HuggingFace compatible directory structure
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    # Quantization method
    method: str = "awq"  # "awq", "gptq", "bitsandbytes"

    # Bit width
    bits: int = 4
    group_size: int = 128  # AWQ/GPTQ group size

    # AWQ-specific
    awq_version: str = "gemm"  # "gemm" or "gemv"
    awq_zero_point: bool = True

    # GPTQ-specific
    gptq_damp_percent: float = 0.01
    gptq_desc_act: bool = False
    gptq_sym: bool = True

    # Calibration
    calibration_samples: int = 128
    calibration_seq_len: int = 512

    # Output
    save_format: str = "safetensors"  # "safetensors", "pt", "both"


# ============================================================================
# Calibration Data Generation
# ============================================================================

def load_calibration_data(
    data_path: Optional[str],
    tokenizer: Any,
    num_samples: int = 128,
    seq_len: int = 512,
) -> List[torch.Tensor]:
    """
    Load or generate calibration data for quantization.

    AWQ and GPTQ require sample inputs to determine optimal quantization
    parameters based on activation patterns.

    Args:
        data_path: Path to JSONL file with text samples
        tokenizer: Tokenizer for encoding text
        num_samples: Number of calibration samples
        seq_len: Sequence length for each sample

    Returns:
        List of input tensors for calibration
    """
    calibration_data = []

    if data_path and Path(data_path).exists():
        logger.info(f"Loading calibration data from {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            texts = []
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # Support various formats
                        text = item.get('text') or item.get('input', '') + ' ' + item.get('output', '')
                        if text.strip():
                            texts.append(text.strip())
                    except json.JSONDecodeError:
                        texts.append(line.strip())

                if len(texts) >= num_samples:
                    break

        for text in texts[:num_samples]:
            try:
                if hasattr(tokenizer, 'encode'):
                    ids = tokenizer.encode(text, max_length=seq_len, truncation=True)
                    if hasattr(ids, 'ids'):
                        ids = ids.ids
                else:
                    ids = tokenizer(text, max_length=seq_len, truncation=True)['input_ids']

                # Pad or truncate to seq_len
                if len(ids) < seq_len:
                    ids = ids + [tokenizer.pad_token_id or 0] * (seq_len - len(ids))
                ids = ids[:seq_len]

                calibration_data.append(torch.tensor(ids, dtype=torch.long))
            except Exception as e:
                logger.warning(f"Failed to encode text: {e}")

    # Generate synthetic data if needed
    if len(calibration_data) < num_samples:
        logger.info(f"Generating {num_samples - len(calibration_data)} synthetic samples")
        vocab_size = getattr(tokenizer, 'vocab_size', 50000)

        for _ in range(num_samples - len(calibration_data)):
            # Generate random but semi-coherent token sequences
            ids = torch.randint(100, min(vocab_size, 10000), (seq_len,))
            calibration_data.append(ids)

    logger.info(f"Loaded {len(calibration_data)} calibration samples")
    return calibration_data


def generate_vietnamese_calibration_samples() -> List[str]:
    """Generate Vietnamese calibration samples for optimal quantization."""
    samples = [
        # Technical Vietnamese
        "Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ đòi hỏi trí thông minh của con người.",
        "Mạng nơ-ron sâu (deep neural network) là một kiến trúc học máy gồm nhiều lớp ẩn, cho phép mô hình học các biểu diễn phức tạp từ dữ liệu.",
        "Thuật toán gradient descent là phương pháp tối ưu hóa được sử dụng rộng rãi để huấn luyện các mô hình học máy bằng cách điều chỉnh trọng số theo hướng giảm hàm mất mát.",

        # General Vietnamese
        "Việt Nam là một quốc gia nằm ở Đông Nam Á, có đường bờ biển dài hơn 3.000 km dọc theo Biển Đông.",
        "Hà Nội là thủ đô của Việt Nam, nổi tiếng với kiến trúc cổ kính và văn hóa nghìn năm văn hiến.",
        "Ẩm thực Việt Nam nổi tiếng thế giới với các món phở, bánh mì, bún chả và nhiều đặc sản vùng miền khác.",

        # Mathematical reasoning
        "<think>[Bước 1] Để giải phương trình bậc hai ax² + bx + c = 0, ta sử dụng công thức nghiệm.</think>",
        "<think>[Bước 2] Công thức: x = (-b ± √(b² - 4ac)) / (2a), trong đó Δ = b² - 4ac là delta.</think>",

        # Code with Vietnamese comments
        "def tinh_giai_thua(n):\n    '''Tính giai thừa của n'''\n    if n <= 1:\n        return 1\n    return n * tinh_giai_thua(n - 1)",

        # Mixed content
        "Công thức Euler: e^(iπ) + 1 = 0, được coi là công thức đẹp nhất trong toán học vì nó kết nối 5 hằng số quan trọng: e, i, π, 1, và 0.",
    ]

    # Expand samples
    expanded = []
    for sample in samples:
        expanded.append(sample)
        # Add variations
        expanded.append(sample.lower())
        expanded.append(sample.upper()[:len(sample)//2] + sample[len(sample)//2:])

    return expanded[:128]


# ============================================================================
# AWQ Quantization
# ============================================================================

class AWQQuantizer:
    """
    AWQ (Activation-aware Weight Quantization) implementation.

    AWQ identifies salient weights (important for accuracy) based on
    activation patterns and preserves them at higher precision.

    Key insight: A small fraction of weights are critical for model
    accuracy. AWQ finds these by analyzing activations and applies
    per-channel scaling to protect them.

    Algorithm:
    1. Run calibration data through model
    2. Compute activation statistics per channel
    3. Identify salient channels (high activation variance)
    4. Apply scaling factors to protect salient weights
    5. Quantize with group-wise quantization
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.bits = config.bits
        self.group_size = config.group_size

    def _find_scales(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find optimal per-channel scales based on activations.

        The scale factor protects salient weights during quantization.
        Higher activation variance = more important = higher scale.
        """
        # Compute activation statistics
        act_mean = activations.mean(dim=0)
        act_std = activations.std(dim=0)

        # Salient channels have high variance relative to mean
        saliency = act_std / (act_mean.abs() + 1e-8)

        # Compute scales: higher saliency = lower scale (less quantization error)
        # This is inverse because we divide weights by scale before quantization
        scales = torch.clamp(saliency, min=1e-4, max=1.0)

        return scales

    def _quantize_weight(
        self,
        weight: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weight tensor to n-bit integers.

        Returns:
            quantized: Quantized weight (int)
            scale: Quantization scale
            zero_point: Quantization zero point
        """
        # Apply channel scales if provided
        if scales is not None:
            weight = weight * scales.unsqueeze(0)

        # Group-wise quantization
        orig_shape = weight.shape
        weight_flat = weight.view(-1, self.group_size)

        # Compute min/max per group
        w_min = weight_flat.min(dim=1, keepdim=True)[0]
        w_max = weight_flat.max(dim=1, keepdim=True)[0]

        # Compute scale and zero point
        max_int = 2 ** self.bits - 1
        scale = (w_max - w_min) / max_int
        scale = torch.clamp(scale, min=1e-8)

        if self.config.awq_zero_point:
            zero_point = (-w_min / scale).round().clamp(0, max_int)
        else:
            zero_point = torch.zeros_like(scale)

        # Quantize
        quantized = ((weight_flat - w_min) / scale).round().clamp(0, max_int).to(torch.int8)

        # Reshape
        quantized = quantized.view(orig_shape)
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)

        return quantized, scale, zero_point

    def quantize_linear(
        self,
        module: nn.Linear,
        activations: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize a linear layer.

        Returns dict with quantized weights and metadata.
        """
        weight = module.weight.data.float()

        # Find scales from activations
        if activations is not None:
            scales = self._find_scales(weight, activations)
        else:
            scales = None

        # Quantize
        q_weight, scale, zero_point = self._quantize_weight(weight, scales)

        result = {
            "qweight": q_weight,
            "scales": scale,
            "qzeros": zero_point,
            "g_idx": torch.arange(weight.shape[0], dtype=torch.int32),
        }

        if module.bias is not None:
            result["bias"] = module.bias.data

        return result


# ============================================================================
# GPTQ Quantization
# ============================================================================

class GPTQQuantizer:
    """
    GPTQ (Gradient-based Post-Training Quantization) implementation.

    GPTQ uses Hessian information to determine optimal quantization order
    and applies iterative error compensation.

    Algorithm:
    1. Compute Hessian diagonal from calibration data
    2. Quantize weights in order of lowest Hessian (least impact)
    3. Update remaining weights to compensate for quantization error
    4. Repeat until all weights quantized
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.bits = config.bits
        self.group_size = config.group_size
        self.damp_percent = config.gptq_damp_percent

    def quantize_linear(
        self,
        module: nn.Linear,
        activations: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize linear layer using GPTQ algorithm.
        """
        weight = module.weight.data.float()
        out_features, in_features = weight.shape

        # Compute Hessian diagonal approximation: H = X^T X
        # activations: [num_samples * seq_len, in_features]
        H = torch.matmul(activations.T, activations)
        H = H / activations.shape[0]

        # Add damping for numerical stability
        damp = self.damp_percent * torch.diag(H).mean()
        H.diagonal().add_(damp)

        # Cholesky decomposition for efficient inverse
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            # Fallback if Cholesky fails
            logger.warning("Cholesky failed, using pseudo-inverse")
            H_inv = torch.linalg.pinv(H)

        # Quantization
        max_int = 2 ** self.bits - 1
        w_min = weight.min(dim=1, keepdim=True)[0]
        w_max = weight.max(dim=1, keepdim=True)[0]
        scale = (w_max - w_min) / max_int
        scale = torch.clamp(scale, min=1e-8)

        if self.config.gptq_sym:
            zero_point = torch.zeros_like(scale)
            w_max_abs = weight.abs().max(dim=1, keepdim=True)[0]
            scale = w_max_abs / (max_int // 2)
        else:
            zero_point = (-w_min / scale).round().clamp(0, max_int)

        # Quantize with error compensation
        q_weight = ((weight - w_min) / scale).round().clamp(0, max_int)

        # Error compensation (simplified)
        quant_error = weight - (q_weight * scale + w_min)
        compensation = torch.matmul(quant_error, H_inv)

        # Apply compensation to scale
        scale = scale * (1 + compensation.mean(dim=1, keepdim=True))

        result = {
            "qweight": q_weight.to(torch.int8),
            "scales": scale.squeeze(),
            "qzeros": zero_point.squeeze(),
            "g_idx": torch.arange(out_features, dtype=torch.int32),
        }

        if module.bias is not None:
            result["bias"] = module.bias.data

        return result


# ============================================================================
# Main Quantization Pipeline
# ============================================================================

def quantize_model(
    model: nn.Module,
    tokenizer: Any,
    config: QuantizationConfig,
    calibration_data: Optional[List[torch.Tensor]] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> Dict[str, Any]:
    """
    Quantize a model using specified method.

    Args:
        model: PyTorch model to quantize
        tokenizer: Tokenizer for calibration
        config: Quantization configuration
        calibration_data: Pre-loaded calibration data
        device: Device for computation

    Returns:
        Dictionary with quantized weights and metadata
    """
    logger.info(f"Quantizing model with {config.method.upper()} ({config.bits}-bit)")

    model = model.to(device)
    model.eval()

    # Prepare calibration data
    if calibration_data is None:
        samples = generate_vietnamese_calibration_samples()
        calibration_data = []
        for text in samples[:config.calibration_samples]:
            try:
                ids = tokenizer.encode(text, max_length=config.calibration_seq_len, truncation=True)
                if hasattr(ids, 'ids'):
                    ids = ids.ids
                calibration_data.append(torch.tensor(ids, dtype=torch.long))
            except Exception:
                pass

    # Stack calibration data
    if calibration_data:
        max_len = max(len(x) for x in calibration_data)
        padded = []
        for x in calibration_data:
            if len(x) < max_len:
                x = torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)])
            padded.append(x[:max_len])
        calib_batch = torch.stack(padded).to(device)
    else:
        calib_batch = None

    # Collect activations
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            activations[name] = input.detach().view(-1, input.shape[-1])
        return hook

    # Register hooks on linear layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Run calibration
    if calib_batch is not None:
        logger.info("Running calibration forward pass...")
        with torch.no_grad():
            try:
                _ = model(calib_batch)
            except Exception as e:
                logger.warning(f"Calibration forward pass failed: {e}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Choose quantizer
    if config.method == "awq":
        quantizer = AWQQuantizer(config)
    elif config.method == "gptq":
        quantizer = GPTQQuantizer(config)
    else:
        raise ValueError(f"Unknown quantization method: {config.method}")

    # Quantize each linear layer
    quantized_state = {}
    total_original = 0
    total_quantized = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            logger.info(f"Quantizing {name}...")

            act = activations.get(name)
            q_result = quantizer.quantize_linear(module, act)

            # Store quantized weights
            for key, value in q_result.items():
                quantized_state[f"{name}.{key}"] = value

            # Track compression
            orig_size = module.weight.numel() * 4  # float32
            quant_size = q_result["qweight"].numel() * (config.bits / 8)
            quant_size += q_result["scales"].numel() * 2  # float16 scales

            total_original += orig_size
            total_quantized += quant_size

    # Add non-quantized weights
    for name, param in model.named_parameters():
        # Skip if already quantized
        if any(name.startswith(qn.rsplit('.', 1)[0]) for qn in quantized_state if 'qweight' in qn):
            continue
        quantized_state[name] = param.data

    # Compute compression ratio
    compression_ratio = total_original / max(1, total_quantized)

    logger.info(f"Quantization complete!")
    logger.info(f"  Original size: {total_original / 1e6:.2f} MB")
    logger.info(f"  Quantized size: {total_quantized / 1e6:.2f} MB")
    logger.info(f"  Compression ratio: {compression_ratio:.2f}x")

    return {
        "state_dict": quantized_state,
        "config": {
            "method": config.method,
            "bits": config.bits,
            "group_size": config.group_size,
            "compression_ratio": compression_ratio,
        },
    }


def save_quantized_model(
    quantized: Dict[str, Any],
    output_path: str,
    save_format: str = "safetensors",
) -> None:
    """
    Save quantized model to disk.

    Args:
        quantized: Result from quantize_model()
        output_path: Output directory or file path
        save_format: "safetensors", "pt", or "both"
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    state_dict = quantized["state_dict"]
    config = quantized["config"]

    if save_format in ["safetensors", "both"]:
        try:
            from safetensors.torch import save_file

            # Convert to safetensors-compatible format
            safe_dict = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    safe_dict[k] = v.contiguous()

            save_file(safe_dict, output_path / "model.safetensors")
            logger.info(f"Saved safetensors to {output_path / 'model.safetensors'}")
        except ImportError:
            logger.warning("safetensors not available, falling back to .pt")
            save_format = "pt"

    if save_format in ["pt", "both"]:
        torch.save(state_dict, output_path / "model.pt")
        logger.info(f"Saved PyTorch model to {output_path / 'model.pt'}")

    # Save config
    with open(output_path / "quantization_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved quantization config to {output_path / 'quantization_config.json'}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantize vAGI model for edge deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # AWQ quantization (recommended)
    python scripts/quantize.py --model checkpoints/model.pt --output checkpoints/model_4bit

    # GPTQ quantization
    python scripts/quantize.py --model gpt2 --method gptq --output checkpoints/gpt2_4bit

    # Custom calibration data
    python scripts/quantize.py --model checkpoints/model.pt \\
        --calibration data/vietnamese_samples.jsonl --output checkpoints/model_4bit

    # 8-bit quantization
    python scripts/quantize.py --model checkpoints/model.pt --bits 8 --output checkpoints/model_8bit
        """
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HuggingFace model name")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for quantized model")
    parser.add_argument("--method", type=str, default="awq", choices=["awq", "gptq"],
                        help="Quantization method (default: awq)")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Quantization bit width (default: 4)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for quantization (default: 128)")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to calibration data (JSONL)")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--format", type=str, default="safetensors",
                        choices=["safetensors", "pt", "both"],
                        help="Output format (default: safetensors)")
    parser.add_argument("--use-autoawq", action="store_true",
                        help="Use AutoAWQ library if available")

    args = parser.parse_args()

    # Create config
    config = QuantizationConfig(
        method=args.method,
        bits=args.bits,
        group_size=args.group_size,
        calibration_samples=args.num_samples,
        save_format=args.format,
    )

    # Try to use AutoAWQ/AutoGPTQ if requested
    if args.use_autoawq and args.method == "awq":
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer

            logger.info("Using AutoAWQ library...")

            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoAWQForCausalLM.from_pretrained(args.model)

            # Generate calibration data
            calib_samples = generate_vietnamese_calibration_samples()

            quant_config = {
                "w_bit": args.bits,
                "q_group_size": args.group_size,
                "version": "gemm",
            }

            model.quantize(tokenizer, quant_config=quant_config)
            model.save_quantized(args.output)

            logger.info(f"Model quantized and saved to {args.output}")
            return

        except ImportError:
            logger.warning("AutoAWQ not available, using built-in quantizer")

    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    except Exception as e:
        logger.warning(f"Failed to load as HuggingFace model: {e}")

        # Try loading as vAGI checkpoint
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.agi import AGIModel, AGIConfig

        checkpoint = torch.load(args.model, map_location='cpu')
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            model_config = AGIConfig()

        model = AGIModel(model_config)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # Create simple tokenizer
        from core.nlp import BytePairTokenizer
        tokenizer = BytePairTokenizer(vocab_size=model_config.vocab_size)

    # Load calibration data
    calibration_data = None
    if args.calibration:
        calibration_data = load_calibration_data(
            args.calibration,
            tokenizer,
            num_samples=config.calibration_samples,
            seq_len=config.calibration_seq_len,
        )

    # Quantize
    start_time = time.time()
    quantized = quantize_model(
        model=model,
        tokenizer=tokenizer,
        config=config,
        calibration_data=calibration_data,
        device=device,
    )
    elapsed = time.time() - start_time

    # Save
    save_quantized_model(quantized, args.output, args.format)

    logger.info(f"\nQuantization completed in {elapsed:.1f} seconds")
    logger.info(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
