#!/usr/bin/env python3
"""
Vision Adapter for vAGI Multimodal Integration.

Implements a modular vision projection system to connect pre-trained Vision
Encoders (SigLIP, CLIP, ViT) to the Language Model's embedding space.

Architecture:
    Image -> Vision Encoder -> [B, N, D_vision] -> Projector -> [B, N, D_llm]
                                                        |
                                                        v
                                    Concatenate with Text Embeddings
                                                        |
                                                        v
                                                    LLM Backbone

Supported Encoders:
    - SigLIP (google/siglip-*): State-of-the-art vision encoder
    - CLIP (openai/clip-*): Classic vision-language model
    - ViT (google/vit-*): Vision Transformer variants
    - DINOv2 (facebook/dinov2-*): Self-supervised vision features

Usage:
    from core.multimodal import VisionProjector, SigLIPEncoder

    # Create encoder and projector
    encoder = SigLIPEncoder(model_name="google/siglip-base-patch16-224")
    projector = VisionProjector(
        vision_dim=encoder.embed_dim,
        llm_dim=4096,
        projector_type="mlp"
    )

    # Forward pass
    image_features = encoder(images)  # [B, N, 768]
    projected = projector(image_features)  # [B, N, 4096]
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class ProjectorType(Enum):
    """Types of vision-to-LLM projectors."""
    LINEAR = "linear"         # Single linear layer
    MLP = "mlp"               # Multi-layer perceptron (2 layers)
    MLP_GELU = "mlp_gelu"     # MLP with GELU activation
    CROSS_ATTENTION = "cross_attention"  # Cross-attention projector
    PERCEIVER = "perceiver"   # Perceiver-style resampler


class EncoderType(Enum):
    """Supported vision encoder types."""
    SIGLIP = "siglip"
    CLIP = "clip"
    VIT = "vit"
    DINOV2 = "dinov2"
    CUSTOM = "custom"


@dataclass
class VisionConfig:
    """Configuration for vision adapter."""
    # Encoder settings
    encoder_type: str = "siglip"
    encoder_name: str = "google/siglip-base-patch16-224"
    image_size: int = 224
    patch_size: int = 16
    vision_dim: int = 768

    # Projector settings
    projector_type: str = "mlp"
    llm_dim: int = 4096
    projector_depth: int = 2
    projector_dropout: float = 0.0

    # Feature extraction
    use_cls_token: bool = True
    pool_type: str = "none"  # "none", "mean", "cls", "attention"
    num_query_tokens: int = 32  # For perceiver-style

    # Training
    freeze_encoder: bool = True
    freeze_projector: bool = False
    gradient_checkpointing: bool = False


# ============================================================================
# Vision Projector (Core Component)
# ============================================================================

class VisionProjector(nn.Module):
    """
    Projects vision encoder features to LLM embedding space.

    This is the key component that bridges the gap between visual features
    and the language model's understanding. Multiple projector types are
    supported for different use cases:

    - LINEAR: Fastest, single matrix multiplication
    - MLP: Better capacity, standard choice for LLaVA-style models
    - CROSS_ATTENTION: Allows LLM to query visual features
    - PERCEIVER: Reduces sequence length with learned queries

    The projector is designed to be modular - you can swap vision encoders
    without changing the projector architecture.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        projector_type: str = "mlp",
        depth: int = 2,
        dropout: float = 0.0,
        num_query_tokens: int = 32,
        num_heads: int = 8,
    ):
        """
        Initialize the vision projector.

        Args:
            vision_dim: Dimension of vision encoder output
            llm_dim: Dimension of LLM embedding space
            projector_type: Type of projector ("linear", "mlp", "cross_attention", "perceiver")
            depth: Number of layers for MLP-based projectors
            dropout: Dropout probability
            num_query_tokens: Number of query tokens for perceiver
            num_heads: Number of attention heads for attention-based projectors
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.projector_type = ProjectorType(projector_type)

        if self.projector_type == ProjectorType.LINEAR:
            self.projector = nn.Linear(vision_dim, llm_dim)

        elif self.projector_type == ProjectorType.MLP:
            # Standard MLP: vision_dim -> llm_dim -> llm_dim
            layers = []
            in_dim = vision_dim
            for i in range(depth):
                out_dim = llm_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < depth - 1:  # No activation after last layer
                    layers.append(nn.GELU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            self.projector = nn.Sequential(*layers)

        elif self.projector_type == ProjectorType.MLP_GELU:
            # LLaVA-1.5 style: Linear -> GELU -> Linear
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim),
            )

        elif self.projector_type == ProjectorType.CROSS_ATTENTION:
            # Cross-attention allows LLM to query visual features
            self.query_proj = nn.Linear(llm_dim, llm_dim)
            self.key_proj = nn.Linear(vision_dim, llm_dim)
            self.value_proj = nn.Linear(vision_dim, llm_dim)
            self.output_proj = nn.Linear(llm_dim, llm_dim)
            self.num_heads = num_heads
            self.head_dim = llm_dim // num_heads
            self.scale = self.head_dim ** -0.5

        elif self.projector_type == ProjectorType.PERCEIVER:
            # Perceiver-style resampler with learned queries
            self.num_query_tokens = num_query_tokens
            self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, llm_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=llm_dim,
                num_heads=num_heads,
                kdim=vision_dim,
                vdim=vision_dim,
                batch_first=True,
            )
            self.ff = nn.Sequential(
                nn.Linear(llm_dim, llm_dim * 4),
                nn.GELU(),
                nn.Linear(llm_dim * 4, llm_dim),
            )
            self.norm1 = nn.LayerNorm(llm_dim)
            self.norm2 = nn.LayerNorm(llm_dim)

        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        self._init_weights()

    def _init_weights(self):
        """Initialize projector weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        vision_features: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project vision features to LLM embedding space.

        Args:
            vision_features: Vision encoder output [batch, num_patches, vision_dim]
            query_embeds: Optional query embeddings for cross-attention [batch, seq, llm_dim]

        Returns:
            Projected features [batch, num_tokens, llm_dim]
        """
        if self.projector_type in [ProjectorType.LINEAR, ProjectorType.MLP, ProjectorType.MLP_GELU]:
            return self.projector(vision_features)

        elif self.projector_type == ProjectorType.CROSS_ATTENTION:
            if query_embeds is None:
                raise ValueError("query_embeds required for cross_attention projector")

            batch_size = vision_features.size(0)

            # Project queries, keys, values
            q = self.query_proj(query_embeds)
            k = self.key_proj(vision_features)
            v = self.value_proj(vision_features)

            # Reshape for multi-head attention
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

            # Reshape and project
            out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.llm_dim)
            return self.output_proj(out)

        elif self.projector_type == ProjectorType.PERCEIVER:
            batch_size = vision_features.size(0)

            # Expand query tokens for batch
            queries = self.query_tokens.expand(batch_size, -1, -1)

            # Cross-attention: queries attend to vision features
            attn_out, _ = self.cross_attn(queries, vision_features, vision_features)
            queries = self.norm1(queries + attn_out)

            # Feed-forward
            ff_out = self.ff(queries)
            output = self.norm2(queries + ff_out)

            return output

        raise RuntimeError(f"Unhandled projector type: {self.projector_type}")

    @property
    def output_dim(self) -> int:
        """Output dimension of the projector."""
        return self.llm_dim

    @property
    def output_tokens(self) -> Optional[int]:
        """Number of output tokens (only for perceiver)."""
        if self.projector_type == ProjectorType.PERCEIVER:
            return self.num_query_tokens
        return None


# ============================================================================
# Vision Encoder Base Class
# ============================================================================

class VisionEncoder(nn.Module, ABC):
    """
    Abstract base class for vision encoders.

    All vision encoders must implement:
    - forward(): Process images and return features
    - embed_dim: Output embedding dimension
    - num_patches: Number of output patches/tokens

    This abstraction allows easy swapping of different vision backbones.
    """

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors.

        Args:
            images: Input images [batch, channels, height, width]

        Returns:
            Vision features [batch, num_patches, embed_dim]
        """
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Output embedding dimension."""
        pass

    @property
    @abstractmethod
    def num_patches(self) -> int:
        """Number of output patches/tokens."""
        pass


# ============================================================================
# SigLIP Encoder (State-of-the-Art)
# ============================================================================

class SigLIPEncoder(VisionEncoder):
    """
    SigLIP Vision Encoder - State-of-the-art for vision-language tasks.

    SigLIP (Sigmoid Loss for Language Image Pre-training) improves upon CLIP
    by using sigmoid loss instead of softmax, enabling better scaling.

    Models:
        - google/siglip-base-patch16-224: Base model (768-dim)
        - google/siglip-large-patch16-256: Large model (1024-dim)
        - google/siglip-so400m-patch14-384: Largest model (1152-dim)

    Usage:
        encoder = SigLIPEncoder("google/siglip-base-patch16-224")
        features = encoder(images)  # [B, 196, 768] for 224x224 images
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        freeze: bool = True,
        use_cls_token: bool = False,
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize SigLIP encoder.

        Args:
            model_name: HuggingFace model name
            freeze: Whether to freeze encoder weights
            use_cls_token: Whether to prepend CLS token (SigLIP doesn't use CLS by default)
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        """
        super().__init__()
        self.model_name = model_name
        self.use_cls_token = use_cls_token
        self._embed_dim = None
        self._num_patches = None

        try:
            from transformers import SiglipVisionModel, SiglipImageProcessor

            logger.info(f"Loading SigLIP encoder: {model_name}")
            self.encoder = SiglipVisionModel.from_pretrained(model_name)
            self.processor = SiglipImageProcessor.from_pretrained(model_name)

            # Get config info
            config = self.encoder.config
            self._embed_dim = config.hidden_size
            self._image_size = config.image_size
            self._patch_size = config.patch_size
            self._num_patches = (self._image_size // self._patch_size) ** 2

            if gradient_checkpointing:
                self.encoder.gradient_checkpointing_enable()

            if freeze:
                self._freeze()

            logger.info(f"SigLIP loaded: embed_dim={self._embed_dim}, patches={self._num_patches}")

        except ImportError:
            logger.warning("transformers not available, using dummy SigLIP encoder")
            self._create_dummy_encoder()

    def _create_dummy_encoder(self):
        """Create a dummy encoder for testing without dependencies."""
        self._embed_dim = 768
        self._num_patches = 196
        self._image_size = 224
        self._patch_size = 16

        # Simple ViT-like encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self._embed_dim, kernel_size=self._patch_size, stride=self._patch_size),
            nn.Flatten(2),  # [B, D, N] -> need to transpose
        )
        self.processor = None

    def _freeze(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        logger.info("SigLIP encoder frozen")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using SigLIP.

        Args:
            images: Input images [batch, channels, height, width]
                    Expected to be normalized to [-1, 1] or [0, 1]

        Returns:
            Vision features [batch, num_patches, embed_dim]
        """
        if hasattr(self.encoder, 'vision_model'):
            # HuggingFace SigLIP
            outputs = self.encoder(images, return_dict=True)
            features = outputs.last_hidden_state  # [B, N, D]
        else:
            # Dummy encoder
            features = self.encoder(images)  # [B, D, N]
            features = features.transpose(1, 2)  # [B, N, D]

        return features

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches


# ============================================================================
# CLIP Encoder (Classic)
# ============================================================================

class CLIPEncoder(VisionEncoder):
    """
    CLIP Vision Encoder - Classic vision-language model.

    Models:
        - openai/clip-vit-base-patch16: Base ViT-B/16 (768-dim)
        - openai/clip-vit-large-patch14: Large ViT-L/14 (1024-dim)

    Usage:
        encoder = CLIPEncoder("openai/clip-vit-base-patch16")
        features = encoder(images)  # [B, 197, 768] including CLS
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze: bool = True,
        include_cls: bool = True,
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize CLIP encoder.

        Args:
            model_name: HuggingFace model name
            freeze: Whether to freeze encoder weights
            include_cls: Whether to include CLS token in output
            gradient_checkpointing: Enable gradient checkpointing
        """
        super().__init__()
        self.model_name = model_name
        self.include_cls = include_cls
        self._embed_dim = None
        self._num_patches = None

        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor

            logger.info(f"Loading CLIP encoder: {model_name}")
            self.encoder = CLIPVisionModel.from_pretrained(model_name)
            self.processor = CLIPImageProcessor.from_pretrained(model_name)

            config = self.encoder.config
            self._embed_dim = config.hidden_size
            self._image_size = config.image_size
            self._patch_size = config.patch_size
            self._num_patches = (self._image_size // self._patch_size) ** 2
            if self.include_cls:
                self._num_patches += 1

            if gradient_checkpointing:
                self.encoder.gradient_checkpointing_enable()

            if freeze:
                self._freeze()

        except ImportError:
            logger.warning("transformers not available, using dummy CLIP encoder")
            self._create_dummy_encoder()

    def _create_dummy_encoder(self):
        """Create dummy encoder."""
        self._embed_dim = 768
        self._num_patches = 197 if self.include_cls else 196

        self.encoder = nn.Sequential(
            nn.Conv2d(3, self._embed_dim, kernel_size=16, stride=16),
            nn.Flatten(2),
        )
        self.processor = None

    def _freeze(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP."""
        if hasattr(self.encoder, 'vision_model'):
            outputs = self.encoder(images, return_dict=True)
            features = outputs.last_hidden_state
            if not self.include_cls:
                features = features[:, 1:, :]  # Remove CLS
        else:
            features = self.encoder(images).transpose(1, 2)

        return features

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches


# ============================================================================
# Multimodal Fusion Module
# ============================================================================

class MultimodalFusion(nn.Module):
    """
    Fuses visual and text embeddings for multimodal understanding.

    Fusion Strategies:
    - CONCAT: Simple concatenation [visual; text]
    - INTERLEAVE: Interleave visual tokens between text
    - PREPEND: Prepend visual tokens before text
    - CROSS_ATTENTION: Cross-attention between modalities

    Usage:
        fusion = MultimodalFusion(
            vision_dim=768,
            text_dim=4096,
            fusion_type="prepend"
        )
        fused = fusion(vision_features, text_embeddings)
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        fusion_type: str = "prepend",
        num_heads: int = 8,
    ):
        """
        Initialize fusion module.

        Args:
            vision_dim: Dimension of projected vision features
            text_dim: Dimension of text embeddings
            fusion_type: Type of fusion ("concat", "prepend", "cross_attention")
            num_heads: Attention heads for cross-attention
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fusion_type = fusion_type

        # Ensure dimensions match for fusion
        if vision_dim != text_dim:
            self.vision_proj = nn.Linear(vision_dim, text_dim)
        else:
            self.vision_proj = nn.Identity()

        if fusion_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(text_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse vision and text features.

        Args:
            vision_features: Projected vision features [batch, num_vision, dim]
            text_embeddings: Text embeddings [batch, seq_len, dim]
            vision_mask: Optional mask for vision tokens
            text_mask: Optional mask for text tokens

        Returns:
            Tuple of (fused_embeddings, attention_mask)
        """
        # Project vision to text dimension if needed
        vision_features = self.vision_proj(vision_features)

        batch_size = vision_features.size(0)
        num_vision = vision_features.size(1)
        seq_len = text_embeddings.size(1)
        device = vision_features.device

        if self.fusion_type == "prepend":
            # [vision_tokens; text_tokens]
            fused = torch.cat([vision_features, text_embeddings], dim=1)

            # Create combined mask
            if vision_mask is None:
                vision_mask = torch.ones(batch_size, num_vision, device=device)
            if text_mask is None:
                text_mask = torch.ones(batch_size, seq_len, device=device)
            combined_mask = torch.cat([vision_mask, text_mask], dim=1)

        elif self.fusion_type == "concat":
            # Same as prepend for simplicity
            fused = torch.cat([vision_features, text_embeddings], dim=1)

            if vision_mask is None:
                vision_mask = torch.ones(batch_size, num_vision, device=device)
            if text_mask is None:
                text_mask = torch.ones(batch_size, seq_len, device=device)
            combined_mask = torch.cat([vision_mask, text_mask], dim=1)

        elif self.fusion_type == "cross_attention":
            # Text attends to vision
            attn_out, _ = self.cross_attn(
                text_embeddings,  # query
                vision_features,  # key
                vision_features,  # value
                key_padding_mask=~vision_mask.bool() if vision_mask is not None else None,
            )
            fused = self.norm(text_embeddings + attn_out)
            combined_mask = text_mask if text_mask is not None else torch.ones(batch_size, seq_len, device=device)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return fused, combined_mask


# ============================================================================
# Complete Vision-Language Model Adapter
# ============================================================================

class VisionLanguageAdapter(nn.Module):
    """
    Complete adapter connecting vision encoder to language model.

    This is the main interface for adding vision capability to an LLM.
    It handles:
    1. Image encoding via vision encoder (SigLIP, CLIP, etc.)
    2. Feature projection to LLM dimension
    3. Fusion with text embeddings

    Usage:
        adapter = VisionLanguageAdapter(
            config=VisionConfig(
                encoder_type="siglip",
                encoder_name="google/siglip-base-patch16-224",
                llm_dim=4096,
            )
        )

        # In forward pass
        fused_embeddings = adapter(images, text_embeddings)
    """

    def __init__(self, config: VisionConfig):
        """
        Initialize vision-language adapter.

        Args:
            config: Vision configuration
        """
        super().__init__()
        self.config = config

        # Create vision encoder based on type
        encoder_type = EncoderType(config.encoder_type)

        if encoder_type == EncoderType.SIGLIP:
            self.encoder = SigLIPEncoder(
                model_name=config.encoder_name,
                freeze=config.freeze_encoder,
                gradient_checkpointing=config.gradient_checkpointing,
            )
        elif encoder_type == EncoderType.CLIP:
            self.encoder = CLIPEncoder(
                model_name=config.encoder_name,
                freeze=config.freeze_encoder,
                gradient_checkpointing=config.gradient_checkpointing,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        # Create projector
        self.projector = VisionProjector(
            vision_dim=self.encoder.embed_dim,
            llm_dim=config.llm_dim,
            projector_type=config.projector_type,
            depth=config.projector_depth,
            dropout=config.projector_dropout,
            num_query_tokens=config.num_query_tokens,
        )

        # Create fusion module
        self.fusion = MultimodalFusion(
            vision_dim=config.llm_dim,
            text_dim=config.llm_dim,
            fusion_type="prepend",
        )

        logger.info(f"VisionLanguageAdapter initialized: {encoder_type.value} -> {config.projector_type}")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to projected features.

        Args:
            images: Input images [batch, channels, height, width]

        Returns:
            Projected features [batch, num_tokens, llm_dim]
        """
        with torch.set_grad_enabled(not self.config.freeze_encoder):
            vision_features = self.encoder(images)

        projected = self.projector(vision_features)
        return projected

    def forward(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, project, fuse.

        Args:
            images: Input images [batch, channels, height, width]
            text_embeddings: Text embeddings from LLM [batch, seq_len, llm_dim]
            image_mask: Optional mask for image tokens
            text_mask: Optional mask for text tokens

        Returns:
            Tuple of (fused_embeddings, combined_mask)
        """
        # Encode and project images
        vision_features = self.encode_images(images)

        # Fuse with text
        fused, mask = self.fusion(
            vision_features,
            text_embeddings,
            vision_mask=image_mask,
            text_mask=text_mask,
        )

        return fused, mask

    @property
    def vision_token_count(self) -> int:
        """Number of vision tokens produced."""
        if self.projector.output_tokens is not None:
            return self.projector.output_tokens
        return self.encoder.num_patches


# ============================================================================
# Image Preprocessing Utilities
# ============================================================================

def preprocess_images(
    images: Union[torch.Tensor, List[Any]],
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Preprocess images for vision encoder.

    Args:
        images: Input images (tensor or list of PIL images)
        image_size: Target size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Preprocessed tensor [batch, 3, image_size, image_size]
    """
    if isinstance(images, torch.Tensor):
        # Assume already in [B, C, H, W] format
        if images.shape[-1] != image_size or images.shape[-2] != image_size:
            images = F.interpolate(images, size=(image_size, image_size), mode='bilinear', align_corners=False)

        # Normalize
        mean_t = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=images.device).view(1, 3, 1, 1)

        if images.max() > 1.0:
            images = images / 255.0

        images = (images - mean_t) / std_t
        return images

    # Handle PIL images
    try:
        from PIL import Image
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        if not isinstance(images, list):
            images = [images]

        tensors = [transform(img) for img in images]
        return torch.stack(tensors)

    except ImportError:
        raise ValueError("PIL/torchvision required for image list preprocessing")


# ============================================================================
# Example Usage
# ============================================================================

def demo_vision_adapter():
    """Demonstrate vision adapter usage."""

    # Create configuration
    config = VisionConfig(
        encoder_type="siglip",
        encoder_name="google/siglip-base-patch16-224",
        llm_dim=4096,
        projector_type="mlp",
    )

    # Create adapter
    adapter = VisionLanguageAdapter(config)

    # Dummy inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    text_embeddings = torch.randn(batch_size, 32, 4096)

    # Forward pass
    fused, mask = adapter(images, text_embeddings)

    print(f"Input images: {images.shape}")
    print(f"Input text: {text_embeddings.shape}")
    print(f"Fused output: {fused.shape}")
    print(f"Output mask: {mask.shape}")
    print(f"Vision tokens: {adapter.vision_token_count}")

    return fused


if __name__ == "__main__":
    demo_vision_adapter()
