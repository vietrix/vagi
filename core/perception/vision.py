"""Advanced vision encoder for multi-modal understanding."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ImageObsEncoder(nn.Module):
    """Simple CNN encoder for image observations (backward compatibility)."""
    
    def __init__(
        self,
        in_channels: int = 3,
        obs_dim: int = 128,
        hidden_size: int = 256
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, hidden_size, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size, obs_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to observation vector."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings."""
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformerBlock(nn.Module):
    """Transformer block for vision."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through transformer block."""
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer encoder."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to visual features."""
        batch_size = x.size(0)
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class CrossModalAttention(nn.Module):
    """Cross-attention between vision and language."""

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, text_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            text_dim,
            num_heads,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(text_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor
    ) -> torch.Tensor:
        """Attend to vision features from text."""
        vision_proj = self.vision_proj(vision_features)
        
        attn_out, _ = self.cross_attn(
            text_features,
            vision_proj,
            vision_proj
        )
        
        output = self.norm(text_features + attn_out)
        
        return output


class ImageTextAligner(nn.Module):
    """Align image and text representations."""

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        shared_dim: int = 512,
    ) -> None:
        super().__init__()
        self.vision_projection = nn.Linear(vision_dim, shared_dim)
        self.text_projection = nn.Linear(text_dim, shared_dim)
        
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image-text similarity."""
        image_embed = F.normalize(self.vision_projection(image_features), dim=-1)
        text_embed = F.normalize(self.text_projection(text_features), dim=-1)
        
        logits = torch.matmul(image_embed, text_embed.T) / self.temperature
        
        return logits, logits.T

    def contrastive_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss."""
        logits_per_image, logits_per_text = self.forward(image_features, text_features)
        
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2


class VideoEncoder(nn.Module):
    """Encode video sequences."""

    def __init__(
        self,
        frame_encoder: nn.Module,
        hidden_size: int,
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder
        self.num_frames = num_frames
        
        self.temporal_encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=8,
            batch_first=True
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to features."""
        batch_size, num_frames, channels, height, width = video.size()
        
        frames = video.view(batch_size * num_frames, channels, height, width)
        
        frame_features = self.frame_encoder(frames)
        
        if frame_features.dim() == 3:
            frame_features = frame_features[:, 0, :]
        
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        temporal_features, _ = self.temporal_encoder(frame_features)
        
        attended_features, _ = self.temporal_attention(
            temporal_features,
            temporal_features,
            temporal_features
        )
        
        video_features = attended_features.mean(dim=1)
        
        return video_features


class MultiModalEncoder(nn.Module):
    """Unified multi-modal encoder."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        fusion_dim: int = 768,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        vision_dim = getattr(vision_encoder, 'norm', None)
        if vision_dim is not None and hasattr(vision_dim, 'normalized_shape'):
            vision_dim = vision_dim.normalized_shape[0]
        else:
            vision_dim = fusion_dim
        
        text_dim = getattr(text_encoder, 'hidden_size', None) or getattr(text_encoder.cfg, 'hidden_size', fusion_dim) if hasattr(text_encoder, 'cfg') else fusion_dim
        
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        self.fusion = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=8,
                dim_feedforward=fusion_dim * 4,
                batch_first=True
            )
            for _ in range(4)
        ])

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None  # Pre-computed vision features
    ) -> torch.Tensor:
        """Encode multi-modal inputs.

        Args:
            image: Raw image tensor (will be encoded if vision_features not provided)
            text: Text input tensor
            vision_features: Pre-computed vision features (avoids redundant encoding)
        """
        modality_features = []

        if vision_features is not None or image is not None:
            # Use pre-computed features if available, otherwise encode
            if vision_features is None:
                vision_features = self.vision_encoder(image)
            if vision_features.dim() == 3:
                vision_proj = self.vision_proj(vision_features)
            else:
                vision_proj = self.vision_proj(vision_features.unsqueeze(1))
            modality_features.append(vision_proj)
        
        if text is not None:
            text_features = self.text_encoder(text)
            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(1)
            text_proj = self.text_proj(text_features)
            modality_features.append(text_proj)
        
        if not modality_features:
            raise ValueError("At least one modality must be provided")
        
        fused = torch.cat(modality_features, dim=1)
        
        for layer in self.fusion:
            fused = layer(fused)
        
        return fused
