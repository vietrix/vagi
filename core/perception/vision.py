"""Advanced vision encoder for multi-modal understanding."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and skip connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out)

        return out


class ImageObsEncoder(nn.Module):
    """ResNet-style CNN encoder for image observations with BatchNorm and skip connections."""

    def __init__(
        self,
        in_channels: int = 3,
        obs_dim: int = 128,
        hidden_size: int = 256,
        num_blocks: Tuple[int, ...] = (2, 2, 2, 2)
    ) -> None:
        super().__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-style layers with skip connections
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(hidden_size, num_blocks[3], stride=2)

        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size, obs_dim)

        # Initialize weights
        self._init_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        downsample = None
        if stride != 1 or self.in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(self.in_planes, out_channels, stride, downsample))
        self.in_planes = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to observation vector with residual connections."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Residual layers with skip connections
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and projection
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings with learnable CLS token for global representation."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        use_cls_token: bool = True,
        use_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.use_pos_embed = use_pos_embed

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable CLS token for global representation
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable position embeddings including CLS position
        if use_pos_embed:
            num_positions = self.num_patches + 1 if use_cls_token else self.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings with optional CLS token.

        Returns:
            If use_cls_token=True: [B, num_patches + 1, embed_dim] with CLS token at position 0
            If use_cls_token=False: [B, num_patches, embed_dim]
        """
        batch_size = x.size(0)

        # Project patches
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Prepend learnable CLS token for global representation
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]

        # Add position embeddings
        if self.use_pos_embed and self.pos_embed is not None:
            x = x + self.pos_embed

        return x

    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the CLS token representing global image features.

        Args:
            x: Output from forward() with shape [B, num_patches + 1, embed_dim]

        Returns:
            CLS token with shape [B, embed_dim]
        """
        if not self.use_cls_token:
            raise ValueError("CLS token not available when use_cls_token=False")
        return x[:, 0]


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
            image_size,
            patch_size,
            in_channels,
            embed_dim,
            use_cls_token=False,
            use_pos_embed=False,
        )
        
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_embed(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate position embeddings for different image sizes.

        Args:
            x: Input tensor with shape [B, num_patches + 1, embed_dim]
            h: Height in patches
            w: Width in patches

        Returns:
            Position embeddings interpolated to match input size
        """
        num_patches = h * w
        num_positions = self.pos_embed.size(1)

        if num_patches + 1 == num_positions:
            return self.pos_embed

        # Separate class token and patch embeddings
        cls_pos = self.pos_embed[:, :1, :]  # [1, 1, D]
        patch_pos = self.pos_embed[:, 1:, :]  # [1, N, D]

        # Calculate original grid size
        orig_size = int(patch_pos.size(1) ** 0.5)

        # Reshape to 2D grid for interpolation
        dim = patch_pos.size(-1)
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)

        # Interpolate to new size
        patch_pos = F.interpolate(
            patch_pos,
            size=(h, w),
            mode='bicubic',
            align_corners=False
        )

        # Reshape back
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        # Concatenate class token
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to visual features.

        Supports dynamic image sizes through position embedding interpolation.
        """
        batch_size = x.size(0)

        x = self.patch_embed(x)
        num_patches = x.size(1)

        # Calculate patch grid dimensions
        h = w = int(num_patches ** 0.5)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Use interpolated position embeddings for dynamic image sizes
        pos_embed = self.interpolate_pos_embed(x, h, w)
        x = x + pos_embed

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
