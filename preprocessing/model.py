"""Trainable video preprocessors used before the frozen codec.

``VideoSwinLitePreprocessor`` is the default dense architecture. The earlier
factorized ViT and small CNN remain available for controlled ablations. All
modules preserve the public ``BTCHW -> BTCHW`` API.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .swin import VideoSwinLitePreprocessor


class ResidualBlock(nn.Module):
    """Small spatial residual block; normalization is avoided for tiny video batches."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x), inplace=True)


class ConditionalFusion(nn.Module):
    """Content-dependent attention for merging temporal and spatial features."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat((spatial, temporal), dim=1))
        return gate * spatial + (1.0 - gate) * temporal


class PaperPreprocessor(nn.Module):
    """Spatial/temporal CNN that emits an RGB residual for every frame.

    Input and output use ``[batch, time, channel, height, width]`` in ``[0, 1]``.
    Each output frame uses a causal window ending at that frame. The first frame is
    repeated on the left when the full temporal context is unavailable.
    """

    def __init__(self, temporal_frames: int = 8, channels: int = 16) -> None:
        super().__init__()
        if temporal_frames < 1:
            raise ValueError("temporal_frames must be positive")
        self.temporal_frames = temporal_frames
        self.channels = channels

        self.spatial_stem = nn.Conv2d(3, channels, 3, padding=1)
        self.spatial_residual = ResidualBlock(channels)
        self.temporal_stem = nn.Conv2d(3 * temporal_frames, channels, 3, padding=1)
        self.fusion = ConditionalFusion(channels)
        self.to_rgb = nn.Conv2d(channels, 3, 3, padding=1)

        # Start as the identity transform. This stabilizes the frozen-analyzer setup.
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def _causal_windows(self, clip: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = clip.shape
        windows: list[torch.Tensor] = []
        for index in range(time):
            begin = max(0, index - self.temporal_frames + 1)
            frames = [clip[:, frame] for frame in range(begin, index + 1)]
            if len(frames) < self.temporal_frames:
                frames = [clip[:, 0]] * (self.temporal_frames - len(frames)) + frames
            windows.append(torch.cat(frames, dim=1))
        return torch.stack(windows, dim=1).reshape(
            batch * time, self.temporal_frames * channels, height, width
        )

    def forward(
        self, clip: torch.Tensor, qp: int | float | torch.Tensor | None = None
    ) -> torch.Tensor:
        del qp  # CNN ablation is intentionally not QP-conditioned.
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        batch, time, _, height, width = clip.shape
        current = clip.reshape(batch * time, 3, height, width)

        spatial = self.spatial_residual(F.relu(self.spatial_stem(current), inplace=True))
        temporal = F.relu(self.temporal_stem(self._causal_windows(clip)), inplace=True)
        fused = self.fusion(spatial, temporal)
        output = (current + self.to_rgb(fused)).clamp(0.0, 1.0)
        return output.reshape(batch, time, 3, height, width)


class VideoTransformerPreprocessor(nn.Module):
    """ViT-inspired video preprocessor with factorized attention.

    Full attention over every space-time patch is unnecessarily expensive for
    video.  This module first attends between patches inside each frame and then
    attends over time at each spatial location (the factorization used by many
    video-transformer variants).  A zero-initialized residual head makes a new
    model an exact identity transform, which is important when the downstream
    codec and analyzer are frozen.
    """

    def __init__(
        self,
        patch_size: int = 8,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_residual: float = 0.25,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        if embed_dim < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if not 0.0 < max_residual <= 1.0:
            raise ValueError("max_residual must be in (0, 1]")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.max_residual = max_residual

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        # Conditional positional encoding avoids a fixed training resolution.
        self.position = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )

        def encoder(layer_count: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=round(embed_dim * mlp_ratio),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=layer_count)

        spatial_depth = (depth + 1) // 2
        temporal_depth = depth // 2
        self.spatial_encoder = encoder(spatial_depth)
        self.temporal_encoder = encoder(temporal_depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.to_rgb = nn.ConvTranspose2d(
            embed_dim, 3, kernel_size=patch_size, stride=patch_size
        )

        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def _pad(self, clip: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = clip.shape[-2:]
        pad_height = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_height or pad_width:
            # Replication padding on a 5-D BTCHW tensor requires padding all
            # three trailing dimensions; the channel dimension receives zero.
            clip = F.pad(
                clip, (0, pad_width, 0, pad_height, 0, 0), mode="replicate"
            )
        return clip, (height, width)

    def _temporal_position(
        self, length: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return deterministic sinusoidal positions for arbitrary clip lengths."""

        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0, device=device)) / self.embed_dim)
        )
        embedding = torch.zeros(length, self.embed_dim, device=device, dtype=torch.float32)
        embedding[:, 0::2] = torch.sin(position * frequencies)
        embedding[:, 1::2] = torch.cos(position * frequencies[: self.embed_dim // 2])
        return embedding.to(dtype=dtype)

    def forward(
        self, clip: torch.Tensor, qp: int | float | torch.Tensor | None = None
    ) -> torch.Tensor:
        del qp  # Factorized-ViT ablation is intentionally not QP-conditioned.
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        batch, time, _, _, _ = clip.shape
        padded, (height, width) = self._pad(clip)
        padded_height, padded_width = padded.shape[-2:]

        frames = padded.reshape(batch * time, 3, padded_height, padded_width)
        feature_map = self.patch_embed(frames)
        feature_map = feature_map + self.position(feature_map)
        patch_height, patch_width = feature_map.shape[-2:]
        patches = patch_height * patch_width

        # Spatial attention: B*T independent sequences of H_p*W_p tokens.
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.spatial_encoder(tokens)

        # Temporal attention: B*H_p*W_p independent sequences of T tokens.
        tokens = tokens.reshape(batch, time, patches, self.embed_dim)
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch * patches, time, self.embed_dim)
        tokens = tokens + self._temporal_position(
            time, device=tokens.device, dtype=tokens.dtype
        ).unsqueeze(0)
        tokens = self.temporal_encoder(tokens)
        tokens = tokens.reshape(batch, patches, time, self.embed_dim)
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch * time, patches, self.embed_dim)
        tokens = self.norm(tokens)

        feature_map = tokens.transpose(1, 2).reshape(
            batch * time, self.embed_dim, patch_height, patch_width
        )
        residual = self.max_residual * torch.tanh(self.to_rgb(feature_map))
        residual = residual[..., :height, :width]
        source = clip.reshape(batch * time, 3, height, width)
        output = (source + residual).clamp(0.0, 1.0)
        return output.reshape(batch, time, 3, height, width)


def build_preprocessor(
    kind: str = "swin",
    *,
    temporal_frames: int = 8,
    patch_size: int = 8,
    embed_dim: int = 96,
    depth: int = 4,
    num_heads: int = 4,
    swin_patch_size: int = 4,
    swin_embed_dim: int = 48,
    swin_depth: int = 4,
    swin_num_heads: int = 4,
    swin_window_size: tuple[int, int, int] = (4, 8, 8),
    swin_qp_conditioning: bool = True,
    swin_qp_embed_dim: int = 64,
    max_residual: float = 0.25,
    swin_gated_smoothing: bool = False,
    swin_smoothing_max_strength: float = 0.5,
) -> nn.Module:
    """Construct a preprocessor from checkpoint/CLI-friendly arguments."""

    if kind == "swin":
        return VideoSwinLitePreprocessor(
            patch_size=swin_patch_size,
            embed_dim=swin_embed_dim,
            depth=swin_depth,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
            qp_conditioning=swin_qp_conditioning,
            qp_embed_dim=swin_qp_embed_dim,
            max_residual=max_residual,
            gated_smoothing=swin_gated_smoothing,
            smoothing_max_strength=swin_smoothing_max_strength,
        )
    if kind == "vit":
        return VideoTransformerPreprocessor(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            max_residual=max_residual,
        )
    if kind == "cnn":
        return PaperPreprocessor(temporal_frames=temporal_frames)
    raise ValueError(f"unknown preprocessor {kind!r}; choose 'swin', 'vit', or 'cnn'")


def preprocessor_from_checkpoint(checkpoint: dict) -> nn.Module:
    """Rebuild the inference model using saved arguments, including legacy defaults."""
    args = checkpoint.get("args", {})
    model = build_preprocessor(
        args.get("preprocessor", "cnn"),
        temporal_frames=int(args.get("temporal_frames", 8)),
        patch_size=int(args.get("vit_patch_size", 8)),
        embed_dim=int(args.get("vit_embed_dim", 96)),
        depth=int(args.get("vit_depth", 4)),
        num_heads=int(args.get("vit_heads", 4)),
        swin_patch_size=int(args.get("swin_patch_size", 4)),
        swin_embed_dim=int(args.get("swin_embed_dim", 48)),
        swin_depth=int(args.get("swin_depth", 4)),
        swin_num_heads=int(args.get("swin_heads", 4)),
        swin_window_size=(int(args.get("swin_window_temporal", 4)),
                          int(args.get("swin_window_spatial", 8)),
                          int(args.get("swin_window_spatial", 8))),
        swin_qp_conditioning=bool(args.get("swin_qp_conditioning", False)),
        swin_qp_embed_dim=int(args.get("swin_qp_embed_dim", 64)),
        swin_gated_smoothing=bool(args.get("swin_gated_smoothing", False)),
        swin_smoothing_max_strength=float(args.get("swin_smoothing_max_strength", 0.5)),
        max_residual=float(args.get("max_residual", 0.25)),
    )
    model.load_state_dict(checkpoint["preprocessor"])
    return model
