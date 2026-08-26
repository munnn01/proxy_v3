"""Lightweight dense Video Swin preprocessor.

The implementation keeps temporal resolution intact and only patchifies the
spatial axes. Alternating 3-D regular/shifted windows mix local motion and image
detail without the quadratic cost of global space-time attention.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        result = tuple(int(item) for item in value)
        if len(result) != 3:
            raise ValueError("expected three values: temporal, height, width")
    if any(item < 1 for item in result):
        raise ValueError("window dimensions must be positive")
    return result


def _window_partition(
    features: torch.Tensor, window_size: tuple[int, int, int]
) -> torch.Tensor:
    """Convert ``[B,T,H,W,C]`` features into ``[B*nW,N,C]`` windows."""

    batch, time, height, width, channels = features.shape
    window_time, window_height, window_width = window_size
    features = features.reshape(
        batch,
        time // window_time,
        window_time,
        height // window_height,
        window_height,
        width // window_width,
        window_width,
        channels,
    )
    return (
        features.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .reshape(-1, window_time * window_height * window_width, channels)
    )


def _window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    batch: int,
    time: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reverse ``_window_partition`` into channel-last video features."""

    window_time, window_height, window_width = window_size
    channels = windows.shape[-1]
    features = windows.reshape(
        batch,
        time // window_time,
        height // window_height,
        width // window_width,
        window_time,
        window_height,
        window_width,
        channels,
    )
    return (
        features.permute(0, 1, 4, 2, 5, 3, 6, 7)
        .contiguous()
        .reshape(batch, time, height, width, channels)
    )


class MLP(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, dimension),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class WindowAttention3D(nn.Module):
    """Multi-head self-attention inside a fixed 3-D video window."""

    def __init__(
        self,
        dimension: int,
        window_size: tuple[int, int, int],
        num_heads: int,
    ) -> None:
        super().__init__()
        if dimension % num_heads:
            raise ValueError("dimension must be divisible by num_heads")
        self.dimension = dimension
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dimension = dimension // num_heads
        self.scale = self.head_dimension**-0.5
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.projection = nn.Linear(dimension, dimension)

        window_time, window_height, window_width = window_size
        relative_positions = (
            (2 * window_time - 1)
            * (2 * window_height - 1)
            * (2 * window_width - 1)
        )
        self.relative_position_bias = nn.Parameter(
            torch.zeros(relative_positions, num_heads)
        )

        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_time),
                torch.arange(window_height),
                torch.arange(window_width),
                indexing="ij",
            )
        )
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[..., 0] += window_time - 1
        relative[..., 1] += window_height - 1
        relative[..., 2] += window_width - 1
        relative[..., 0] *= (2 * window_height - 1) * (2 * window_width - 1)
        relative[..., 1] *= 2 * window_width - 1
        self.register_buffer(
            "relative_position_index", relative.sum(-1), persistent=False
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(
        self, windows: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_windows, tokens, channels = windows.shape
        qkv = self.qkv(windows).reshape(
            batch_windows, tokens, 3, self.num_heads, self.head_dimension
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)

        relative_bias = self.relative_position_bias[
            self.relative_position_index.reshape(-1)
        ]
        relative_bias = relative_bias.reshape(tokens, tokens, self.num_heads).to(
            dtype=attention.dtype
        )
        attention = attention + relative_bias.permute(2, 0, 1).unsqueeze(0)

        if attention_mask is not None:
            window_count = attention_mask.shape[0]
            if batch_windows % window_count:
                raise RuntimeError("attention-mask window count does not divide batch")
            attention = attention.reshape(
                batch_windows // window_count,
                window_count,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + attention_mask[None, :, None, :, :]
            attention = attention.reshape(
                batch_windows, self.num_heads, tokens, tokens
            )

        attention = attention.softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(
            batch_windows, tokens, channels
        )
        return self.projection(output)


class SwinTransformerBlock3D(nn.Module):
    """A dense Video Swin block with optional cyclic window shift."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        window_size: int | Sequence[int] = (4, 8, 8),
        shift_size: int | Sequence[int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.window_size = _triple(window_size)
        self.shift_size = _triple_allow_zero(shift_size)
        if any(
            shift >= window
            for shift, window in zip(self.shift_size, self.window_size, strict=True)
        ):
            raise ValueError("shift dimensions must be smaller than window dimensions")
        self.normalization1 = nn.LayerNorm(dimension)
        self.attention = WindowAttention3D(dimension, self.window_size, num_heads)
        self.normalization2 = nn.LayerNorm(dimension)
        self.mlp = MLP(dimension, round(dimension * mlp_ratio))

    @staticmethod
    def _slices(size: int, window: int, shift: int) -> tuple[slice, ...]:
        if shift == 0:
            return (slice(0, size),)
        return (
            slice(0, size - window),
            slice(size - window, size - shift),
            slice(size - shift, size),
        )

    def _attention_mask(
        self,
        padded_size: tuple[int, int, int],
        shift_size: tuple[int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not any(shift_size):
            return None
        time, height, width = padded_size
        window_time, window_height, window_width = self.window_size
        shift_time, shift_height, shift_width = shift_size
        region_mask = torch.zeros((1, time, height, width, 1), device=device)
        label = 0
        for time_slice in self._slices(time, window_time, shift_time):
            for height_slice in self._slices(height, window_height, shift_height):
                for width_slice in self._slices(width, window_width, shift_width):
                    region_mask[:, time_slice, height_slice, width_slice] = label
                    label += 1
        mask_windows = _window_partition(region_mask, self.window_size).squeeze(-1)
        attention_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        return attention_mask.to(dtype=dtype).masked_fill(
            attention_mask.ne(0), -100.0
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError("Video Swin features must use [B,T,H,W,C]")
        shortcut = features
        features = self.normalization1(features)
        batch, time, height, width, channels = features.shape
        window_time, window_height, window_width = self.window_size
        shift_size = tuple(
            0 if size <= window else shift
            for size, window, shift in zip(
                (time, height, width),
                self.window_size,
                self.shift_size,
                strict=True,
            )
        )

        padded_time = _round_up(time, window_time)
        padded_height = _round_up(height, window_height)
        padded_width = _round_up(width, window_width)
        channel_first = features.permute(0, 4, 1, 2, 3)
        channel_first = F.pad(
            channel_first,
            (0, padded_width - width, 0, padded_height - height, 0, padded_time - time),
        )
        features = channel_first.permute(0, 2, 3, 4, 1)

        if any(shift_size):
            features = torch.roll(
                features,
                shifts=tuple(-shift for shift in shift_size),
                dims=(1, 2, 3),
            )
        attention_mask = self._attention_mask(
            (padded_time, padded_height, padded_width),
            shift_size,
            device=features.device,
            dtype=features.dtype,
        )
        windows = _window_partition(features, self.window_size)
        windows = self.attention(windows, attention_mask)
        features = _window_reverse(
            windows,
            self.window_size,
            batch,
            padded_time,
            padded_height,
            padded_width,
        )
        if any(shift_size):
            features = torch.roll(features, shifts=shift_size, dims=(1, 2, 3))
        features = features[:, :time, :height, :width, :channels]

        features = shortcut + features
        return features + self.mlp(self.normalization2(features))


def _triple_allow_zero(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        result = tuple(int(item) for item in value)
        if len(result) != 3:
            raise ValueError("expected three shift values")
    if any(item < 0 for item in result):
        raise ValueError("shift dimensions cannot be negative")
    return result


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class VideoSwinLitePreprocessor(nn.Module):
    """QP-conditioned single-stage Video Swin predicting a bounded RGB residual.

    The temporal axis is never downsampled. Spatial patchification is reversed
    exactly by a transposed 3-D convolution. The zero-initialized reconstruction
    head makes a newly created model an exact identity mapping. When enabled,
    an H.264/H.265 QP embedding FiLM-modulates every Swin block and gates the
    final residual so one model can learn different behavior at each QP.
    """

    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 48,
        depth: int = 4,
        num_heads: int = 4,
        window_size: int | Sequence[int] = (4, 8, 8),
        mlp_ratio: float = 4.0,
        max_residual: float = 0.25,
        qp_conditioning: bool = True,
        qp_embed_dim: int = 64,
        default_qp: float = 35.0,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        if embed_dim < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if not 0.0 < max_residual <= 1.0:
            raise ValueError("max_residual must be in (0, 1]")
        if qp_embed_dim < 1:
            raise ValueError("qp_embed_dim must be positive")
        if not 0.0 <= default_qp <= 51.0:
            raise ValueError("default_qp must be in [0, 51]")
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = _triple(window_size)
        self.mlp_ratio = mlp_ratio
        self.max_residual = max_residual
        self.qp_conditioning = bool(qp_conditioning)
        self.qp_embed_dim = qp_embed_dim
        self.default_qp = float(default_qp)

        self.patch_embedding = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        self.position = nn.Conv3d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )
        self.embedding_normalization = nn.LayerNorm(embed_dim)
        half_window = tuple(size // 2 for size in self.window_size)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    embed_dim,
                    num_heads,
                    self.window_size,
                    (0, 0, 0) if index % 2 == 0 else half_window,
                    mlp_ratio,
                )
                for index in range(depth)
            ]
        )
        if self.qp_conditioning:
            self.qp_embedding: nn.Module | None = nn.Sequential(
                nn.Linear(1, qp_embed_dim),
                nn.GELU(),
                nn.Linear(qp_embed_dim, qp_embed_dim),
                nn.GELU(),
            )
            self.qp_films = nn.ModuleList(
                nn.Linear(qp_embed_dim, embed_dim * 2) for _ in range(depth)
            )
            self.qp_residual_gate: nn.Linear | None = nn.Linear(qp_embed_dim, 1)
        else:
            self.qp_embedding = None
            self.qp_films = nn.ModuleList()
            self.qp_residual_gate = None
        self.normalization = nn.LayerNorm(embed_dim)
        self.to_rgb = nn.ConvTranspose3d(
            embed_dim,
            3,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        self.apply(self._initialize_transformer)
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)
        if self.qp_residual_gate is not None:
            # sigmoid(2) ~= 0.88: start close to the configured max residual
            # while retaining a hard [0, 1] QP-dependent gate.
            nn.init.zeros_(self.qp_residual_gate.weight)
            nn.init.constant_(self.qp_residual_gate.bias, 2.0)

    @staticmethod
    def _initialize_transformer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _qp_condition(
        self,
        qp: int | float | torch.Tensor | None,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.qp_conditioning:
            return None
        value = self.default_qp if qp is None else qp
        qp_tensor = torch.as_tensor(value, device=device, dtype=dtype).flatten()
        if qp_tensor.numel() == 1:
            qp_tensor = qp_tensor.expand(batch)
        if qp_tensor.numel() != batch:
            raise ValueError(f"QP must be scalar or have {batch} elements")
        if not bool(torch.isfinite(qp_tensor).all()):
            raise ValueError("QP must contain only finite values")
        if bool(((qp_tensor < 0) | (qp_tensor > 51)).any()):
            raise ValueError("QP must be in [0, 51]")
        normalized = ((qp_tensor - 25.5) / 25.5).unsqueeze(1)
        assert self.qp_embedding is not None
        return self.qp_embedding(normalized)

    @staticmethod
    def _apply_film(
        features: torch.Tensor, film: nn.Linear, condition: torch.Tensor
    ) -> torch.Tensor:
        parameters = torch.tanh(film(condition))
        gamma, beta = parameters.chunk(2, dim=1)
        gamma = gamma[:, None, None, None, :]
        beta = beta[:, None, None, None, :]
        return features * (1.0 + gamma) + beta

    def forward(
        self, clip: torch.Tensor, qp: int | float | torch.Tensor | None = None
    ) -> torch.Tensor:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        batch, _, _, height, width = clip.shape
        pad_height = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - width % self.patch_size) % self.patch_size
        channel_first = clip.permute(0, 2, 1, 3, 4)
        if pad_height or pad_width:
            channel_first = F.pad(
                channel_first,
                (0, pad_width, 0, pad_height, 0, 0),
                mode="replicate",
            )

        features = self.patch_embedding(channel_first)
        features = features + self.position(features)
        features = features.permute(0, 2, 3, 4, 1)
        features = self.embedding_normalization(features)
        condition = self._qp_condition(
            qp,
            batch,
            device=features.device,
            dtype=features.dtype,
        )
        for index, block in enumerate(self.blocks):
            if condition is not None:
                features = self._apply_film(
                    features, self.qp_films[index], condition
                )
            features = block(features)
        features = self.normalization(features).permute(0, 4, 1, 2, 3)

        residual = self.max_residual * torch.tanh(self.to_rgb(features))
        if condition is not None:
            assert self.qp_residual_gate is not None
            gate = torch.sigmoid(self.qp_residual_gate(condition))
            residual = residual * gate[:, :, None, None, None]
        residual = residual[..., :height, :width]
        source = clip.permute(0, 2, 1, 3, 4)
        output = (source + residual).clamp(0.0, 1.0)
        return output.permute(0, 2, 1, 3, 4)
