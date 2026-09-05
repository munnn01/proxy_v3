"""Small spatial filters shared by preprocessing and proxy calibration."""

import torch
from torch.nn import functional as F


def spatial_lowpass(clip: torch.Tensor) -> torch.Tensor:
    """3x3 binomial blur on BTCHW video; never mix different frames."""
    if clip.ndim != 5:
        raise ValueError("spatial_lowpass expects [B,T,C,H,W]")
    batch, time, channels, height, width = clip.shape
    kernel = clip.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    kernel = kernel.expand(channels, 1, 3, 3)
    frames = clip.reshape(batch * time, channels, height, width)
    filtered = F.conv2d(F.pad(frames, (1, 1, 1, 1), mode="replicate"),
                        kernel, groups=channels)
    return filtered.reshape_as(clip)
