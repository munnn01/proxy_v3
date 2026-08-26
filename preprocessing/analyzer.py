"""Frozen action-recognition analyzers used for the paper's accuracy loss."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FrozenVideoAnalyzer(nn.Module):
    """Torchvision Kinetics-400 model with its official input transform."""

    def __init__(self, name: str = "r3d_18") -> None:
        super().__init__()
        from torchvision.models.video import (
            MC3_18_Weights,
            R2Plus1D_18_Weights,
            R3D_18_Weights,
            S3D_Weights,
            Swin3D_T_Weights,
            mc3_18,
            r2plus1d_18,
            r3d_18,
            s3d,
            swin3d_t,
        )

        choices = {
            "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
            "mc3_18": (mc3_18, MC3_18_Weights.DEFAULT),
            "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.DEFAULT),
            "s3d": (s3d, S3D_Weights.DEFAULT),
            "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        }
        if name not in choices:
            raise ValueError(f"unknown analyzer {name!r}; choose from {sorted(choices)}")
        builder, weights = choices[name]
        self.name = name
        self.network = builder(weights=weights)
        self.transform = weights.transforms()
        self.categories = list(weights.meta["categories"])
        self.network.requires_grad_(False)
        self.network.eval()

    def train(self, mode: bool = True) -> FrozenVideoAnalyzer:
        # The analyzer is fixed in the paper, including its normalization statistics.
        super().train(False)
        self.network.eval()
        return self

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        return self.network(self.transform(clip))

    def forward_with_features(
        self, clip: torch.Tensor, layers: Sequence[str]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return logits and named intermediate activations in one analyzer pass.

        Hooks are installed only for this call and removed in ``finally`` so a
        failed batch cannot leak hooks into later epochs. Frozen analyzer
        parameters remain fixed, while reconstruction gradients can still pass
        through the captured activations to the preprocessor input.
        """

        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        requested = tuple(dict.fromkeys(layers))
        if not requested:
            return self.forward(clip), {}
        modules = dict(self.network.named_modules())
        missing = [name for name in requested if name not in modules]
        if missing:
            available = [name for name in modules if name]
            raise ValueError(
                f"analyzer {self.name!r} has no feature layers {missing}; "
                f"available examples: {available[:12]}"
            )

        features: dict[str, torch.Tensor] = {}
        handles = []
        for name in requested:
            handles.append(
                modules[name].register_forward_hook(
                    lambda _module, _inputs, output, key=name: features.__setitem__(
                        key, output
                    )
                )
            )
        try:
            logits = self.network(self.transform(clip))
        finally:
            for handle in handles:
                handle.remove()
        return logits, features
