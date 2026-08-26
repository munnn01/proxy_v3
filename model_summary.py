"""Print torchinfo summaries for the video preprocessor and codec proxy."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torchinfo import summary

from preprocessing import StandardCodecProxy, build_preprocessor


class PreprocessorAtFixedQP(nn.Module):
    """Bind QP so torchinfo exercises the conditioned preprocessor path."""

    def __init__(self, preprocessor: nn.Module, qp: int) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.qp = qp

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.preprocessor(clip, self.qp)


class ProxyAtFixedQP(nn.Module):
    """Bind QP so torchinfo sees a conventional one-input module."""

    def __init__(self, proxy: StandardCodecProxy, qp: int) -> None:
        super().__init__()
        self.proxy = proxy
        self.qp = qp

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.proxy(clip, self.qp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("preprocessor", "proxy", "all"), default="all"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--depth", type=int, default=4, help="torchinfo display depth")

    parser.add_argument(
        "--preprocessor", choices=("swin", "vit", "cnn"), default="swin"
    )
    parser.add_argument("--temporal-frames", type=int, default=8)
    parser.add_argument("--vit-patch-size", type=int, default=8)
    parser.add_argument("--vit-embed-dim", type=int, default=96)
    parser.add_argument("--vit-depth", type=int, default=4)
    parser.add_argument("--vit-heads", type=int, default=4)
    parser.add_argument("--swin-patch-size", type=int, default=4)
    parser.add_argument("--swin-embed-dim", type=int, default=48)
    parser.add_argument("--swin-depth", type=int, default=4)
    parser.add_argument("--swin-heads", type=int, default=4)
    parser.add_argument("--swin-window-temporal", type=int, default=4)
    parser.add_argument("--swin-window-spatial", type=int, default=8)
    parser.add_argument(
        "--swin-qp-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--swin-qp-embed-dim", type=int, default=64)
    parser.add_argument("--max-residual", type=float, default=0.25)

    parser.add_argument("--proxy-checkpoint")
    parser.add_argument("--proxy-hidden-channels", type=int, default=48)
    parser.add_argument("--proxy-latent-channels", type=int, default=64)
    parser.add_argument("--proxy-bottleneck-channels", type=int, default=96)
    parser.add_argument("--proxy-blocks-per-stage", type=int, default=2)
    parser.add_argument("--proxy-film-channels", type=int, default=64)
    parser.add_argument("--proxy-qp-step-divisor", type=float, default=12.0)
    parser.add_argument("--proxy-max-delta", type=float, default=0.5)
    parser.add_argument("--qp", type=int, default=35)
    return parser.parse_args()


def show(name: str, model: nn.Module, args: argparse.Namespace, device: str) -> None:
    print(f"\n{'=' * 28} {name} {'=' * 28}")
    summary(
        model.to(device).eval(),
        input_size=(args.batch_size, args.frames, 3, args.frame_size, args.frame_size),
        device=device,
        depth=args.depth,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        verbose=1,
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    if args.model in {"preprocessor", "all"}:
        preprocessor = build_preprocessor(
            args.preprocessor,
            temporal_frames=args.temporal_frames,
            patch_size=args.vit_patch_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            swin_patch_size=args.swin_patch_size,
            swin_embed_dim=args.swin_embed_dim,
            swin_depth=args.swin_depth,
            swin_num_heads=args.swin_heads,
            swin_window_size=(
                args.swin_window_temporal,
                args.swin_window_spatial,
                args.swin_window_spatial,
            ),
            swin_qp_conditioning=args.swin_qp_conditioning,
            swin_qp_embed_dim=args.swin_qp_embed_dim,
            max_residual=args.max_residual,
        )
        show(
            "PREPROCESSOR",
            PreprocessorAtFixedQP(preprocessor, args.qp),
            args,
            device,
        )

    if args.model in {"proxy", "all"}:
        if args.proxy_checkpoint:
            proxy = StandardCodecProxy.from_checkpoint(args.proxy_checkpoint)
        else:
            proxy = StandardCodecProxy(
                hidden_channels=args.proxy_hidden_channels,
                latent_channels=args.proxy_latent_channels,
                bottleneck_channels=args.proxy_bottleneck_channels,
                blocks_per_stage=args.proxy_blocks_per_stage,
                film_channels=args.proxy_film_channels,
                qp_step_divisor=args.proxy_qp_step_divisor,
                max_delta=args.proxy_max_delta,
            )
        show("CODEC PROXY", ProxyAtFixedQP(proxy, args.qp), args, device)


if __name__ == "__main__":
    main()
