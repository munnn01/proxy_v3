"""Distill a differentiable proxy from FFmpeg H.264/H.265 reconstructions."""

from __future__ import annotations

import argparse
import math
import random
from copy import copy
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision.models.video import R3D_18_Weights

from preprocessing import StandardCodecProxy, StandardVideoCodec
from preprocessing.data import (
    MixedQPBatchSampler,
    PrecomputedCodecDataset,
    VideoFolderDataset,
    stratified_split_indices,
    stratified_limit_indices,
)
from preprocessing.filters import spatial_lowpass
from preprocessing.model import preprocessor_from_checkpoint
from preprocessing.proxy_training import (
    log_rate, mixed_qp_roundtrip, probe_rate_descent, rate_delta_loss, rate_fit_loss,
)
from preprocessing.standard_codec import require_ffmpeg
from preprocessing.utils import AverageMeter, save_checkpoint, seed_everything, validate_run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument(
        "--precomputed-root",
        help="cache made by precompute_codec.py; avoids FFmpeg during training",
    )
    data.add_argument("--data-root", help="root containing train/ and optionally val/")
    data.add_argument("--train-dir")
    data.add_argument("--val-dir")
    data.add_argument("--val-ratio", type=float, default=0.1)
    data.add_argument("--frames", type=int, default=16)
    data.add_argument("--frame-stride", type=int, default=2)
    data.add_argument("--frame-size", type=int, default=128)
    data.add_argument("--limit-train", type=int)
    data.add_argument("--limit-val", type=int)
    data.add_argument("--workers", type=int, default=2)

    codec = parser.add_argument_group("codec")
    codec.add_argument("--codec", choices=("h264", "h265"), default="h264")
    codec.add_argument("--qps", type=int, nargs="+", default=[30, 35, 40, 45])
    codec.add_argument("--fps", type=float, default=30.0)
    codec.add_argument("--preset", default="medium")
    codec.add_argument("--ffmpeg", default="ffmpeg")
    codec.add_argument("--codec-io", choices=("pipe", "png"), default="pipe")
    codec.add_argument("--codec-workers", type=int, default=2)
    codec.add_argument("--ffmpeg-threads", type=int, default=1)

    model = parser.add_argument_group("proxy")
    model.add_argument("--hidden-channels", type=int, default=48)
    model.add_argument("--latent-channels", type=int, default=64)
    model.add_argument("--bottleneck-channels", type=int, default=96)
    model.add_argument("--blocks-per-stage", type=int, default=2)
    model.add_argument("--film-channels", type=int, default=64)
    model.add_argument("--qp-step-divisor", type=float, default=12.0)
    model.add_argument("--max-delta", type=float, default=0.5)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--epochs", type=int, default=20)
    optimization.add_argument("--batch-size", type=int, default=8)
    optimization.add_argument("--lr", type=float, default=2e-4)
    optimization.add_argument("--rate-weight", type=float, default=0.1)
    optimization.add_argument("--rate-loss", choices=("absolute", "log"), default="absolute")
    optimization.add_argument("--rate-delta-weight", type=float, default=0.0)
    optimization.add_argument(
        "--pair-strengths", type=float, nargs="+",
        help="encode paired variants with these spatial blur strengths; enables real FFmpeg",
    )
    optimization.add_argument(
        "--preprocessor-checkpoint",
        help="frozen Swin used to generate paired on-policy variants; requires --pair-strengths",
    )
    optimization.add_argument("--init-checkpoint", help="load proxy weights only for a new calibration run")
    optimization.add_argument("--gradient-probe-batches", type=int, default=0)
    optimization.add_argument("--gradient-probe-step", type=float, default=2.0 / 255.0)
    optimization.add_argument("--weight-decay", type=float, default=1e-4)
    optimization.add_argument("--clip-grad", type=float, default=1.0)
    optimization.add_argument("--scheduler-factor", type=float, default=0.5)
    optimization.add_argument("--scheduler-patience", type=int, default=3)
    optimization.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    optimization.add_argument("--device", default="cuda")
    optimization.add_argument("--seed", type=int, default=42)
    optimization.add_argument("--resume")
    optimization.add_argument("--output-dir", default="checkpoints/proxy")
    optimization.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_directories(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.train_dir:
        return Path(args.train_dir), Path(args.val_dir) if args.val_dir else None
    if not args.data_root:
        raise ValueError("provide --data-root or --train-dir")
    root = Path(args.data_root)
    train = root / "train" if (root / "train").is_dir() else root
    validation = root / "val"
    return train, validation if validation.is_dir() else None


def validate_cache(args: argparse.Namespace, dataset: PrecomputedCodecDataset) -> None:
    manifest = dataset.manifest
    codec = manifest["codec"]
    video = manifest["video"]
    expected = {
        "codec": (codec["name"], args.codec),
        "fps": (float(codec["fps"]), float(args.fps)),
        "preset": (codec["preset"], args.preset),
        "frames": (int(video["frames"]), args.frames),
        "frame_stride": (int(video["frame_stride"]), args.frame_stride),
        "frame_size": (int(video["frame_size"]), args.frame_size),
    }
    mismatches = [
        f"{name}: cache={cached!r}, CLI={requested!r}"
        for name, (cached, requested) in expected.items()
        if cached != requested
    ]
    if mismatches:
        raise ValueError("precomputed cache configuration mismatch: " + "; ".join(mismatches))


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    common = {
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    if args.precomputed_root:
        train_set = PrecomputedCodecDataset(args.precomputed_root, "train", args.qps)
        val_set = PrecomputedCodecDataset(args.precomputed_root, "val", args.qps)
        validate_cache(args, train_set)
        for dataset, limit in ((train_set, 8 if args.smoke_test else args.limit_train),
                               (val_set, 4 if args.smoke_test else args.limit_val)):
            if limit is not None:
                sample_info = [(Path(sample["source"]), int(sample["label"]))
                               for sample in dataset.samples]
                indices = stratified_limit_indices(sample_info, list(range(len(sample_info))),
                                                   limit, args.seed)
                dataset.samples = [dataset.samples[index] for index in indices]
                dataset.items = [(sample, qp) for sample in dataset.samples for qp in dataset.qps]
        batch_sampler = MixedQPBatchSampler(
            train_set, args.batch_size, seed=args.seed
        )
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler, **common)
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        )
        return train_loader, val_loader

    train_dir, val_dir = resolve_directories(args)
    categories = list(R3D_18_Weights.DEFAULT.meta["categories"])
    options = {
        "frames": args.frames,
        "stride": args.frame_stride,
        "size": args.frame_size,
    }
    train_limit = 8 if args.smoke_test else args.limit_train
    val_limit = 4 if args.smoke_test else args.limit_val
    if val_dir is not None:
        train_set = VideoFolderDataset(
            train_dir, categories, train=True, limit=train_limit, **options
        )
        val_set = VideoFolderDataset(
            val_dir, categories, train=False, limit=val_limit, **options
        )
    else:
        source = VideoFolderDataset(train_dir, categories, train=False, **options)
        train_indices, val_indices = stratified_split_indices(
            source.samples, args.val_ratio, args.seed
        )
        train_indices = stratified_limit_indices(source.samples, train_indices, train_limit, args.seed)
        val_indices = stratified_limit_indices(source.samples, val_indices, val_limit, args.seed)
        augmented = copy(source)
        augmented.train = True
        train_set = Subset(augmented, train_indices)
        val_set = Subset(source, val_indices)
    common["batch_size"] = args.batch_size
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **common),
        DataLoader(val_set, shuffle=False, drop_last=False, **common),
    )


def run_epoch(
    loader: DataLoader,
    proxy: StandardCodecProxy,
    real_codec: StandardVideoCodec | None,
    args: argparse.Namespace,
    device: torch.device,
    *,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int = 0,
    preprocessor: torch.nn.Module | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    proxy.train(training)
    if training and hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)
    names = ["loss", "reconstruction", "rate", "rate_delta", "rate_mape_percent",
             "pair_direction_accuracy", "probe_real_delta_percent",
             "probe_proxy_down_fraction", "probe_real_down_fraction"]
    names += [f"qp{qp}_rate_mape_percent" for qp in args.qps]
    names += [f"qp{qp}_variant_rate_mape_percent" for qp in args.qps]
    meters = {name: AverageMeter() for name in names}
    use_amp = bool(args.amp and device.type == "cuda")
    iterator = tqdm(loader, desc="proxy train" if training else "proxy valid", leave=False)
    context = torch.enable_grad if training else torch.no_grad
    if training:
        optimizer.zero_grad(set_to_none=True)
    with context():
        for step, batch_data in enumerate(iterator):
            if args.precomputed_root:
                clips, real_reconstruction, real_bpp, qp = batch_data
                clips = clips.to(device, non_blocking=True)
                real_reconstruction = real_reconstruction.to(device, non_blocking=True)
                real_bpp = real_bpp.to(device, non_blocking=True)
                qp = qp.to(device, non_blocking=True)
                qp_display = "mixed"
            else:
                clips, _ = batch_data
                qp = random.choice(args.qps) if training else args.qps[step % len(args.qps)]
                assert real_codec is not None
                real_codec.set_qp(qp)
                real_reconstruction, real_bpp = real_codec(clips)
                clips = clips.to(device, non_blocking=True)
                real_reconstruction = real_reconstruction.to(device, non_blocking=True)
                real_bpp = real_bpp.to(device, non_blocking=True)
                qp_display = str(qp)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                proxy_reconstruction, proxy_bpp = proxy(clips, qp)
            reconstruction_loss = F.l1_loss(
                proxy_reconstruction.float(), real_reconstruction.float()
            )
            rate_loss = rate_fit_loss(proxy_bpp, real_bpp, args.rate_loss)
            delta_loss = rate_loss.new_zeros(())
            qp_values = torch.as_tensor(qp, device=device).flatten().expand(clips.shape[0])
            predicted_rates, measured_rates = proxy_bpp, real_bpp
            metric_qps = qp_values
            probe_clips, probe_bpp = clips, real_bpp
            if args.pair_strengths is not None:
                assert real_codec is not None
                # Validation uses a fixed strength schedule, independent of the epoch.
                offset = epoch if training else 0
                strength = args.pair_strengths[(step + offset) % len(args.pair_strengths)]
                with torch.no_grad():
                    variant = clips if preprocessor is None else preprocessor(clips, qp_values)
                    variant = ((1 - strength) * variant + strength * spatial_lowpass(variant)).detach()
                    variant_reconstruction, variant_bpp = mixed_qp_roundtrip(real_codec, variant, qp_values)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    predicted_reconstruction, predicted_bpp = proxy(variant, qp_values)
                reconstruction_loss = 0.5 * (reconstruction_loss + F.l1_loss(
                    predicted_reconstruction.float(), variant_reconstruction.float()))
                rate_loss = 0.5 * (rate_loss + rate_fit_loss(predicted_bpp, variant_bpp, args.rate_loss))
                delta_loss = rate_delta_loss(proxy_bpp, predicted_bpp, real_bpp, variant_bpp)
                predicted_delta = log_rate(predicted_bpp.detach()) - log_rate(proxy_bpp.detach())
                measured_delta = log_rate(variant_bpp) - log_rate(real_bpp)
                informative = measured_delta.abs() > 0.01
                if bool(informative.any()):
                    direction = (predicted_delta[informative].sign() == measured_delta[informative].sign())
                    meters["pair_direction_accuracy"].update(float(direction.float().mean()), int(informative.sum()))
                predicted_rates = torch.cat((proxy_bpp, predicted_bpp))
                measured_rates = torch.cat((real_bpp, variant_bpp))
                metric_qps = qp_values.repeat(2)
                probe_clips, probe_bpp = variant, variant_bpp
                variant_error = 100 * (predicted_bpp.detach().float() / variant_bpp - 1).abs()
                for current_qp in args.qps:
                    selected = variant_error[qp_values == current_qp]
                    if selected.numel():
                        meters[f"qp{current_qp}_variant_rate_mape_percent"].update(
                            float(selected.mean()), selected.numel())
            loss = reconstruction_loss + args.rate_weight * rate_loss + args.rate_delta_weight * delta_loss
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(proxy.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch = clips.shape[0]
            values = torch.stack(
                (loss.detach(), reconstruction_loss.detach(), rate_loss.detach())
            ).float().cpu()
            meters["loss"].update(values[0].item(), batch)
            meters["reconstruction"].update(values[1].item(), batch)
            meters["rate"].update(values[2].item(), batch)
            meters["rate_delta"].update(float(delta_loss.detach()), batch)
            relative_error = 100 * (predicted_rates.detach().float() / measured_rates - 1).abs()
            meters["rate_mape_percent"].update(float(relative_error.mean()), len(relative_error))
            for current_qp in args.qps:
                selected = relative_error[metric_qps == current_qp]
                if selected.numel():
                    meters[f"qp{current_qp}_rate_mape_percent"].update(float(selected.mean()), selected.numel())
            if not training and step < args.gradient_probe_batches:
                assert real_codec is not None
                diagnostics = probe_rate_descent(proxy, real_codec, probe_clips, qp_values,
                                                 probe_bpp, args.gradient_probe_step)
                for name, value in diagnostics.items():
                    meters[name].update(value, batch)
            iterator.set_postfix(loss=f"{meters['loss'].average:.4f}", qp=qp_display)
    return {name: meter.average for name, meter in meters.items() if meter.count > 0}


def main() -> None:
    args = parse_args()
    validate_run_directory(args.output_dir, args.resume)
    if args.init_checkpoint and args.resume:
        raise ValueError("choose --init-checkpoint or --resume, not both")
    if args.preprocessor_checkpoint and args.pair_strengths is None:
        raise ValueError("--preprocessor-checkpoint requires --pair-strengths (include 0 for unblurred outputs)")
    if args.pair_strengths is not None and any(
        not math.isfinite(s) or not 0 <= s <= 1 for s in args.pair_strengths
    ):
        raise ValueError("--pair-strengths must be finite values in [0, 1]")
    if args.rate_delta_weight < 0 or not math.isfinite(args.rate_delta_weight):
        raise ValueError("--rate-delta-weight must be finite and non-negative")
    if args.rate_delta_weight > 0 and args.pair_strengths is None:
        raise ValueError("--rate-delta-weight requires paired variants via --pair-strengths")
    if args.gradient_probe_batches < 0 or not 0 < args.gradient_probe_step <= 1:
        raise ValueError("gradient probes require a non-negative batch count and step in (0, 1]")
    needs_real_codec = (not args.precomputed_root or args.pair_strengths is not None
                        or args.gradient_probe_batches > 0)
    if args.smoke_test:
        args.epochs = 1
    if args.precomputed_root and (args.data_root or args.train_dir or args.val_dir):
        raise ValueError("use --precomputed-root or raw video paths, not both")
    if not args.qps or any(qp < 0 or qp > 51 for qp in args.qps):
        raise ValueError("--qps must contain values in [0, 51]")
    if args.clip_grad <= 0:
        raise ValueError("--clip-grad must be positive")
    if args.frame_size % 2:
        raise ValueError("--frame-size must be even for yuv420p H.264/H.265")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    if needs_real_codec:
        require_ffmpeg(args.ffmpeg)
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader = make_loaders(args)
    proxy = StandardCodecProxy(
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        bottleneck_channels=args.bottleneck_channels,
        blocks_per_stage=args.blocks_per_stage,
        film_channels=args.film_channels,
        qp_step_divisor=args.qp_step_divisor,
        max_delta=args.max_delta,
    ).to(device)
    if args.init_checkpoint:
        initial_proxy = StandardCodecProxy.from_checkpoint(args.init_checkpoint)
        if initial_proxy.config != proxy.config:
            raise ValueError("--init-checkpoint architecture differs from proxy CLI arguments")
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        codec_config = initial.get("codec_config", {})
        for name, current in (("codec", args.codec), ("fps", args.fps), ("preset", args.preset)):
            if name in codec_config and codec_config[name] != current:
                raise ValueError(f"proxy initialization changed codec setting {name}")
        proxy.load_state_dict(initial_proxy.state_dict())
        del initial_proxy, initial
        print(f"[init] proxy weights from {args.init_checkpoint}; optimizer reset")
    preprocessor = None
    if args.preprocessor_checkpoint:
        saved = torch.load(args.preprocessor_checkpoint, map_location="cpu", weights_only=False)
        saved_args = saved.get("args", {})
        for name, current in (("codec", args.codec), ("codec_fps", args.fps),
                              ("codec_preset", args.preset), ("frames", args.frames),
                              ("frame_stride", args.frame_stride), ("frame_size", args.frame_size)):
            if name in saved_args and saved_args[name] != current:
                raise ValueError(f"preprocessor calibration changed {name}")
        if args.precomputed_root:
            split_config = train_loader.dataset.manifest.get("split", {})
            for name in ("seed", "val_ratio"):
                if name in saved_args and name in split_config and saved_args[name] != split_config[name]:
                    raise ValueError(f"preprocessor and codec cache use different split {name}")
        preprocessor = preprocessor_from_checkpoint(saved).to(device).requires_grad_(False).eval()
        print(f"[calibration] frozen preprocessor={args.preprocessor_checkpoint}")
    real_codec = None
    if needs_real_codec:
        real_codec = StandardVideoCodec(
            args.codec,
            args.qps[0],
            fps=args.fps,
            preset=args.preset,
            ffmpeg=args.ffmpeg,
            io_backend=args.codec_io,
            codec_workers=args.codec_workers,
            ffmpeg_threads=args.ffmpeg_threads,
        )
    optimizer = AdamW(proxy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        old_args = checkpoint.get("args", {})
        for name, default in (("rate_loss", "absolute"), ("rate_weight", 0.1),
                              ("rate_delta_weight", 0.0), ("pair_strengths", None),
                              ("preprocessor_checkpoint", None)):
            if old_args.get(name, default) != getattr(args, name):
                raise ValueError(f"resume changed {name}; use --init-checkpoint for recalibration")
        checkpoint_config = checkpoint.get("proxy_config", {})
        checkpoint_architecture = checkpoint_config.get("architecture")
        if checkpoint_architecture != StandardCodecProxy.ARCHITECTURE:
            raise ValueError(
                "--resume points to a legacy shallow proxy checkpoint. "
                "FiLM deeper-3D changes tensor shapes, so start from epoch 1 "
                "with a new --output-dir. The precomputed codec cache can be reused."
            )
        if checkpoint_config != proxy.config:
            differences = [
                f"{name}: checkpoint={checkpoint_config.get(name)!r}, "
                f"CLI={proxy.config.get(name)!r}"
                for name in sorted(set(checkpoint_config) | set(proxy.config))
                if checkpoint_config.get(name) != proxy.config.get(name)
            ]
            raise ValueError(
                "proxy architecture arguments differ from the resume checkpoint: "
                + "; ".join(differences)
            )
        proxy.load_state_dict(checkpoint["proxy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_val_loss", best_loss))

    output_dir = Path(args.output_dir)
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader,
            proxy,
            real_codec,
            args,
            device,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            preprocessor=preprocessor,
        )
        val_metrics = run_epoch(
            val_loader, proxy, real_codec, args, device, epoch=epoch,
            preprocessor=preprocessor,
        )
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch}/{args.epochs}] train={train_metrics} "
            f"valid={val_metrics} lr={current_lr:.3e}"
        )
        payload = {
            "epoch": epoch,
            "proxy": proxy.state_dict(),
            "proxy_config": proxy.config,
            "codec_config": {
                "codec": args.codec,
                "qps": list(args.qps),
                "fps": args.fps,
                "preset": args.preset,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": min(best_loss, val_metrics["loss"]),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            payload["best_val_loss"] = best_loss
            save_checkpoint(output_dir / "best.pt", payload)
            print(f"[checkpoint] new best proxy loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
