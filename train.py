"""Train a video preprocessor through a standard codec and frozen proxy/analyzer."""

from __future__ import annotations

import argparse
import json
import random
from copy import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from preprocessing import (
    FrozenVideoAnalyzer,
    ParallelStandardVideoCodec,
    StandardCodecProxy,
    StandardVideoCodec,
    build_preprocessor,
)
from preprocessing.data import (
    VideoFolderDataset,
    stratified_limit_indices,
    stratified_split_indices,
)
from preprocessing.evaluation import calculate_bd_rate
from preprocessing.standard_codec import require_ffmpeg
from preprocessing.utils import (
    AverageMeter,
    save_checkpoint,
    seed_everything,
    topk_correct,
    write_json,
)


def compression_loss(
    source: torch.Tensor,
    processed: torch.Tensor,
    reconstruction: torch.Tensor,
    bpp: torch.Tensor,
    alpha: float = 10.0,
    rate_lambda: float = 0.001,
    reconstruction_weight: float = 1.0,
    anchor_bpp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a task-aware hybrid rate-distortion component.

    ``reconstruction_weight=1`` reproduces the v1 objective. Lower values let
    the preprocessor remove codec-expensive detail while retaining an explicit
    post-codec fidelity term. Optional anchor normalization makes the rate term
    dimensionless; it is a relative-rate surrogate, not BD-rate itself.
    """

    reconstruction_distortion = F.mse_loss(reconstruction, source)
    processed_distortion = F.mse_loss(processed, source)
    distortion = (
        reconstruction_weight * reconstruction_distortion
        + (1.0 - reconstruction_weight) * processed_distortion
    )
    rate = bpp.mean()
    rate_term = rate if anchor_bpp is None else rate / anchor_bpp
    return (
        alpha * (distortion + rate_lambda * rate_term),
        distortion,
        reconstruction_distortion,
        processed_distortion,
        rate,
    )


def build_qp_lambda_map(
    codec_qps: list[int], rate_lambdas: list[float]
) -> dict[int, float]:
    """Expand a scalar rate lambda or map one value to every codec QP."""

    if not codec_qps:
        raise ValueError("--codec-qps must contain at least one QP")
    if len(set(codec_qps)) != len(codec_qps):
        raise ValueError("--codec-qps must not contain duplicate values")
    if not rate_lambdas:
        raise ValueError("--rate-lambda must contain at least one value")
    if any(value < 0 for value in rate_lambdas):
        raise ValueError("--rate-lambda values must be non-negative")
    if len(rate_lambdas) == 1:
        rate_lambdas = rate_lambdas * len(codec_qps)
    elif len(rate_lambdas) != len(codec_qps):
        raise ValueError(
            "--rate-lambda must be either one value or one value per --codec-qps"
        )
    return dict(zip(codec_qps, rate_lambdas, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument("--data-root", help="root containing train/ and optionally val/ folders")
    data.add_argument("--train-dir", help="explicit class-folder training directory")
    data.add_argument("--val-dir", help="explicit class-folder validation directory")
    data.add_argument("--train-split", default="train")
    data.add_argument("--val-split", default="val")
    data.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="stratified validation fraction when no validation directory exists",
    )
    data.add_argument("--frames", type=int, default=16)
    data.add_argument("--frame-stride", type=int, default=2)
    data.add_argument("--frame-size", type=int, default=128)
    data.add_argument("--limit-train", type=int)
    data.add_argument("--limit-val", type=int)
    data.add_argument("--workers", type=int, default=4)

    model = parser.add_argument_group("model")
    model.add_argument("--preprocessor", choices=("swin", "vit", "cnn"), default="swin")
    model.add_argument("--temporal-frames", type=int, default=8)
    model.add_argument("--vit-patch-size", type=int, default=8)
    model.add_argument("--vit-embed-dim", type=int, default=96)
    model.add_argument("--vit-depth", type=int, default=4)
    model.add_argument("--vit-heads", type=int, default=4)
    model.add_argument("--swin-patch-size", type=int, default=4)
    model.add_argument("--swin-embed-dim", type=int, default=48)
    model.add_argument("--swin-depth", type=int, default=4)
    model.add_argument("--swin-heads", type=int, default=4)
    model.add_argument("--swin-window-temporal", type=int, default=4)
    model.add_argument("--swin-window-spatial", type=int, default=8)
    model.add_argument(
        "--swin-qp-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="condition Video Swin features and residual strength on codec QP",
    )
    model.add_argument("--swin-qp-embed-dim", type=int, default=64)
    model.add_argument("--max-residual", type=float, default=0.25)
    model.add_argument("--analyzer", default="r3d_18")
    model.add_argument(
        "--codec-qps",
        type=int,
        nargs="+",
        default=[30, 35, 40, 45],
        help="standard-codec QPs sampled once per training batch",
    )
    model.add_argument("--codec", choices=("h264", "h265"), default="h264")
    model.add_argument("--proxy-checkpoint", required=True)
    model.add_argument("--codec-fps", type=float, default=30.0)
    model.add_argument("--codec-preset", default="medium")
    model.add_argument("--ffmpeg", default="ffmpeg")

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--epochs", type=int, default=30)
    optimization.add_argument("--batch-size", type=int, default=2)
    optimization.add_argument("--accumulation-steps", type=int, default=1)
    optimization.add_argument("--lr", type=float, default=1e-4)
    optimization.add_argument("--alpha", type=float, default=10.0)
    optimization.add_argument(
        "--rate-lambda",
        type=float,
        nargs="+",
        default=[0.001],
        help="one shared value or one value per --codec-qps, in the same order",
    )
    optimization.add_argument(
        "--distortion-reconstruction-weight",
        type=float,
        default=0.25,
        help="eta in eta*MSE(reconstruction, source)+(1-eta)*MSE(processed, source)",
    )
    optimization.add_argument("--ce-weight", type=float, default=1.0)
    optimization.add_argument(
        "--kd-weight",
        type=float,
        default=0.5,
        help="clean-video logit distillation weight; zero disables KD",
    )
    optimization.add_argument("--kd-temperature", type=float, default=2.0)
    optimization.add_argument(
        "--feature-weight",
        type=float,
        default=0.05,
        help="normalized clean/reconstruction feature matching weight; zero disables it",
    )
    optimization.add_argument(
        "--feature-layer",
        default="layer4",
        help="named frozen-analyzer module used for feature matching",
    )
    optimization.add_argument(
        "--normalize-rate-by-anchor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="divide each QP's BPP by its cached validation anchor BPP",
    )
    optimization.add_argument(
        "--qp-sampling-weights",
        type=float,
        nargs="+",
        help="optional non-negative training sampling weights matching --codec-qps",
    )
    optimization.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    optimization.add_argument("--weight-decay", type=float, default=0.0)
    optimization.add_argument("--clip-grad", type=float, default=1.0)
    optimization.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", default="checkpoints")
    runtime.add_argument("--resume")
    runtime.add_argument(
        "--checkpoint-metric",
        choices=("task_bd_rate", "loss", "top1", "ce"),
        default="task_bd_rate",
        help="metric used for best.pt; all metric-specific best files are also saved",
    )
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_data_directories(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.train_dir:
        train_dir = Path(args.train_dir)
        return train_dir, Path(args.val_dir) if args.val_dir else None
    if args.data_root:
        root = Path(args.data_root)
        train_candidate = root / args.train_split
        train_dir = train_candidate if train_candidate.is_dir() else root
        val_candidate = root / args.val_split
        val_dir = val_candidate if val_candidate.is_dir() else None
        if not train_dir.is_dir():
            raise FileNotFoundError(f"training directory does not exist: {train_dir}")
        return train_dir, val_dir
    raise ValueError("provide --data-root or --train-dir")


def make_loaders(args: argparse.Namespace, categories: list[str]) -> tuple[DataLoader, DataLoader]:
    train_dir, val_dir = resolve_data_directories(args)
    train_limit = 8 if args.smoke_test else args.limit_train
    val_limit = 4 if args.smoke_test else args.limit_val
    dataset_options = {
        "frames": args.frames,
        "stride": args.frame_stride,
        "size": args.frame_size,
    }
    if val_dir is not None:
        train_source = VideoFolderDataset(
            train_dir, categories, train=True, **dataset_options
        )
        val_source = VideoFolderDataset(
            val_dir, categories, train=False, **dataset_options
        )
        train_indices = stratified_limit_indices(
            train_source.samples,
            range(len(train_source)),
            train_limit,
            args.seed + 101,
        )
        val_indices = stratified_limit_indices(
            val_source.samples,
            range(len(val_source)),
            val_limit,
            args.seed + 202,
        )
        train_set = Subset(train_source, train_indices)
        val_set = Subset(val_source, val_indices)
    else:
        train_source = VideoFolderDataset(train_dir, categories, train=True, **dataset_options)
        val_source = copy(train_source)
        val_source.train = False
        train_indices, val_indices = stratified_split_indices(
            train_source.samples, args.val_ratio, args.seed
        )
        train_indices = stratified_limit_indices(
            train_source.samples, train_indices, train_limit, args.seed + 101
        )
        val_indices = stratified_limit_indices(
            val_source.samples, val_indices, val_limit, args.seed + 202
        )
        train_set = Subset(train_source, train_indices)
        val_set = Subset(val_source, val_indices)
        print(
            f"[data] no validation directory found; stratified split "
            f"train={len(train_set)} val={len(val_set)} ratio={args.val_ratio:.3f}"
        )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=len(train_set) >= args.batch_size, **common),
        DataLoader(val_set, shuffle=False, drop_last=False, **common),
    )


def forward_losses(
    clips: torch.Tensor,
    labels: torch.Tensor,
    preprocessor: nn.Module,
    codec: ParallelStandardVideoCodec,
    analyzer: FrozenVideoAnalyzer,
    args: argparse.Namespace,
    use_amp: bool,
    qp: int,
    clean_logits: torch.Tensor | None = None,
    clean_features: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    device_type = clips.device.type
    feature_weight = float(getattr(args, "feature_weight", 0.0))
    feature_layer = str(getattr(args, "feature_layer", "layer4"))
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
        processed = preprocessor(clips, qp)
        # Forward values come from FFmpeg; the frozen proxy supplies only the
        # reconstruction/rate Jacobian needed to update the preprocessor.
        reconstructed, bpp = codec(processed)
        if feature_weight > 0:
            logits, reconstructed_features = analyzer.forward_with_features(
                reconstructed, [feature_layer]
            )
        else:
            logits = analyzer(reconstructed)
            reconstructed_features = {}

    # Accumulate probability-derived rate and both losses in FP32. This avoids
    # precision loss without forcing the expensive codec/analyzer convolutions to FP32.
    with torch.autocast(device_type=device_type, enabled=False):
        anchor_bpp = None
        if bool(getattr(args, "normalize_rate_by_anchor", False)):
            anchor_bpp = float(args.qp_to_anchor_bpp[qp])
        (
            rd_loss,
            distortion,
            reconstruction_distortion,
            processed_distortion,
            rate,
        ) = compression_loss(
            clips.float(),
            processed.float(),
            reconstructed.float(),
            bpp.float(),
            args.alpha,
            args.qp_to_rate_lambda[qp],
            float(getattr(args, "distortion_reconstruction_weight", 1.0)),
            anchor_bpp,
        )
        ce_loss = F.cross_entropy(logits.float(), labels)
        kd_loss = logits.new_zeros((), dtype=torch.float32)
        kd_weight = float(getattr(args, "kd_weight", 0.0))
        if kd_weight > 0:
            if clean_logits is None:
                raise ValueError("KD is enabled but clean logits were not provided")
            temperature = float(getattr(args, "kd_temperature", 2.0))
            kd_loss = F.kl_div(
                F.log_softmax(logits.float() / temperature, dim=1),
                F.log_softmax(clean_logits.float() / temperature, dim=1),
                reduction="batchmean",
                log_target=True,
            ) * (temperature * temperature)

        feature_loss = logits.new_zeros((), dtype=torch.float32)
        if feature_weight > 0:
            if clean_features is None or feature_layer not in clean_features:
                raise ValueError("feature matching is enabled but clean features are missing")
            proposed_feature = F.normalize(
                reconstructed_features[feature_layer].float(), dim=1
            )
            reference_feature = F.normalize(
                clean_features[feature_layer].float(), dim=1
            )
            feature_loss = 1.0 - (
                proposed_feature * reference_feature
            ).sum(dim=1).mean()

        accuracy_loss = (
            float(getattr(args, "ce_weight", 1.0)) * ce_loss
            + kd_weight * kd_loss
            + feature_weight * feature_loss
        )
        total = rd_loss + accuracy_loss
    return {
        "total": total,
        "distortion": distortion,
        "reconstruction_distortion": reconstruction_distortion,
        "processed_distortion": processed_distortion,
        "rate": rate,
        "accuracy_loss": accuracy_loss,
        "ce_loss": ce_loss,
        "kd_loss": kd_loss,
        "feature_loss": feature_loss,
        "logits": logits,
    }


def run_epoch(
    loader: DataLoader,
    preprocessor: nn.Module,
    codec: ParallelStandardVideoCodec,
    analyzer: FrozenVideoAnalyzer,
    args: argparse.Namespace,
    device: torch.device,
    *,
    optimizer: Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    qp_rng: random.Random | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    preprocessor.train(training)
    codec.train(training)
    analyzer.eval()
    meter_names = (
        "loss",
        "distortion",
        "reconstruction_distortion",
        "processed_distortion",
        "bpp",
        "task_loss",
        "ce_loss",
        "kd_loss",
        "feature_loss",
        "clean_ce",
    )
    meters = {name: AverageMeter() for name in meter_names}
    correct1 = correct5 = examples = 0
    clean_correct1 = clean_correct5 = clean_examples = 0
    per_qp_meters = {
        qp: {name: AverageMeter() for name in ("loss", "bpp")}
        for qp in args.codec_qps
    }
    per_qp_correct = {qp: {"top1": 0, "top5": 0, "examples": 0} for qp in args.codec_qps}
    use_amp = bool(args.amp and device.type == "cuda")
    need_clean = (
        float(getattr(args, "kd_weight", 0.0)) > 0
        or float(getattr(args, "feature_weight", 0.0)) > 0
    )
    if training:
        optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(loader, desc="train" if training else "valid", leave=False)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, (clips, labels) in enumerate(iterator, start=1):
            if training:
                if qp_rng is None:
                    raise ValueError("training requires qp_rng")
                qp_weights = getattr(args, "qp_sampling_weights", None)
                batch_qps = [
                    qp_rng.choice(args.codec_qps)
                    if qp_weights is None
                    else qp_rng.choices(args.codec_qps, weights=qp_weights, k=1)[0]
                ]
            else:
                # Evaluate every validation clip at every QP. This removes the
                # clip/QP subsampling noise from checkpoint selection.
                batch_qps = args.codec_qps
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean_logits: torch.Tensor | None = None
            clean_features: dict[str, torch.Tensor] | None = None
            if need_clean:
                with torch.no_grad():
                    if float(getattr(args, "feature_weight", 0.0)) > 0:
                        clean_logits, clean_features = analyzer.forward_with_features(
                            clips, [str(getattr(args, "feature_layer", "layer4"))]
                        )
                    else:
                        clean_logits = analyzer(clips)
                        clean_features = {}
                batch = labels.numel()
                meters["clean_ce"].update(
                    float(F.cross_entropy(clean_logits.float(), labels)), batch
                )
                clean_correct1 += topk_correct(clean_logits, labels, 1)
                clean_correct5 += topk_correct(clean_logits, labels, 5)
                clean_examples += batch
            for qp in batch_qps:
                codec.set_qp(qp)
                losses = forward_losses(
                    clips,
                    labels,
                    preprocessor,
                    codec,
                    analyzer,
                    args,
                    use_amp,
                    qp,
                    clean_logits,
                    clean_features,
                )
                if training:
                    scaled_loss = losses["total"] / args.accumulation_steps
                    assert scaler is not None
                    scaler.scale(scaled_loss).backward()
                    should_step = step % args.accumulation_steps == 0 or step == len(loader)
                    if should_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(preprocessor.parameters(), args.clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                batch = labels.numel()
                loss_value = float(losses["total"].detach())
                bpp_value = float(losses["rate"].detach())
                top1 = topk_correct(losses["logits"].detach(), labels, 1)
                top5 = topk_correct(losses["logits"].detach(), labels, 5)
                meters["loss"].update(loss_value, batch)
                meters["distortion"].update(float(losses["distortion"].detach()), batch)
                meters["reconstruction_distortion"].update(
                    float(losses["reconstruction_distortion"].detach()), batch
                )
                meters["processed_distortion"].update(
                    float(losses["processed_distortion"].detach()), batch
                )
                meters["bpp"].update(bpp_value, batch)
                meters["task_loss"].update(float(losses["accuracy_loss"].detach()), batch)
                meters["ce_loss"].update(float(losses["ce_loss"].detach()), batch)
                meters["kd_loss"].update(float(losses["kd_loss"].detach()), batch)
                meters["feature_loss"].update(float(losses["feature_loss"].detach()), batch)
                correct1 += top1
                correct5 += top5
                examples += batch
                if not training:
                    per_qp_meters[qp]["loss"].update(loss_value, batch)
                    per_qp_meters[qp]["bpp"].update(bpp_value, batch)
                    per_qp_correct[qp]["top1"] += top1
                    per_qp_correct[qp]["top5"] += top5
                    per_qp_correct[qp]["examples"] += batch
            iterator.set_postfix(
                loss=f"{meters['loss'].average:.4f}",
                bpp=f"{meters['bpp'].average:.3f}",
                qp=batch_qps[0] if training else "all",
            )

    metrics = {
        "loss": meters["loss"].average,
        "distortion": meters["distortion"].average,
        "reconstruction_distortion": meters["reconstruction_distortion"].average,
        "processed_distortion": meters["processed_distortion"].average,
        "bpp": meters["bpp"].average,
        "task_loss": meters["task_loss"].average,
        "ce_loss": meters["ce_loss"].average,
        "kd_loss": meters["kd_loss"].average,
        "feature_loss": meters["feature_loss"].average,
        "top1": correct1 / max(examples, 1),
        "top5": correct5 / max(examples, 1),
    }
    if need_clean:
        metrics["clean_ce"] = meters["clean_ce"].average
        metrics["clean_top1"] = clean_correct1 / max(clean_examples, 1)
        metrics["clean_top5"] = clean_correct5 / max(clean_examples, 1)
    if not training:
        for qp in args.codec_qps:
            qp_examples = per_qp_correct[qp]["examples"]
            metrics[f"qp{qp}_loss"] = per_qp_meters[qp]["loss"].average
            metrics[f"qp{qp}_bpp"] = per_qp_meters[qp]["bpp"].average
            metrics[f"qp{qp}_top1"] = per_qp_correct[qp]["top1"] / max(qp_examples, 1)
            metrics[f"qp{qp}_top5"] = per_qp_correct[qp]["top5"] / max(qp_examples, 1)
    return metrics


def evaluate_anchor_validation(
    loader: DataLoader,
    codec: ParallelStandardVideoCodec,
    analyzer: FrozenVideoAnalyzer,
    codec_qps: list[int],
    device: torch.device,
) -> dict[str, float]:
    """Measure the fixed validation anchor once for BD-rate checkpointing."""

    codec.eval()
    analyzer.eval()
    bpp_meters = {qp: AverageMeter() for qp in codec_qps}
    correct = {
        qp: {"top1": 0, "top5": 0, "examples": 0} for qp in codec_qps
    }
    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="anchor validation", leave=False):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch = labels.numel()
            for qp in codec_qps:
                codec.set_qp(qp)
                reconstructed, bpp = codec(clips, use_proxy_gradient=False)
                logits = analyzer(reconstructed)
                bpp_meters[qp].update(float(bpp.float().mean()), batch)
                correct[qp]["top1"] += topk_correct(logits, labels, 1)
                correct[qp]["top5"] += topk_correct(logits, labels, 5)
                correct[qp]["examples"] += batch

    metrics: dict[str, float] = {}
    for qp in codec_qps:
        examples = correct[qp]["examples"]
        metrics[f"qp{qp}_bpp"] = bpp_meters[qp].average
        metrics[f"qp{qp}_top1"] = correct[qp]["top1"] / max(examples, 1)
        metrics[f"qp{qp}_top5"] = correct[qp]["top5"] / max(examples, 1)
    return metrics


def load_or_evaluate_anchor_validation(
    output_dir: Path,
    loader: DataLoader,
    codec: ParallelStandardVideoCodec,
    analyzer: FrozenVideoAnalyzer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Reuse a compatible anchor curve or compute and cache it once."""

    cache_path = output_dir / "anchor_validation.json"
    expected_qps = list(args.codec_qps)
    expected_examples = len(loader.dataset)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("codec") == args.codec
            and cached.get("qps") == expected_qps
            and cached.get("validation_examples") == expected_examples
            and float(cached.get("val_ratio", args.val_ratio)) == float(args.val_ratio)
            and int(cached.get("seed", args.seed)) == int(args.seed)
        ):
            print(f"[anchor] reused {cache_path}")
            return {key: float(value) for key, value in cached["metrics"].items()}
        print("[anchor] cached curve does not match this run; recomputing")

    metrics = evaluate_anchor_validation(
        loader, codec, analyzer, args.codec_qps, device
    )
    write_json(
        cache_path,
        {
            "codec": args.codec,
            "qps": expected_qps,
            "validation_examples": expected_examples,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "metrics": metrics,
        },
    )
    print(f"[anchor] wrote {cache_path}")
    return metrics


def validation_task_bd_rate(
    anchor_metrics: dict[str, float],
    proposed_metrics: dict[str, float],
    codec_qps: list[int],
) -> float | None:
    """Calculate Top-1 BD-rate from per-QP validation summaries."""

    rows: list[dict[str, float | str]] = []
    for method, metrics in (
        ("anchor", anchor_metrics),
        ("preprocessed", proposed_metrics),
    ):
        for qp in codec_qps:
            rows.append(
                {
                    "method": method,
                    "bpp": metrics[f"qp{qp}_bpp"],
                    "top1_percent": 100.0 * metrics[f"qp{qp}_top1"],
                }
            )
    return calculate_bd_rate(rows, "top1_percent")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for debugging")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if not args.codec_qps or any(qp < 0 or qp > 51 for qp in args.codec_qps):
        raise ValueError("--codec-qps must contain values in [0, 51]")
    if not 0.0 <= args.distortion_reconstruction_weight <= 1.0:
        raise ValueError("--distortion-reconstruction-weight must be in [0, 1]")
    if args.ce_weight < 0 or args.kd_weight < 0 or args.feature_weight < 0:
        raise ValueError("CE, KD and feature weights must be non-negative")
    if args.kd_temperature <= 0:
        raise ValueError("--kd-temperature must be positive")
    if args.qp_sampling_weights is not None:
        if len(args.qp_sampling_weights) != len(args.codec_qps):
            raise ValueError(
                "--qp-sampling-weights must contain one value per --codec-qps"
            )
        if any(weight < 0 for weight in args.qp_sampling_weights) or not any(
            args.qp_sampling_weights
        ):
            raise ValueError("--qp-sampling-weights must be non-negative with a positive sum")
    args.qp_to_rate_lambda = build_qp_lambda_map(args.codec_qps, args.rate_lambda)
    if args.frame_size % 2:
        raise ValueError("--frame-size must be even for yuv420p H.264/H.265")
    require_ffmpeg(args.ffmpeg)
    print(f"[setup] device={device} autocast={bool(args.amp and device.type == 'cuda')}")
    print(f"[setup] rate lambdas by QP={args.qp_to_rate_lambda}")
    print(
        "[setup] task objective "
        f"CE={args.ce_weight} KD={args.kd_weight} T={args.kd_temperature} "
        f"feature={args.feature_weight}@{args.feature_layer} "
        f"distortion_eta={args.distortion_reconstruction_weight}"
    )
    analyzer = FrozenVideoAnalyzer(args.analyzer).to(device)
    train_loader, val_loader = make_loaders(args, analyzer.categories)
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
    ).to(device)
    proxy_checkpoint = torch.load(
        args.proxy_checkpoint, map_location="cpu", weights_only=False
    )
    proxy_codec_config = proxy_checkpoint.get("codec_config", {})
    trained_codec = proxy_codec_config.get("codec")
    if trained_codec and trained_codec != args.codec:
        raise ValueError(
            f"proxy was distilled for {trained_codec}, but --codec is {args.codec}"
        )
    trained_qps = set(proxy_codec_config.get("qps", []))
    missing_qps = set(args.codec_qps) - trained_qps if trained_qps else set()
    if missing_qps:
        raise ValueError(f"proxy checkpoint was not trained for QPs {sorted(missing_qps)}")
    trained_fps = proxy_codec_config.get("fps")
    if trained_fps is not None and abs(float(trained_fps) - args.codec_fps) > 1e-6:
        raise ValueError(
            f"proxy was distilled at {trained_fps} fps, but --codec-fps is {args.codec_fps}"
        )
    trained_preset = proxy_codec_config.get("preset")
    if trained_preset and trained_preset != args.codec_preset:
        raise ValueError(
            f"proxy was distilled with preset {trained_preset}, "
            f"but --codec-preset is {args.codec_preset}"
        )
    proxy_args = proxy_checkpoint.get("args", {})
    for name in ("frames", "frame_stride", "frame_size"):
        trained_value = proxy_args.get(name)
        current_value = getattr(args, name)
        if trained_value is not None and int(trained_value) != int(current_value):
            raise ValueError(
                f"proxy was distilled with {name}={trained_value}, "
                f"but preprocessor training uses {current_value}"
            )
    proxy = StandardCodecProxy.from_checkpoint(args.proxy_checkpoint).to(device)
    standard_codec = StandardVideoCodec(
        args.codec,
        args.codec_qps[0],
        fps=args.codec_fps,
        preset=args.codec_preset,
        ffmpeg=args.ffmpeg,
    )
    codec = ParallelStandardVideoCodec(standard_codec, proxy).to(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_metrics = load_or_evaluate_anchor_validation(
        output_dir, val_loader, codec, analyzer, args, device
    )
    args.qp_to_anchor_bpp = {
        qp: anchor_metrics[f"qp{qp}_bpp"] for qp in args.codec_qps
    }
    print(f"[setup] validation anchor BPP by QP={args.qp_to_anchor_bpp}")
    if args.normalize_rate_by_anchor:
        print("[setup] rate normalization by validation anchor BPP is enabled")
    optimizer_class = AdamW if args.optimizer == "adamw" else Adam
    optimizer = optimizer_class(
        preprocessor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_loss = float("inf")
    best_ce = float("inf")
    best_top1 = float("-inf")
    best_task_bd_rate = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        preprocessor.load_state_dict(checkpoint["preprocessor"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_val_loss", best_loss))
        best_ce = float(checkpoint.get("best_val_ce", best_ce))
        best_top1 = float(checkpoint.get("best_val_top1", best_top1))
        best_task_bd_rate = float(
            checkpoint.get("best_val_task_bd_rate", best_task_bd_rate)
        )
        print(
            f"[resume] epoch={start_epoch} best_loss={best_loss:.6f} "
            f"best_top1={best_top1:.4%} "
            f"best_task_bd_rate={best_task_bd_rate:+.3f}%"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        qp_rng = random.Random(args.seed + 17 + epoch)
        print(
            f"\n[epoch {epoch}/{args.epochs}] {args.codec.upper()} "
            f"mixed QPs={args.codec_qps} lambdas={args.qp_to_rate_lambda}"
        )
        train_metrics = run_epoch(
            train_loader,
            preprocessor,
            codec,
            analyzer,
            args,
            device,
            optimizer=optimizer,
            scaler=scaler,
            qp_rng=qp_rng,
        )
        val_metrics = run_epoch(val_loader, preprocessor, codec, analyzer, args, device)
        task_bd_rate = validation_task_bd_rate(
            anchor_metrics, val_metrics, args.codec_qps
        )
        val_metrics["task_bd_rate_percent"] = task_bd_rate
        scheduler.step(val_metrics["loss"])
        print(f"train={train_metrics}")
        print(f"valid={val_metrics}")

        new_best_loss = val_metrics["loss"] < best_loss
        new_best_ce = val_metrics["ce_loss"] < best_ce
        new_best_top1 = val_metrics["top1"] > best_top1
        new_best_task_bd_rate = (
            task_bd_rate is not None and task_bd_rate < best_task_bd_rate
        )
        if new_best_loss:
            best_loss = val_metrics["loss"]
        if new_best_ce:
            best_ce = val_metrics["ce_loss"]
        if new_best_top1:
            best_top1 = val_metrics["top1"]
        if new_best_task_bd_rate:
            best_task_bd_rate = float(task_bd_rate)

        payload = {
            "epoch": epoch,
            "preprocessor": preprocessor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_loss,
            "best_val_ce": best_ce,
            "best_val_top1": best_top1,
            "best_val_task_bd_rate": best_task_bd_rate,
            "anchor_val_metrics": anchor_metrics,
            "codec": args.codec,
            "codec_qp": args.codec_qps[len(args.codec_qps) // 2],
            "codec_qps": list(args.codec_qps),
            "rate_lambdas_by_qp": dict(args.qp_to_rate_lambda),
            "proxy_checkpoint": str(args.proxy_checkpoint),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        metric_updates = {
            "loss": new_best_loss,
            "ce": new_best_ce,
            "top1": new_best_top1,
            "task_bd_rate": new_best_task_bd_rate,
        }
        metric_filenames = {
            "loss": "best_loss.pt",
            "ce": "best_ce.pt",
            "top1": "best_top1.pt",
            "task_bd_rate": "best_task_bd_rate.pt",
        }
        for metric, improved in metric_updates.items():
            if improved:
                save_checkpoint(output_dir / metric_filenames[metric], payload)

        primary_improved = metric_updates[args.checkpoint_metric]
        if (
            args.checkpoint_metric == "task_bd_rate"
            and task_bd_rate is None
            and not (output_dir / "best.pt").exists()
        ):
            primary_improved = new_best_loss
            print("[checkpoint] Task BD-rate undefined; best.pt temporarily uses val loss")
        if primary_improved:
            save_checkpoint(output_dir / "best.pt", payload)
            primary_metric_key = {
                "loss": "loss",
                "ce": "ce_loss",
                "top1": "top1",
            }.get(args.checkpoint_metric, "loss")
            primary_value = (
                f"{task_bd_rate:+.3f}%"
                if args.checkpoint_metric == "task_bd_rate" and task_bd_rate is not None
                else f"{val_metrics[primary_metric_key]:.6f}"
            )
            print(
                f"[checkpoint] new best {args.checkpoint_metric}: {primary_value}"
            )


if __name__ == "__main__":
    main()
