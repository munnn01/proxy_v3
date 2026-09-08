"""Train a video preprocessor through a standard codec and frozen proxy/analyzer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from copy import copy, deepcopy
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
from preprocessing.feature_distillation import (
    feature_configuration, feature_distillation, validate_feature_resume,
)
from preprocessing.standard_codec import require_ffmpeg
from preprocessing.utils import (
    AverageMeter,
    save_checkpoint,
    seed_everything,
    topk_correct,
    write_json,
    validate_run_directory,
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


def analyzer_view_box(
    transform: object, height: int, width: int
) -> tuple[int, int, int, int] | None:
    """Return ``(top, left, view_h, view_w)`` of the analyzer view in source pixels.

    Torchvision video presets resize a clip to ``resize_size`` and then center-crop
    to ``crop_size``, so the frozen analyzer never sees the frame border. Pixels
    outside that box cannot change the prediction, yet they still cost bits.
    Returns ``None`` for transforms that do not expose the preset attributes.
    """

    resize_size = getattr(transform, "resize_size", None)
    crop_size = getattr(transform, "crop_size", None)
    if not resize_size or not crop_size:
        return None
    resize = [int(value) for value in resize_size]
    if len(resize) >= 2:
        resized_h, resized_w = resize[0], resize[1]
    elif height <= width:
        resized_h, resized_w = resize[0], max(1, round(width * resize[0] / height))
    else:
        resized_h, resized_w = max(1, round(height * resize[0] / width)), resize[0]
    crop = [int(value) for value in crop_size]
    crop_h, crop_w = (crop[0], crop[1]) if len(crop) >= 2 else (crop[0], crop[0])
    crop_h, crop_w = min(crop_h, resized_h), min(crop_w, resized_w)
    scale_y, scale_x = height / resized_h, width / resized_w
    top = int(round(((resized_h - crop_h) // 2) * scale_y))
    left = int(round(((resized_w - crop_w) // 2) * scale_x))
    view_h = max(1, min(height - top, int(round(crop_h * scale_y))))
    view_w = max(1, min(width - left, int(round(crop_w * scale_x))))
    return top, left, view_h, view_w


def build_rate_weight(
    feature: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    *,
    box: tuple[int, int, int, int] | None = None,
    gamma: float = 1.0,
    outside_weight: float = 1.0,
) -> torch.Tensor:
    """Return detached per-pixel rate weights in ``[0, 1]`` shaped ``[B,T,1,H,W]``.

    ``feature`` is a frozen-analyzer activation ``[B,C,T',H',W']`` measured on the
    clean clip, so the weight is a constant and no gradient reaches it. High
    activation energy means the analyzer relies on that region, so its weight
    approaches zero and the rate penalty leaves it alone. Pixels outside the
    analyzer's own crop receive ``outside_weight``.
    """

    energy = feature.detach().float().abs().mean(dim=1, keepdim=True)
    inner_h, inner_w = (height, width) if box is None else (box[2], box[3])
    energy = F.interpolate(
        energy, size=(frames, inner_h, inner_w), mode="trilinear", align_corners=False
    )
    lowest = energy.amin(dim=(2, 3, 4), keepdim=True)
    highest = energy.amax(dim=(2, 3, 4), keepdim=True)
    saliency = ((energy - lowest) / (highest - lowest).clamp_min(1e-6)).clamp(0.0, 1.0)
    if gamma != 1.0:
        saliency = saliency.pow(gamma)
    inner = 1.0 - saliency
    if box is None:
        weight = inner
    else:
        top, left, view_h, view_w = box
        weight = energy.new_full(
            (energy.shape[0], 1, frames, height, width), float(outside_weight)
        )
        weight[..., top : top + view_h, left : left + view_w] = inner
    return weight.permute(0, 2, 1, 3, 4)


def masked_total_variation(
    target: torch.Tensor, weight: torch.Tensor, *,
    temporal_weight: float = 1.0,
    temporal_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the weighted mean absolute spatio-temporal gradient of ``target``.

    H.264 spends bits on spatial and temporal detail, so this is a differentiable
    stand-in for bitrate that a per-pixel weight can steer, unlike the single
    scalar ``lambda * bpp``. Each finite difference takes the smaller of its two
    endpoint weights, so detail bordering a protected pixel stays protected.
    """

    if target.ndim != 5 or weight.ndim != 5:
        raise ValueError("masked_total_variation expects [B,T,C,H,W] tensors")
    if temporal_weight < 0 or not math.isfinite(temporal_weight):
        raise ValueError("temporal_weight must be finite and non-negative")
    temporal = target if temporal_target is None else temporal_target
    if temporal.shape != target.shape:
        raise ValueError("temporal_target must have the same shape as target")
    total = target.new_zeros(())
    counted = target.new_zeros(())
    differences = (
        (
            target[..., 1:] - target[..., :-1],
            torch.minimum(weight[..., 1:], weight[..., :-1]),
        ),
        (
            target[..., 1:, :] - target[..., :-1, :],
            torch.minimum(weight[..., 1:, :], weight[..., :-1, :]),
        ),
        (
            temporal[:, 1:] - temporal[:, :-1],
            temporal_weight * torch.minimum(weight[:, 1:], weight[:, :-1]),
        ),
    )
    for difference, pair_weight in differences:
        if difference.numel() == 0:
            continue
        total = total + (pair_weight * difference.abs()).sum()
        counted = counted + pair_weight.expand_as(difference).sum()
    return total / counted.clamp_min(1e-6)


def validation_accuracy_guard(
    anchor: dict[str, float], proposed: dict[str, float], qps: list[int],
    max_drop_pp: float | None,
) -> tuple[bool, dict[int, float]]:
    """Check per-QP Top-1 drops in percentage points, on the same real-codec split."""
    if max_drop_pp is None:
        return True, {}
    if not math.isfinite(max_drop_pp) or max_drop_pp < 0:
        raise ValueError("max_top1_drop_pp must be finite and non-negative")
    drops = {qp: 100.0 * (anchor[f"qp{qp}_top1"] - proposed[f"qp{qp}_top1"])
             for qp in qps}
    return all(math.isfinite(drop) and drop <= max_drop_pp + 1e-9
               for drop in drops.values()), drops


def clean_feature_layers(args: argparse.Namespace) -> list[str]:
    """Return the analyzer layers that must be captured on the clean clip."""

    layers: list[str] = []
    if float(getattr(args, "feature_weight", 0.0)) > 0:
        layers.extend(feature_configuration(args)[0])
    if float(getattr(args, "mask_rate_weight", 0.0)) > 0:
        layers.append(str(getattr(args, "mask_rate_layer", "layer4")))
    return list(dict.fromkeys(layers))


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


def initialize_dual_rate_state(
    codec_qps: list[int], initial_weight: float, start_epoch: int = 1
) -> dict[str, object]:
    """Create serializable per-QP state for the masked-rate dual controller."""

    if initial_weight <= 0:
        raise ValueError("dual rate control requires a positive initial mask weight")
    return {
        "start_epoch": int(start_epoch),
        "log_w_mask_by_qp": {
            int(qp): math.log(float(initial_weight)) for qp in codec_qps
        },
        "ema_ratio_by_qp": {},
        "last_ratio_by_qp": {},
    }


def dual_mask_weights(
    state: dict[str, object],
    codec_qps: list[int],
    epoch: int,
    warmup_epochs: int,
) -> dict[int, float]:
    """Return the per-QP weights used in one epoch, including linear warmup."""

    log_weights = state["log_w_mask_by_qp"]
    if not isinstance(log_weights, dict):
        raise TypeError("dual state log_w_mask_by_qp must be a mapping")
    local_epoch = max(1, int(epoch) - int(state.get("start_epoch", 1)) + 1)
    warmup_scale = (
        1.0 if warmup_epochs <= 0 else min(1.0, local_epoch / warmup_epochs)
    )
    return {
        qp: math.exp(float(log_weights[qp])) * warmup_scale for qp in codec_qps
    }


def validation_bpp_ratios(
    anchor_metrics: dict[str, float],
    proposed_metrics: dict[str, float],
    codec_qps: list[int],
) -> tuple[dict[int, float], float]:
    """Return per-QP ratios and the ratio of mean proposed/anchor BPP."""

    ratios: dict[int, float] = {}
    anchor_total = proposed_total = 0.0
    for qp in codec_qps:
        anchor_bpp = float(anchor_metrics[f"qp{qp}_bpp"])
        proposed_bpp = float(proposed_metrics[f"qp{qp}_bpp"])
        if not math.isfinite(anchor_bpp) or anchor_bpp <= 0:
            raise ValueError(f"anchor BPP at QP {qp} must be positive and finite")
        if not math.isfinite(proposed_bpp) or proposed_bpp < 0:
            raise ValueError(f"proposed BPP at QP {qp} must be non-negative and finite")
        ratios[qp] = proposed_bpp / anchor_bpp
        anchor_total += anchor_bpp
        proposed_total += proposed_bpp
    return ratios, proposed_total / anchor_total


def update_dual_rate_state(
    state: dict[str, object],
    ratios: dict[int, float],
    *,
    target_ratio: float,
    kappa: float,
    ema_beta: float,
    minimum_weight: float,
    maximum_weight: float,
) -> dict[int, float]:
    """Update log-domain multipliers and return the un-ramped next weights."""

    log_weights = state["log_w_mask_by_qp"]
    ema_ratios = state["ema_ratio_by_qp"]
    if not isinstance(log_weights, dict) or not isinstance(ema_ratios, dict):
        raise TypeError("dual state weights and EMA values must be mappings")
    minimum_log = math.log(minimum_weight)
    maximum_log = math.log(maximum_weight)
    for qp, ratio in ratios.items():
        previous_ema = ema_ratios.get(qp)
        ema = (
            float(ratio)
            if previous_ema is None
            else ema_beta * float(previous_ema) + (1.0 - ema_beta) * float(ratio)
        )
        ema_ratios[qp] = ema
        updated = float(log_weights[qp]) + kappa * (ema - target_ratio)
        log_weights[qp] = min(max(updated, minimum_log), maximum_log)
    state["last_ratio_by_qp"] = {int(qp): float(value) for qp, value in ratios.items()}
    return {qp: math.exp(float(log_weights[qp])) for qp in ratios}


RATE_DUAL_STATE_VERSION = 2


def build_rate_dual_initial_weights(
    codec_qps: list[int],
    anchor_bpp_by_qp: dict[int, float],
    *,
    alpha: float,
    parity_lambda: float,
) -> dict[int, float]:
    """Match the raw-BPP gradient coefficient of ``alpha * parity_lambda``."""

    if alpha <= 0 or parity_lambda <= 0:
        raise ValueError("rate-dual alpha and parity lambda must be positive")
    return {
        qp: alpha * parity_lambda * float(anchor_bpp_by_qp[qp]) for qp in codec_qps
    }


def initialize_rate_dual_state(
    initial_weights: dict[int, float],
    *,
    start_epoch: int,
    target_ratio: float,
    codec: str,
    train_codec_source: str,
    kappa: float,
    ema_beta: float,
    minimum_weight: float,
    maximum_weight: float,
    max_proxy_underestimate_percent: float = 10.0,
    proxy_guard_patience: int = 1,
) -> dict[str, object]:
    """Create versioned direct-rate dual state that is safe to resume."""

    if not initial_weights or any(weight <= 0 for weight in initial_weights.values()):
        raise ValueError("direct-rate dual weights must be positive")
    return {
        "version": RATE_DUAL_STATE_VERSION,
        "start_epoch": int(start_epoch),
        "target_bpp_ratio": float(target_ratio),
        "codec_qps": list(initial_weights),
        "codec": str(codec),
        "train_codec_source": str(train_codec_source),
        "kappa": float(kappa),
        "ema_beta": float(ema_beta),
        "minimum_weight": float(minimum_weight),
        "maximum_weight": float(maximum_weight),
        "max_proxy_underestimate_percent": float(max_proxy_underestimate_percent),
        "proxy_guard_patience": int(proxy_guard_patience),
        "proxy_guard_streak": 0,
        "proxy_guard_bad_qps": [],
        "last_proxy_drift_percent_by_qp": {},
        "initial_weight_by_qp": {
            int(qp): float(weight) for qp, weight in initial_weights.items()
        },
        "log_dual_weight_by_qp": {
            int(qp): math.log(float(weight)) for qp, weight in initial_weights.items()
        },
        "ema_ratio_by_qp": {},
        "last_ratio_by_qp": {},
    }


def validate_rate_dual_resume_state(
    state: dict[str, object],
    *,
    initial_weights: dict[int, float],
    codec_qps: list[int],
    target_ratio: float,
    codec: str,
    train_codec_source: str,
    kappa: float,
    ema_beta: float,
    minimum_weight: float,
    maximum_weight: float,
    max_proxy_underestimate_percent: float = 10.0,
    proxy_guard_patience: int = 1,
) -> None:
    """Reject a resumed direct-rate controller whose semantics changed."""

    if int(state.get("version", -1)) != RATE_DUAL_STATE_VERSION:
        raise ValueError("resume checkpoint has an incompatible rate-dual state version")
    if list(state.get("codec_qps", [])) != list(codec_qps):
        raise ValueError("resume checkpoint rate-dual QPs do not match --codec-qps")
    if not math.isclose(float(state.get("target_bpp_ratio", math.nan)), target_ratio):
        raise ValueError("resume target BPP ratio differs from the checkpoint")
    if state.get("codec") != codec:
        raise ValueError("resume rate-dual codec differs from the checkpoint")
    if state.get("train_codec_source") != train_codec_source:
        raise ValueError("resume train codec source differs from the checkpoint")
    for name, current in (
        ("kappa", kappa),
        ("ema_beta", ema_beta),
        ("minimum_weight", minimum_weight),
        ("maximum_weight", maximum_weight),
        ("max_proxy_underestimate_percent", max_proxy_underestimate_percent),
    ):
        if not math.isclose(float(state.get(name, math.nan)), current):
            raise ValueError(f"resume rate-dual {name} differs from the checkpoint")
    saved_initial = state.get("initial_weight_by_qp")
    if not isinstance(saved_initial, dict):
        raise ValueError("resume checkpoint has no direct-rate initial weights")
    for qp, current in initial_weights.items():
        if qp not in saved_initial or not math.isclose(
            float(saved_initial[qp]), current, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                "resume direct-rate calibration/parity coefficient differs from "
                f"the checkpoint at QP {qp}"
            )
    if int(state.get("proxy_guard_patience", -1)) != int(proxy_guard_patience):
        raise ValueError(
            "resume rate-dual proxy guard patience differs from the checkpoint"
        )


def rate_dual_weights(
    state: dict[str, object], codec_qps: list[int]
) -> dict[int, float]:
    log_weights = state["log_dual_weight_by_qp"]
    if not isinstance(log_weights, dict):
        raise TypeError("rate-dual log weights must be a mapping")
    return {qp: math.exp(float(log_weights[qp])) for qp in codec_qps}


def update_rate_dual_state(
    state: dict[str, object],
    ratios: dict[int, float],
    *,
    target_ratio: float,
    kappa: float,
    ema_beta: float,
    minimum_weight: float,
    maximum_weight: float,
) -> dict[int, float]:
    """Apply multiplicative mirror ascent to direct rate multipliers."""

    log_weights = state["log_dual_weight_by_qp"]
    ema_ratios = state["ema_ratio_by_qp"]
    if not isinstance(log_weights, dict) or not isinstance(ema_ratios, dict):
        raise TypeError("rate-dual weights and EMA values must be mappings")
    minimum_log = math.log(minimum_weight)
    maximum_log = math.log(maximum_weight)
    for qp, ratio in ratios.items():
        previous = ema_ratios.get(qp)
        ema = (
            float(ratio)
            if previous is None
            else ema_beta * float(previous) + (1.0 - ema_beta) * float(ratio)
        )
        ema_ratios[qp] = ema
        updated = float(log_weights[qp]) + kappa * (ema - target_ratio)
        log_weights[qp] = min(max(updated, minimum_log), maximum_log)
    state["last_ratio_by_qp"] = {int(qp): float(value) for qp, value in ratios.items()}
    return rate_dual_weights(state, list(ratios))


def update_rate_dual_proxy_guard(
    state: dict[str, object],
    real_ratio_by_qp: dict[int, float],
    proxy_drift_percent_by_qp: dict[int, float],
    *,
    maximum_allowed_ratio: float,
    max_underestimate_percent: float,
    patience: int,
) -> tuple[list[int], bool]:
    """Track epochs where the frozen proxy dangerously understates real BPP."""

    if max_underestimate_percent <= 0:
        raise ValueError("maximum proxy underestimation must be positive")
    if patience < 1:
        raise ValueError("rate-dual proxy guard patience must be positive")
    bad_qps = sorted(
        int(qp)
        for qp, drift in proxy_drift_percent_by_qp.items()
        if float(real_ratio_by_qp[qp]) > float(maximum_allowed_ratio)
        and float(drift) < -float(max_underestimate_percent)
    )
    streak = int(state.get("proxy_guard_streak", 0)) + 1 if bad_qps else 0
    state["proxy_guard_streak"] = streak
    state["proxy_guard_bad_qps"] = bad_qps
    state["last_proxy_drift_percent_by_qp"] = {
        int(qp): float(drift) for qp, drift in proxy_drift_percent_by_qp.items()
    }
    return bad_qps, bool(bad_qps and streak >= patience)


def should_save_primary_checkpoint(
    selected_metric_improved: bool,
    *,
    rate_dual_enabled: bool,
    new_best_feasible: bool,
) -> bool:
    """Never expose an infeasible direct-rate checkpoint as ``best.pt``."""

    return new_best_feasible if rate_dual_enabled else selected_metric_improved


def validate_resume_proxy(
    checkpoint: dict[str, object], proxy_path: str, proxy_sha256: str, *, refresh: bool
) -> bool:
    """Keep ordinary resume strict; require a verifiable new proxy for refresh."""

    old_hash = checkpoint.get("proxy_sha256")
    old_args = checkpoint.get("args", {})
    if refresh:
        if not isinstance(old_hash, str) or not old_hash:
            raise ValueError("proxy refresh requires a resume checkpoint with proxy_sha256")
        if old_hash == proxy_sha256:
            raise ValueError(
                "--refresh-proxy-on-resume requires newly calibrated proxy weights; "
                "omit the flag for ordinary continuation with the same proxy"
            )
        return True
    if old_hash is not None and old_hash != proxy_sha256:
        raise ValueError(
            "resume changed frozen proxy weights; use --refresh-proxy-on-resume "
            "after recalibration, or --init-checkpoint in a new run"
        )
    if old_args.get("proxy_checkpoint", proxy_path) != proxy_path:
        raise ValueError("resume changed --proxy-checkpoint; use the saved proxy path")
    return False


def validate_refreshed_proxy(
    state: dict[str, object],
    anchor_metrics: dict[str, float],
    proposed_metrics: dict[str, float],
    codec_qps: list[int],
    *,
    maximum_allowed_ratio: float,
    max_underestimate_percent: float,
    patience: int,
) -> tuple[dict[str, object], dict[int, float]]:
    """Reset only proxy guard history after real-codec validation accepts a refresh.

    Work on a copy so a failed validation leaves the resumed controller untouched.
    Real-rate multipliers and EMA history remain valid across proxy replacements.
    """

    ratios, _ = validation_bpp_ratios(anchor_metrics, proposed_metrics, codec_qps)
    drifts = {}
    for qp in codec_qps:
        real_bpp = float(proposed_metrics[f"qp{qp}_bpp"])
        proxy_bpp = float(proposed_metrics[f"qp{qp}_proxy_bpp"])
        if not math.isfinite(real_bpp) or real_bpp <= 0:
            raise ValueError(f"proxy refresh real BPP at QP {qp} must be positive and finite")
        if not math.isfinite(proxy_bpp) or proxy_bpp <= 0:
            raise ValueError(f"proxy refresh predicted BPP at QP {qp} must be positive and finite")
        drifts[qp] = 100.0 * (proxy_bpp / real_bpp - 1.0)
    refreshed_state = deepcopy(state)
    bad_qps, _ = update_rate_dual_proxy_guard(
        refreshed_state, ratios, drifts,
        maximum_allowed_ratio=maximum_allowed_ratio,
        max_underestimate_percent=max_underestimate_percent,
        patience=patience,
    )
    if bad_qps:
        raise RuntimeError(
            f"refreshed proxy failed real-codec validation at QPs {bad_qps}: "
            f"drift_percent={drifts}; no training step or resume checkpoint was changed"
        )
    return refreshed_state, drifts


def require_feasible_rate_dual_result(
    rate_dual_enabled: bool, best_feasible_task_bd_rate: float
) -> None:
    """Fail a completed direct-rate run that never met its codec constraint."""

    if rate_dual_enabled and not math.isfinite(best_feasible_task_bd_rate):
        raise RuntimeError(
            "rate-dual training finished without a feasible checkpoint; "
            "best.pt was intentionally not written. Inspect last.pt proxy drift, "
            "recalibrate the proxy on preprocessor outputs, and start a fresh run."
        )


def validate_rate_dual_checkpoint_metric(
    rate_dual_enabled: bool, checkpoint_metric: str
) -> None:
    """Keep constrained primary-checkpoint ordering unambiguous."""

    if rate_dual_enabled and checkpoint_metric != "task_bd_rate":
        raise ValueError(
            "--rate-dual-control requires --checkpoint-metric task_bd_rate so "
            "best.pt has unambiguous feasible-selection semantics"
        )


class PresetArgumentParser(argparse.ArgumentParser):
    """Parser whose ``@preset`` files allow comments and several tokens per line."""

    def convert_arg_line_to_args(self, arg_line: str):
        line = arg_line.split("#", 1)[0].strip()
        return line.split() if line else []


def build_parser() -> PresetArgumentParser:
    parser = PresetArgumentParser(description=__doc__, fromfile_prefix_chars="@")
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
    data.add_argument(
        "--controller-limit-val",
        type=int,
        help="stratified real-codec calibration subset used by direct-rate dual runs",
    )
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
    model.add_argument("--swin-gated-smoothing", action=argparse.BooleanOptionalAction,
                       default=False)
    model.add_argument("--swin-smoothing-max-strength", type=float, default=0.5)
    model.add_argument("--init-checkpoint", help="initialize preprocessor weights only; new run")
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
    model.add_argument(
        "--train-codec-source",
        choices=("real", "proxy"),
        default="real",
        help="use exact real-forward/proxy-backward training or fast proxy-only training",
    )

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
        "--max-top1-drop-pp", type=float,
        help="require every real-codec QP to retain anchor Top-1 within this many percentage points",
    )
    optimization.add_argument("--mask-rate-temporal-weight", type=float, default=1.0)
    optimization.add_argument("--mask-rate-temporal-target", choices=("same", "output", "residual"),
                             default="same", help="same preserves the spatial target used by legacy runs")
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
    optimization.add_argument("--feature-layers", nargs="+",
                              help="feature modules; overrides legacy --feature-layer")
    optimization.add_argument("--feature-layer-weights", nargs="+", type=float,
                              help="relative layer weights, normalized to sum to one")
    optimization.add_argument("--feature-loss", choices=("cosine", "mse", "relative_mse"),
                              default="cosine", help="feature distance; cosine preserves the legacy objective")
    optimization.add_argument(
        "--mask-rate-weight",
        type=float,
        default=0.0,
        help="saliency-masked total-variation rate penalty weight; zero disables it",
    )
    optimization.add_argument(
        "--mask-rate-layer",
        default="layer4",
        help="frozen-analyzer module whose clean activation energy builds the mask",
    )
    optimization.add_argument(
        "--mask-rate-target",
        choices=("output", "residual"),
        default="output",
        help="penalize detail in the preprocessed clip, or only detail it adds",
    )
    optimization.add_argument(
        "--mask-rate-gamma",
        type=float,
        default=1.0,
        help="exponent sharpening the saliency map; above one protects only its peaks",
    )
    optimization.add_argument(
        "--mask-rate-outside-weight",
        type=float,
        default=1.0,
        help="weight for pixels outside the analyzer crop; zero ignores that border",
    )
    optimization.add_argument(
        "--dual-rate-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="adapt an independent masked-rate weight at each QP from validation BPP",
    )
    optimization.add_argument(
        "--target-bpp-ratio",
        type=float,
        default=0.98,
        help="target preprocessed/anchor validation BPP ratio for every QP",
    )
    optimization.add_argument(
        "--dual-kappa",
        type=float,
        default=3.0,
        help="log-domain dual-controller step size",
    )
    optimization.add_argument(
        "--dual-ema-beta",
        type=float,
        default=0.8,
        help="EMA coefficient used to smooth per-QP validation BPP ratios",
    )
    optimization.add_argument(
        "--dual-warmup-epochs",
        type=int,
        default=2,
        help="linearly ramp the initial mask weight before dual updates take effect",
    )
    optimization.add_argument("--mask-rate-min", type=float, default=0.05)
    optimization.add_argument("--mask-rate-max", type=float, default=20.0)
    optimization.add_argument(
        "--dual-feasibility-tolerance",
        type=float,
        default=0.01,
        help="allowed mean-ratio slack; each QP may use twice this slack",
    )
    optimization.add_argument(
        "--rate-dual-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply the per-QP dual multiplier directly to anchor-normalized BPP",
    )
    optimization.add_argument(
        "--rate-dual-parity-lambda",
        type=float,
        default=0.05,
        help="legacy raw-BPP lambda whose initial gradient coefficient is reproduced",
    )
    optimization.add_argument("--rate-dual-kappa", type=float, default=3.0)
    optimization.add_argument("--rate-dual-ema-beta", type=float, default=0.8)
    optimization.add_argument("--rate-dual-min", type=float, default=0.0001)
    optimization.add_argument("--rate-dual-max", type=float, default=10.0)
    optimization.add_argument(
        "--rate-dual-max-proxy-underestimate-percent",
        type=float,
        default=10.0,
        help="abort when proxy BPP understates real BPP beyond this percentage",
    )
    optimization.add_argument(
        "--rate-dual-proxy-guard-patience",
        type=int,
        default=1,
        help="consecutive unsafe validation epochs allowed before aborting",
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
        help="optional training sampling weights matching --codec-qps; omit for the "
        "recommended uniform sampling",
    )
    optimization.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    optimization.add_argument("--weight-decay", type=float, default=0.0)
    optimization.add_argument("--clip-grad", type=float, default=1.0)
    optimization.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", default="checkpoints")
    runtime.add_argument("--resume")
    runtime.add_argument("--validate-initial", action="store_true",
                         help="measure the initialized checkpoint on this controller before any updates")
    runtime.add_argument("--initial-validation-only", action="store_true",
                         help="write initial_validation.json and return without training; requires --init-checkpoint")
    runtime.add_argument(
        "--refresh-proxy-on-resume", action="store_true",
        help="resume Swin/optimizer/controller with a newly calibrated frozen proxy; "
        "requires direct-rate control and real-codec validation before training",
    )
    runtime.add_argument(
        "--checkpoint-metric",
        choices=("task_bd_rate", "loss", "top1", "ce"),
        default="task_bd_rate",
        help="metric used for best.pt; all metric-specific best files are also saved",
    )
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--smoke-test", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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
    if (
        not args.smoke_test
        and args.rate_dual_control
        and args.controller_limit_val is not None
    ):
        val_limit = args.controller_limit_val
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
    *,
    codec_source: str = "real",
    report_proxy_rate: bool = False,
) -> dict[str, torch.Tensor]:
    device_type = clips.device.type
    feature_weight = float(getattr(args, "feature_weight", 0.0))
    feature_layers, feature_weights, feature_mode = feature_configuration(args)
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
        processed = preprocessor(clips, qp)
        reconstructed, bpp = codec(processed, codec_source=codec_source)
        proxy_bpp = bpp
        if report_proxy_rate and codec_source == "real":
            # Calibration validation measures drift without paying for another
            # FFmpeg call. The proxy reconstruction is intentionally discarded.
            _, proxy_bpp = codec.proxy(processed, qp)
        if feature_weight > 0:
            logits, reconstructed_features = analyzer.forward_with_features(
                reconstructed, feature_layers
            )
        else:
            logits = analyzer(reconstructed)
            reconstructed_features = {}

    # Accumulate probability-derived rate and both losses in FP32. This avoids
    # precision loss without forcing the expensive codec/analyzer convolutions to FP32.
    with torch.autocast(device_type=device_type, enabled=False):
        anchor_bpp = None
        rate_dual_enabled = bool(getattr(args, "rate_dual_control", False))
        if bool(getattr(args, "normalize_rate_by_anchor", False)) or rate_dual_enabled:
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
        rate_ratio = rate if anchor_bpp is None else rate / anchor_bpp
        proxy_rate = proxy_bpp.float().mean()
        proxy_rate_ratio = (
            proxy_rate if anchor_bpp is None else proxy_rate / anchor_bpp
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
        feature_terms = {}
        if feature_weight > 0:
            if clean_features is None:
                raise ValueError("feature matching is enabled but clean features are missing")
            feature_loss, feature_terms = feature_distillation(
                reconstructed_features, clean_features, feature_layers, feature_weights, feature_mode,
            )

        mask_rate_loss = logits.new_zeros((), dtype=torch.float32)
        qp_weights = getattr(args, "qp_to_mask_rate_weight", {})
        mask_rate_weight = float(
            qp_weights.get(qp, getattr(args, "mask_rate_weight", 0.0))
        )
        if mask_rate_weight > 0:
            mask_layer = str(getattr(args, "mask_rate_layer", "layer4"))
            if clean_features is None or mask_layer not in clean_features:
                raise ValueError(
                    "the masked rate penalty is enabled but clean features are missing"
                )
            _, frames, _, height, width = clips.shape
            rate_weight = build_rate_weight(
                clean_features[mask_layer],
                frames,
                height,
                width,
                box=analyzer_view_box(
                    getattr(analyzer, "transform", None), height, width
                ),
                gamma=float(getattr(args, "mask_rate_gamma", 1.0)),
                outside_weight=float(getattr(args, "mask_rate_outside_weight", 1.0)),
            )
            target = processed.float()
            if str(getattr(args, "mask_rate_target", "output")) == "residual":
                target = target - clips.float()
            temporal_mode = getattr(args, "mask_rate_temporal_target", "same")
            temporal_target = target
            if temporal_mode == "residual":
                temporal_target = processed.float() - clips.float()
            elif temporal_mode == "output":
                temporal_target = processed.float()
            mask_rate_loss = masked_total_variation(
                target, rate_weight,
                temporal_weight=float(getattr(args, "mask_rate_temporal_weight", 1.0)),
                temporal_target=temporal_target,
            )

        accuracy_loss = (
            float(getattr(args, "ce_weight", 1.0)) * ce_loss
            + kd_weight * kd_loss
            + feature_weight * feature_loss
        )
        fixed_objective = rd_loss + accuracy_loss + mask_rate_weight * mask_rate_loss
        rate_dual_loss = logits.new_zeros((), dtype=torch.float32)
        monitor_loss = fixed_objective
        if rate_dual_enabled:
            dual_weight = float(args.qp_to_rate_dual_weight[qp])
            rate_dual_loss = dual_weight * (
                rate_ratio - float(args.target_bpp_ratio)
            )
            monitor_weight = float(args.qp_to_rate_dual_initial_weight[qp])
            monitor_loss = fixed_objective + monitor_weight * rate_ratio
        total = fixed_objective + rate_dual_loss
    return {
        "total": total,
        "monitor_loss": monitor_loss,
        "distortion": distortion,
        "reconstruction_distortion": reconstruction_distortion,
        "processed_distortion": processed_distortion,
        "rate": rate,
        "rate_ratio": rate_ratio,
        "proxy_rate": proxy_rate,
        "proxy_rate_ratio": proxy_rate_ratio,
        "rate_dual_loss": rate_dual_loss,
        "accuracy_loss": accuracy_loss,
        "ce_loss": ce_loss,
        "kd_loss": kd_loss,
        "feature_loss": feature_loss,
        "feature_loss_weighted": feature_weight * feature_loss,
        **{f"feature_loss_{layer}": value for layer, value in feature_terms.items()},
        "mask_rate_loss": mask_rate_loss,
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
        "monitor_loss",
        "distortion",
        "reconstruction_distortion",
        "processed_distortion",
        "bpp",
        "rate_ratio",
        "proxy_bpp",
        "proxy_rate_ratio",
        "rate_dual_loss",
        "task_loss",
        "ce_loss",
        "kd_loss",
        "feature_loss",
        "mask_rate",
        "clean_ce",
    )
    meters = {name: AverageMeter() for name in meter_names}
    feature_meter_names = ["feature_loss_weighted"]
    if float(getattr(args, "feature_weight", 0.0)) > 0:
        feature_meter_names.extend(f"feature_loss_{layer}" for layer in feature_configuration(args)[0])
    meters.update({name: AverageMeter() for name in feature_meter_names})
    correct1 = correct5 = examples = 0
    clean_correct1 = clean_correct5 = clean_examples = 0
    per_qp_meters = {
        qp: {
            name: AverageMeter()
            for name in (
                "loss",
                "monitor_loss",
                "bpp",
                "rate_ratio",
                "proxy_bpp",
                "proxy_rate_ratio",
            )
        }
        for qp in args.codec_qps
    }
    per_qp_correct = {qp: {"top1": 0, "top5": 0, "examples": 0} for qp in args.codec_qps}
    use_amp = bool(args.amp and device.type == "cuda")
    clean_layers = clean_feature_layers(args)
    need_clean = float(getattr(args, "kd_weight", 0.0)) > 0 or bool(clean_layers)
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
                    if clean_layers:
                        clean_logits, clean_features = analyzer.forward_with_features(
                            clips, clean_layers
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
                    codec_source=(
                        str(getattr(args, "train_codec_source", "real"))
                        if training
                        else "real"
                    ),
                    report_proxy_rate=(
                        not training and bool(getattr(args, "rate_dual_control", False))
                    ),
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
                monitor_value = float(losses["monitor_loss"].detach())
                bpp_value = float(losses["rate"].detach())
                rate_ratio_value = float(losses["rate_ratio"].detach())
                proxy_bpp_value = float(losses["proxy_rate"].detach())
                proxy_ratio_value = float(losses["proxy_rate_ratio"].detach())
                top1 = topk_correct(losses["logits"].detach(), labels, 1)
                top5 = topk_correct(losses["logits"].detach(), labels, 5)
                meters["loss"].update(loss_value, batch)
                meters["monitor_loss"].update(monitor_value, batch)
                meters["distortion"].update(float(losses["distortion"].detach()), batch)
                meters["reconstruction_distortion"].update(
                    float(losses["reconstruction_distortion"].detach()), batch
                )
                meters["processed_distortion"].update(
                    float(losses["processed_distortion"].detach()), batch
                )
                meters["bpp"].update(bpp_value, batch)
                meters["rate_ratio"].update(rate_ratio_value, batch)
                meters["proxy_bpp"].update(proxy_bpp_value, batch)
                meters["proxy_rate_ratio"].update(proxy_ratio_value, batch)
                meters["rate_dual_loss"].update(
                    float(losses["rate_dual_loss"].detach()), batch
                )
                meters["task_loss"].update(float(losses["accuracy_loss"].detach()), batch)
                meters["ce_loss"].update(float(losses["ce_loss"].detach()), batch)
                meters["kd_loss"].update(float(losses["kd_loss"].detach()), batch)
                meters["feature_loss"].update(float(losses["feature_loss"].detach()), batch)
                for name in feature_meter_names:
                    meters[name].update(float(losses[name].detach()), batch)
                meters["mask_rate"].update(float(losses["mask_rate_loss"].detach()), batch)
                correct1 += top1
                correct5 += top5
                examples += batch
                if not training:
                    per_qp_meters[qp]["loss"].update(loss_value, batch)
                    per_qp_meters[qp]["monitor_loss"].update(monitor_value, batch)
                    per_qp_meters[qp]["bpp"].update(bpp_value, batch)
                    per_qp_meters[qp]["rate_ratio"].update(rate_ratio_value, batch)
                    per_qp_meters[qp]["proxy_bpp"].update(proxy_bpp_value, batch)
                    per_qp_meters[qp]["proxy_rate_ratio"].update(
                        proxy_ratio_value, batch
                    )
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
        "monitor_loss": meters["monitor_loss"].average,
        "distortion": meters["distortion"].average,
        "reconstruction_distortion": meters["reconstruction_distortion"].average,
        "processed_distortion": meters["processed_distortion"].average,
        "bpp": meters["bpp"].average,
        "rate_ratio": meters["rate_ratio"].average,
        "proxy_bpp": meters["proxy_bpp"].average,
        "proxy_rate_ratio": meters["proxy_rate_ratio"].average,
        "rate_dual_loss": meters["rate_dual_loss"].average,
        "task_loss": meters["task_loss"].average,
        "ce_loss": meters["ce_loss"].average,
        "kd_loss": meters["kd_loss"].average,
        "feature_loss": meters["feature_loss"].average,
        "mask_rate": meters["mask_rate"].average,
        "top1": correct1 / max(examples, 1),
        "top5": correct5 / max(examples, 1),
    }
    if need_clean:
        metrics["clean_ce"] = meters["clean_ce"].average
        metrics["clean_top1"] = clean_correct1 / max(clean_examples, 1)
        metrics["clean_top5"] = clean_correct5 / max(clean_examples, 1)
    metrics.update({name: meters[name].average for name in feature_meter_names})
    if not training:
        for qp in args.codec_qps:
            qp_examples = per_qp_correct[qp]["examples"]
            metrics[f"qp{qp}_loss"] = per_qp_meters[qp]["loss"].average
            metrics[f"qp{qp}_monitor_loss"] = per_qp_meters[qp][
                "monitor_loss"
            ].average
            metrics[f"qp{qp}_bpp"] = per_qp_meters[qp]["bpp"].average
            metrics[f"qp{qp}_rate_ratio"] = per_qp_meters[qp][
                "rate_ratio"
            ].average
            metrics[f"qp{qp}_proxy_bpp"] = per_qp_meters[qp]["proxy_bpp"].average
            metrics[f"qp{qp}_proxy_rate_ratio"] = per_qp_meters[qp][
                "proxy_rate_ratio"
            ].average
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
    validate_run_directory(args.output_dir, args.resume)
    if args.init_checkpoint and args.resume:
        raise ValueError("choose --init-checkpoint for a new run or --resume for continuation")
    if (args.validate_initial or args.initial_validation_only) and not args.init_checkpoint:
        raise ValueError("initial validation requires --init-checkpoint in a new run")
    feature_configuration(args)
    if args.refresh_proxy_on_resume and not (args.resume and args.rate_dual_control):
        raise ValueError("--refresh-proxy-on-resume requires --resume and --rate-dual-control")
    if args.max_top1_drop_pp is not None:
        if not math.isfinite(args.max_top1_drop_pp) or args.max_top1_drop_pp < 0:
            raise ValueError("--max-top1-drop-pp must be finite and non-negative")
        if args.checkpoint_metric != "task_bd_rate":
            raise ValueError("accuracy guard requires --checkpoint-metric task_bd_rate")
    if args.mask_rate_temporal_weight < 0 or not math.isfinite(args.mask_rate_temporal_weight):
        raise ValueError("--mask-rate-temporal-weight must be finite and non-negative")
    if args.swin_gated_smoothing and args.preprocessor != "swin":
        raise ValueError("--swin-gated-smoothing requires --preprocessor swin")
    guarded_selection = args.rate_dual_control or args.max_top1_drop_pp is not None
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
    if any(not math.isfinite(w) or w < 0 for w in (args.ce_weight, args.kd_weight, args.feature_weight)):
        raise ValueError("CE, KD and feature weights must be finite and non-negative")
    if args.kd_temperature <= 0:
        raise ValueError("--kd-temperature must be positive")
    if args.mask_rate_weight < 0:
        raise ValueError("--mask-rate-weight must be non-negative")
    if args.mask_rate_gamma <= 0:
        raise ValueError("--mask-rate-gamma must be positive")
    if not 0.0 <= args.mask_rate_outside_weight <= 1.0:
        raise ValueError("--mask-rate-outside-weight must be in [0, 1]")
    if args.dual_rate_control:
        if args.mask_rate_weight <= 0:
            raise ValueError("--dual-rate-control requires --mask-rate-weight > 0")
        if args.target_bpp_ratio <= 0:
            raise ValueError("--target-bpp-ratio must be positive")
        if args.dual_kappa <= 0:
            raise ValueError("--dual-kappa must be positive")
        if not 0.0 <= args.dual_ema_beta < 1.0:
            raise ValueError("--dual-ema-beta must be in [0, 1)")
        if args.dual_warmup_epochs < 0:
            raise ValueError("--dual-warmup-epochs must be non-negative")
        if args.mask_rate_min <= 0 or args.mask_rate_max < args.mask_rate_min:
            raise ValueError("mask-rate bounds must satisfy 0 < min <= max")
        if args.dual_feasibility_tolerance < 0:
            raise ValueError("--dual-feasibility-tolerance must be non-negative")
    if args.rate_dual_control:
        if args.dual_rate_control:
            raise ValueError(
                "--rate-dual-control and legacy --dual-rate-control are mutually exclusive"
            )
        validate_rate_dual_checkpoint_metric(
            args.rate_dual_control, args.checkpoint_metric
        )
        if args.target_bpp_ratio <= 0:
            raise ValueError("--target-bpp-ratio must be positive")
        if args.rate_dual_parity_lambda <= 0:
            raise ValueError("--rate-dual-parity-lambda must be positive")
        if args.rate_dual_kappa <= 0:
            raise ValueError("--rate-dual-kappa must be positive")
        if not 0.0 <= args.rate_dual_ema_beta < 1.0:
            raise ValueError("--rate-dual-ema-beta must be in [0, 1)")
        if args.rate_dual_min <= 0 or args.rate_dual_max < args.rate_dual_min:
            raise ValueError("rate-dual bounds must satisfy 0 < min <= max")
        if args.rate_dual_max_proxy_underestimate_percent <= 0:
            raise ValueError(
                "--rate-dual-max-proxy-underestimate-percent must be positive"
            )
        if args.rate_dual_proxy_guard_patience < 1:
            raise ValueError("--rate-dual-proxy-guard-patience must be positive")
        if args.controller_limit_val is not None and args.controller_limit_val < 1:
            raise ValueError("--controller-limit-val must be positive")
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
    if args.rate_dual_control and any(args.qp_to_rate_lambda.values()):
        raise ValueError(
            "direct-rate dual control requires --rate-lambda 0 to avoid counting rate twice"
        )
    if args.frame_size % 2:
        raise ValueError("--frame-size must be even for yuv420p H.264/H.265")
    require_ffmpeg(args.ffmpeg)
    print(f"[setup] device={device} autocast={bool(args.amp and device.type == 'cuda')}")
    print(f"[setup] rate lambdas by QP={args.qp_to_rate_lambda}")
    print(
        "[setup] task objective "
        f"CE={args.ce_weight} KD={args.kd_weight} T={args.kd_temperature} "
        f"feature={args.feature_weight}@{feature_configuration(args)} "
        f"distortion_eta={args.distortion_reconstruction_weight}"
    )
    if args.mask_rate_weight > 0:
        print(
            "[setup] masked rate penalty "
            f"weight={args.mask_rate_weight} target={args.mask_rate_target} "
            f"mask={args.mask_rate_layer} gamma={args.mask_rate_gamma} "
            f"outside={args.mask_rate_outside_weight}"
        )
    if args.dual_rate_control:
        print(
            "[setup] per-QP dual rate control "
            f"target={args.target_bpp_ratio:.3f} kappa={args.dual_kappa} "
            f"ema_beta={args.dual_ema_beta} warmup={args.dual_warmup_epochs} "
            f"bounds=[{args.mask_rate_min}, {args.mask_rate_max}]"
        )
    if args.rate_dual_control:
        print(
            "[setup] direct per-QP rate dual "
            f"target={args.target_bpp_ratio:.3f} kappa={args.rate_dual_kappa} "
            f"ema_beta={args.rate_dual_ema_beta} "
            f"bounds=[{args.rate_dual_min}, {args.rate_dual_max}] "
            f"train_codec={args.train_codec_source}"
        )
    if args.qp_sampling_weights is None:
        print(f"[setup] QP sampling: uniform over {args.codec_qps}")
    else:
        total_weight = float(sum(args.qp_sampling_weights))
        shares = {
            qp: round(weight / total_weight, 3)
            for qp, weight in zip(args.codec_qps, args.qp_sampling_weights, strict=True)
        }
        print(f"[setup] QP sampling shares={shares}")
        print(
            "[setup] uniform sampling is the v3 default: with four QPs the BD-rate "
            "polynomial interpolates exactly, so a tilt optimizes a fit artifact"
        )
    effective_val_limit = (
        args.controller_limit_val
        if args.rate_dual_control and args.controller_limit_val is not None
        else args.limit_val
    )
    if effective_val_limit is not None and args.checkpoint_metric == "task_bd_rate":
        print(
            f"[warn] Task BD-rate is being selected on {effective_val_limit} validation "
            "clips. Top-1 on a small subset is optimistic by roughly one point, which "
            "is a few BD points. Rank epochs here, then confirm on the full split."
        )
    analyzer = FrozenVideoAnalyzer(args.analyzer).to(device)
    view_box = analyzer_view_box(analyzer.transform, args.frame_size, args.frame_size)
    if view_box is not None:
        top, left, view_h, view_w = view_box
        share = (view_h * view_w) / float(args.frame_size * args.frame_size)
        print(
            f"[setup] analyzer view: rows {top}:{top + view_h} cols {left}:{left + view_w} "
            f"of {args.frame_size}x{args.frame_size} ({share:.1%} of every frame)"
        )
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
        swin_gated_smoothing=args.swin_gated_smoothing,
        swin_smoothing_max_strength=args.swin_smoothing_max_strength,
        max_residual=args.max_residual,
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        # A newly enabled smoothing branch starts at zero, preserving the old output.
        incompatible = preprocessor.load_state_dict(initial["preprocessor"], strict=False)
        allowed = {"smoothing_head.weight", "smoothing_head.bias"} if args.swin_gated_smoothing else set()
        if incompatible.unexpected_keys or set(incompatible.missing_keys) - allowed:
            raise ValueError(f"incompatible preprocessor initialization: {incompatible}")
        print(f"[init] preprocessor weights from {args.init_checkpoint}; optimizer/controller reset")
    proxy_sha256 = hashlib.sha256(Path(args.proxy_checkpoint).read_bytes()).hexdigest()
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
    if args.rate_dual_control:
        args.qp_to_rate_dual_initial_weight = build_rate_dual_initial_weights(
            args.codec_qps,
            args.qp_to_anchor_bpp,
            alpha=args.alpha,
            parity_lambda=args.rate_dual_parity_lambda,
        )
    else:
        args.qp_to_rate_dual_initial_weight = {
            qp: 0.0 for qp in args.codec_qps
        }
    args.qp_to_rate_dual_weight = dict(args.qp_to_rate_dual_initial_weight)
    if args.rate_dual_control:
        print(
            "[setup] direct-rate initial weights by QP="
            f"{args.qp_to_rate_dual_initial_weight}"
        )
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
    best_feasible_task_bd_rate = float("inf")
    checkpoint: dict[str, object] | None = None
    proxy_refresh_pending = False
    proxy_refresh_history: list[dict[str, object]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        old_args = checkpoint.get("args", {})
        validate_feature_resume(old_args, args)
        proxy_refresh_pending = validate_resume_proxy(
            checkpoint, args.proxy_checkpoint, proxy_sha256,
            refresh=args.refresh_proxy_on_resume,
        )
        proxy_refresh_history = deepcopy(checkpoint.get("proxy_refresh_history", []))
        for name, default in (("max_top1_drop_pp", None),
                              ("mask_rate_temporal_weight", 1.0),
                              ("mask_rate_temporal_target", "same"),
                              ("swin_gated_smoothing", False),
                              ("swin_smoothing_max_strength", 0.5),
                              ("max_residual", 0.25)):
            if old_args.get(name, default) != getattr(args, name):
                raise ValueError(f"resume changed {name}; use --init-checkpoint in a new output directory")
        preprocessor.load_state_dict(checkpoint["preprocessor"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if proxy_refresh_pending and args.epochs < start_epoch:
            raise ValueError(f"--epochs must be at least {start_epoch} to continue after proxy refresh")
        best_loss = float(checkpoint.get("best_val_loss", best_loss))
        best_ce = float(checkpoint.get("best_val_ce", best_ce))
        best_top1 = float(checkpoint.get("best_val_top1", best_top1))
        best_task_bd_rate = float(
            checkpoint.get("best_val_task_bd_rate", best_task_bd_rate)
        )
        best_feasible_task_bd_rate = float(
            checkpoint.get(
                "best_feasible_task_bd_rate", best_feasible_task_bd_rate
            )
        )
        print(
            f"[resume] epoch={start_epoch} best_loss={best_loss:.6f} "
            f"best_top1={best_top1:.4%} "
            f"best_task_bd_rate={best_task_bd_rate:+.3f}%"
        )

    dual_state: dict[str, object] | None = None
    if args.dual_rate_control:
        saved_dual_state = checkpoint.get("dual_state") if checkpoint else None
        if isinstance(saved_dual_state, dict):
            dual_state = saved_dual_state
            saved_qps = set(dual_state.get("log_w_mask_by_qp", {}))
            if saved_qps != set(args.codec_qps):
                raise ValueError(
                    "resume checkpoint dual QPs do not match --codec-qps"
                )
            print("[resume] restored per-QP dual-controller state")
        else:
            dual_state = initialize_dual_rate_state(
                args.codec_qps, args.mask_rate_weight, start_epoch
            )
            if checkpoint is not None:
                print("[resume] checkpoint has no dual state; starting controller warmup")
    args.qp_to_mask_rate_weight = {
        qp: float(args.mask_rate_weight) for qp in args.codec_qps
    }

    rate_dual_state: dict[str, object] | None = None
    if args.rate_dual_control:
        saved_rate_dual_state = (
            checkpoint.get("rate_dual_state") if checkpoint else None
        )
        if checkpoint is not None and not isinstance(saved_rate_dual_state, dict):
            raise ValueError(
                "checkpoint predates direct-rate dual V5 and cannot be resumed; "
                "start a fresh V5 run"
            )
        if isinstance(saved_rate_dual_state, dict):
            validate_rate_dual_resume_state(
                saved_rate_dual_state,
                initial_weights=args.qp_to_rate_dual_initial_weight,
                codec_qps=args.codec_qps,
                target_ratio=args.target_bpp_ratio,
                codec=args.codec,
                train_codec_source=args.train_codec_source,
                kappa=args.rate_dual_kappa,
                ema_beta=args.rate_dual_ema_beta,
                minimum_weight=args.rate_dual_min,
                maximum_weight=args.rate_dual_max,
                max_proxy_underestimate_percent=(
                    args.rate_dual_max_proxy_underestimate_percent
                ),
                proxy_guard_patience=args.rate_dual_proxy_guard_patience,
            )
            rate_dual_state = saved_rate_dual_state
            saved_initial_weights = rate_dual_state["initial_weight_by_qp"]
            assert isinstance(saved_initial_weights, dict)
            args.qp_to_rate_dual_initial_weight = {
                qp: float(saved_initial_weights[qp]) for qp in args.codec_qps
            }
            print("[resume] restored versioned direct-rate dual state")
        else:
            rate_dual_state = initialize_rate_dual_state(
                args.qp_to_rate_dual_initial_weight,
                start_epoch=start_epoch,
                target_ratio=args.target_bpp_ratio,
                codec=args.codec,
                train_codec_source=args.train_codec_source,
                kappa=args.rate_dual_kappa,
                ema_beta=args.rate_dual_ema_beta,
                minimum_weight=args.rate_dual_min,
                maximum_weight=args.rate_dual_max,
                max_proxy_underestimate_percent=(
                    args.rate_dual_max_proxy_underestimate_percent
                ),
                proxy_guard_patience=args.rate_dual_proxy_guard_patience,
            )

    if proxy_refresh_pending:
        assert checkpoint is not None and rate_dual_state is not None
        args.qp_to_rate_dual_weight = rate_dual_weights(rate_dual_state, args.codec_qps)
        print("[proxy-refresh] validating new proxy on resumed Swin with real codec before training")
        refresh_metrics = run_epoch(val_loader, preprocessor, codec, analyzer, args, device)
        rate_dual_state, refresh_drifts = validate_refreshed_proxy(
            rate_dual_state, anchor_metrics, refresh_metrics, args.codec_qps,
            maximum_allowed_ratio=(args.target_bpp_ratio + 2.0 * args.dual_feasibility_tolerance),
            max_underestimate_percent=args.rate_dual_max_proxy_underestimate_percent,
            patience=args.rate_dual_proxy_guard_patience,
        )
        refresh_event = {
            "after_epoch": start_epoch - 1,
            "previous_proxy_sha256": checkpoint["proxy_sha256"],
            "proxy_sha256": proxy_sha256,
            "proxy_checkpoint": str(args.proxy_checkpoint),
            "previous_guard_streak": checkpoint["rate_dual_state"].get("proxy_guard_streak", 0),
            "proxy_drift_percent_by_qp": refresh_drifts,
        }
        proxy_refresh_history.append(refresh_event)
        print(
            f"[proxy-refresh] accepted drift_percent={refresh_drifts}; "
            f"continuing epoch {start_epoch} with restored optimizer/scheduler/scaler "
            "and unchanged rate-dual weights/EMA"
        )

    if args.validate_initial or args.initial_validation_only:
        print("[initial] evaluating the starting Swin on the same controller before optimizer updates")
        # Avoid shifting the subsequent training sampler or augmentation RNG.
        import numpy as np
        random_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                initial_metrics = run_epoch(val_loader, preprocessor, codec, analyzer, args, device)
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)
        initial_metrics["task_bd_rate_percent"] = validation_task_bd_rate(
            anchor_metrics, initial_metrics, args.codec_qps,
        )
        ratios, mean_ratio = validation_bpp_ratios(anchor_metrics, initial_metrics, args.codec_qps)
        accuracy_ok, drops = validation_accuracy_guard(
            anchor_metrics, initial_metrics, args.codec_qps, args.max_top1_drop_pp,
        )
        initial_metrics.update({"mean_bpp_ratio": mean_ratio, "accuracy_feasible": accuracy_ok})
        initial_metrics.update({f"qp{qp}_bpp_ratio": ratio for qp, ratio in ratios.items()})
        initial_metrics.update({f"qp{qp}_top1_drop_pp": drop for qp, drop in drops.items()})
        write_json(output_dir / "initial_validation.json", {
            "checkpoint": str(Path(args.init_checkpoint).resolve()),
            "checkpoint_sha256": hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
            "proxy_sha256": proxy_sha256, "args": vars(args),
            "anchor_metrics": anchor_metrics, "val_metrics": initial_metrics,
            "optimizer_updates": 0,
        })
        print(f"[initial] valid={initial_metrics}")
        if args.initial_validation_only:
            return

    for epoch in range(start_epoch, args.epochs + 1):
        proxy_guard_abort = False
        proxy_guard_bad_qps: list[int] = []
        if dual_state is not None:
            args.qp_to_mask_rate_weight = dual_mask_weights(
                dual_state,
                args.codec_qps,
                epoch,
                args.dual_warmup_epochs,
            )
        if rate_dual_state is not None:
            args.qp_to_rate_dual_weight = rate_dual_weights(
                rate_dual_state, args.codec_qps
            )
        qp_rng = random.Random(args.seed + 17 + epoch)
        print(
            f"\n[epoch {epoch}/{args.epochs}] {args.codec.upper()} "
            f"mixed QPs={args.codec_qps} lambdas={args.qp_to_rate_lambda} "
            f"mask_weights={args.qp_to_mask_rate_weight} "
            f"rate_dual_weights={args.qp_to_rate_dual_weight}"
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
        bpp_ratios: dict[int, float] = {}
        mean_bpp_ratio: float | None = None
        accuracy_feasible, accuracy_drops = validation_accuracy_guard(
            anchor_metrics, val_metrics, args.codec_qps, args.max_top1_drop_pp
        )
        feasible = accuracy_feasible if args.max_top1_drop_pp is not None else False
        if args.max_top1_drop_pp is not None:
            val_metrics["accuracy_feasible"] = accuracy_feasible
            for qp, drop in accuracy_drops.items():
                val_metrics[f"qp{qp}_top1_drop_pp"] = drop
        if dual_state is not None or rate_dual_state is not None:
            bpp_ratios, mean_bpp_ratio = validation_bpp_ratios(
                anchor_metrics, val_metrics, args.codec_qps
            )
            for qp, ratio in bpp_ratios.items():
                val_metrics[f"qp{qp}_bpp_ratio"] = ratio
                if dual_state is not None:
                    val_metrics[f"qp{qp}_mask_rate_weight"] = (
                        args.qp_to_mask_rate_weight[qp]
                    )
                if rate_dual_state is not None:
                    val_metrics[f"qp{qp}_rate_dual_weight"] = (
                        args.qp_to_rate_dual_weight[qp]
                    )
                    proxy_ratio = val_metrics[f"qp{qp}_proxy_rate_ratio"]
                    val_metrics[f"qp{qp}_proxy_drift_percent"] = 100.0 * (
                        proxy_ratio / ratio - 1.0
                    )
            val_metrics["mean_bpp_ratio"] = mean_bpp_ratio
            tolerance = args.dual_feasibility_tolerance
            feasible = (
                accuracy_feasible
                and mean_bpp_ratio <= args.target_bpp_ratio + tolerance
                and max(bpp_ratios.values())
                <= args.target_bpp_ratio + 2.0 * tolerance
            )
            val_metrics["dual_feasible"] = feasible

            if rate_dual_state is not None:
                proxy_drifts = {
                    qp: float(val_metrics[f"qp{qp}_proxy_drift_percent"])
                    for qp in args.codec_qps
                }
                proxy_guard_bad_qps, proxy_guard_abort = update_rate_dual_proxy_guard(
                    rate_dual_state,
                    bpp_ratios,
                    proxy_drifts,
                    maximum_allowed_ratio=(
                        args.target_bpp_ratio + 2.0 * args.dual_feasibility_tolerance
                    ),
                    max_underestimate_percent=(
                        args.rate_dual_max_proxy_underestimate_percent
                    ),
                    patience=args.rate_dual_proxy_guard_patience,
                )
                val_metrics["rate_dual_proxy_guard_bad_qps"] = proxy_guard_bad_qps
                val_metrics["rate_dual_proxy_guard_streak"] = int(
                    rate_dual_state["proxy_guard_streak"]
                )
                val_metrics["rate_dual_proxy_guard_abort"] = proxy_guard_abort
                if proxy_guard_bad_qps:
                    rate_dual_state["last_ratio_by_qp"] = {
                        int(qp): float(value) for qp, value in bpp_ratios.items()
                    }
                    next_weights = rate_dual_weights(rate_dual_state, args.codec_qps)
                else:
                    next_weights = update_rate_dual_state(
                        rate_dual_state,
                        bpp_ratios,
                        target_ratio=args.target_bpp_ratio,
                        kappa=args.rate_dual_kappa,
                        ema_beta=args.rate_dual_ema_beta,
                        minimum_weight=args.rate_dual_min,
                        maximum_weight=args.rate_dual_max,
                    )
                weight_map = args.qp_to_rate_dual_weight
                drift_text = " ".join(
                    f"QP{qp}:proxy_drift="
                    f"{val_metrics[f'qp{qp}_proxy_drift_percent']:+.1f}%"
                    for qp in args.codec_qps
                )
                if proxy_guard_bad_qps:
                    drift_text += (
                        " proxy_guard=freeze"
                        f" bad_qps={proxy_guard_bad_qps}"
                        f" streak={rate_dual_state['proxy_guard_streak']}"
                    )
            else:
                assert dual_state is not None
                local_epoch = epoch - int(dual_state.get("start_epoch", 1)) + 1
                if local_epoch >= args.dual_warmup_epochs:
                    next_weights = update_dual_rate_state(
                        dual_state,
                        bpp_ratios,
                        target_ratio=args.target_bpp_ratio,
                        kappa=args.dual_kappa,
                        ema_beta=args.dual_ema_beta,
                        minimum_weight=args.mask_rate_min,
                        maximum_weight=args.mask_rate_max,
                    )
                else:
                    log_weights = dual_state["log_w_mask_by_qp"]
                    assert isinstance(log_weights, dict)
                    next_weights = {
                        qp: math.exp(float(log_weights[qp])) for qp in args.codec_qps
                    }
                weight_map = args.qp_to_mask_rate_weight
                drift_text = ""
            ratios_text = " ".join(
                f"QP{qp}:ratio={bpp_ratios[qp]:.3f} "
                f"w={weight_map[qp]:.4f} "
                f"next={next_weights[qp]:.3f}"
                for qp in args.codec_qps
            )
            print(
                f"[dual] {ratios_text} mean={mean_bpp_ratio:.3f} "
                f"target={args.target_bpp_ratio:.3f} feasible={feasible} "
                f"{drift_text}".rstrip()
            )
        scheduler_metric = (
            val_metrics["monitor_loss"]
            if args.rate_dual_control
            else val_metrics["loss"]
        )
        scheduler.step(scheduler_metric)
        print(f"train={train_metrics}")
        print(f"valid={val_metrics}")

        new_best_loss = scheduler_metric < best_loss
        new_best_ce = val_metrics["ce_loss"] < best_ce
        new_best_top1 = val_metrics["top1"] > best_top1
        new_best_task_bd_rate = (
            task_bd_rate is not None and task_bd_rate < best_task_bd_rate
        )
        new_best_feasible = (
            feasible
            and task_bd_rate is not None
            and task_bd_rate < best_feasible_task_bd_rate
        )
        if new_best_loss:
            best_loss = scheduler_metric
        if new_best_ce:
            best_ce = val_metrics["ce_loss"]
        if new_best_top1:
            best_top1 = val_metrics["top1"]
        if new_best_task_bd_rate:
            best_task_bd_rate = float(task_bd_rate)
        if new_best_feasible:
            best_feasible_task_bd_rate = float(task_bd_rate)

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
            "best_feasible_task_bd_rate": best_feasible_task_bd_rate,
            "anchor_val_metrics": anchor_metrics,
            "codec": args.codec,
            "codec_qp": args.codec_qps[len(args.codec_qps) // 2],
            "codec_qps": list(args.codec_qps),
            "rate_lambdas_by_qp": dict(args.qp_to_rate_lambda),
            "mask_rate_weights_by_qp": dict(args.qp_to_mask_rate_weight),
            "dual_state": dual_state,
            "rate_dual_weights_by_qp": dict(args.qp_to_rate_dual_weight),
            "rate_dual_state": rate_dual_state,
            "proxy_checkpoint": str(args.proxy_checkpoint),
            "proxy_sha256": proxy_sha256,
            "proxy_refresh_history": proxy_refresh_history,
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "run_status": (
                "proxy_guard_failed"
                if proxy_guard_abort
                else (
                    "has_feasible_checkpoint"
                    if math.isfinite(best_feasible_task_bd_rate)
                    else "infeasible"
                )
            ),
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if new_best_feasible:
            save_checkpoint(output_dir / "best_feasible.pt", payload)
            print(
                "[checkpoint] new best feasible: "
                f"Task BD-rate={task_bd_rate:+.3f}% "
                f"mean BPP ratio={mean_bpp_ratio} accuracy_feasible={accuracy_feasible}"
            )
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

        primary_improved = should_save_primary_checkpoint(
            metric_updates[args.checkpoint_metric],
            rate_dual_enabled=guarded_selection,
            new_best_feasible=new_best_feasible,
        )
        if (
            not guarded_selection
            and args.checkpoint_metric == "task_bd_rate"
            and task_bd_rate is None
            and not (output_dir / "best.pt").exists()
        ):
            primary_improved = new_best_loss
            print("[checkpoint] Task BD-rate undefined; best.pt temporarily uses val loss")
        if primary_improved:
            save_checkpoint(output_dir / "best.pt", payload)
            primary_metric_key = {
                "loss": "monitor_loss" if args.rate_dual_control else "loss",
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
        if proxy_guard_abort:
            raise RuntimeError(
                "rate-dual proxy guard aborted training after frozen proxy BPP "
                "underestimated real BPP by more than "
                f"{args.rate_dual_max_proxy_underestimate_percent:.1f}% at "
                f"QPs {proxy_guard_bad_qps}; last.pt contains the diagnostic state"
            )

    require_feasible_rate_dual_result(
        guarded_selection, best_feasible_task_bd_rate
    )


if __name__ == "__main__":
    main()
