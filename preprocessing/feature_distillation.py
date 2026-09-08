"""Feature fidelity for a frozen task network; no extra inference network."""

import math
from types import SimpleNamespace

import torch
from torch.nn import functional as F


def feature_configuration(args):
    """Resolve old single-layer options and new weighted multi-layer options."""
    layers = getattr(args, "feature_layers", None)
    layers = list(layers) if layers is not None else [getattr(args, "feature_layer", "layer4")]
    if not layers or len(set(layers)) != len(layers) or any(not x for x in layers):
        raise ValueError("feature layers must be nonempty and unique")
    weights = getattr(args, "feature_layer_weights", None)
    weights = list(weights) if weights is not None else [1.0] * len(layers)
    if (len(weights) != len(layers) or any(not math.isfinite(x) or x < 0 for x in weights)
            or not math.isfinite(sum(weights)) or sum(weights) <= 0):
        raise ValueError("feature layer weights must match layers, be finite/nonnegative with positive sum")
    mode = getattr(args, "feature_loss", "cosine")
    if mode not in {"cosine", "mse", "relative_mse"}:
        raise ValueError(f"unknown feature loss: {mode}")
    return layers, [w / sum(weights) for w in weights], mode


def feature_weight_map(args):
    """Absolute feature coefficients, aligned with codec_qps; scalar fallback for old runs."""
    qps = list(getattr(args, "codec_qps", [30, 35, 40, 45]))
    if not qps or len(set(qps)) != len(qps):
        raise ValueError("feature weights require nonempty, unique codec QPs")
    weights = getattr(args, "feature_weights_by_qp", None)
    weights = list(weights) if weights is not None else [float(getattr(args, "feature_weight", 0.0))] * len(qps)
    if len(weights) != len(qps) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("--feature-weights-by-qp requires one finite nonnegative weight per --codec-qps entry")
    return dict(zip(qps, weights, strict=True))


def feature_weight_for_qp(args, qp):
    if getattr(args, "feature_weights_by_qp", None) is None:
        return float(getattr(args, "feature_weight", 0.0))
    return feature_weight_map(args)[qp]


def feature_enabled(args):
    return any(w > 0 for w in feature_weight_map(args).values())


def validate_feature_resume(saved, current):
    """Changing the objective requires a fresh optimizer/controller run."""
    previous = SimpleNamespace(**saved)
    if (feature_weight_map(previous) != feature_weight_map(current)
            or feature_configuration(previous) != feature_configuration(current)):
        raise ValueError("resume changed feature objective; use --init-checkpoint in a new output directory")


def feature_distillation(proposed, reference, layers, weights, mode):
    """Weighted layer means with detached clean targets, evaluated in FP32.

    relative_mse preserves feature magnitudes. Each clip/layer MSE is divided
    by that clean clip/layer's mean squared activation (floor 1e-6), so neither
    layer size nor scale alone dominates the aggregation. This normalization
    is our adaptation, not the unnormalized MSE used in the source papers.
    """
    terms = {}
    for layer in layers:
        if layer not in proposed or layer not in reference:
            raise ValueError(f"feature matching is enabled but clean/reconstructed features are missing: {layer}")
        student, teacher = proposed[layer].float(), reference[layer].detach().float()
        if student.shape != teacher.shape or student.ndim < 2:
            raise ValueError(f"feature shape mismatch at {layer}")
        if mode == "cosine":
            loss = 1.0 - (F.normalize(student, dim=1) * F.normalize(teacher, dim=1)).sum(dim=1).mean()
        elif mode == "mse":
            loss = F.mse_loss(student, teacher)
        elif mode == "relative_mse":
            axes = tuple(range(1, student.ndim))
            error = (student - teacher).square().mean(dim=axes)
            energy = teacher.square().mean(dim=axes).clamp_min(1e-6)
            loss = (error / energy).mean()
        else:
            raise ValueError(f"unknown feature loss: {mode}")
        terms[layer] = loss
    total = sum(weight * terms[layer] for layer, weight in zip(layers, weights, strict=True))
    return total, terms
