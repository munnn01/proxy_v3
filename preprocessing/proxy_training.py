"""Rate losses and real-codec checks used only while fitting the codec proxy."""

import torch
from torch.nn import functional as F


def log_rate(rate: torch.Tensor) -> torch.Tensor:
    return rate.float().clamp_min(1e-6).log()


def rate_fit_loss(predicted: torch.Tensor, measured: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "log":
        return F.smooth_l1_loss(log_rate(predicted), log_rate(measured))
    if mode == "absolute":
        return F.smooth_l1_loss(predicted.float(), measured.float())
    raise ValueError(f"unknown rate loss: {mode}")


def rate_delta_loss(
    base_predicted: torch.Tensor, variant_predicted: torch.Tensor,
    base_measured: torch.Tensor, variant_measured: torch.Tensor,
) -> torch.Tensor:
    """Match within-clip relative bitrate changes, not unrelated batch samples."""
    return F.smooth_l1_loss(
        log_rate(variant_predicted) - log_rate(base_predicted),
        log_rate(variant_measured) - log_rate(base_measured),
    )


@torch.no_grad()
def mixed_qp_roundtrip(codec, clips: torch.Tensor, qps: torch.Tensor):
    """Group FFmpeg calls by QP, retaining the original sample order."""
    qps = torch.as_tensor(qps, device=clips.device).flatten()
    if qps.numel() == 1:
        qps = qps.expand(clips.shape[0])
    if qps.numel() != clips.shape[0]:
        raise ValueError("one QP is required per clip")
    reconstruction = torch.empty_like(clips)
    rates = torch.empty(clips.shape[0], device=clips.device, dtype=torch.float32)
    previous_qp = codec.qp
    try:
        for qp in qps.unique(sorted=True).tolist():
            indices = (qps == qp).nonzero(as_tuple=True)[0]
            codec.set_qp(int(qp))
            decoded, bpp = codec(clips[indices])
            reconstruction[indices] = decoded
            rates[indices] = bpp.float()
    finally:
        codec.set_qp(previous_qp)
    return reconstruction, rates


def probe_rate_descent(proxy, codec, clips, qps, real_bpp, step_size):
    """Test one bounded proxy-rate descent step against real encoded BPP.

    This is a diagnostic, never a training update. The finite pixel step survives
    8-bit rounding; no claim is made that the discontinuous codec has a Jacobian.
    """
    with torch.enable_grad():
        source = clips.detach().float().requires_grad_(True)
        _, before = proxy(source, qps)
        gradient, = torch.autograd.grad(log_rate(before).mean(), source)
        proposal = (source - step_size * gradient.sign()).clamp(0, 1).detach()
    with torch.no_grad():
        _, after = proxy(proposal, qps)
        _, measured_after = mixed_qp_roundtrip(codec, proposal, qps)
    predicted_down = after < before.detach()
    actual_down = measured_after < real_bpp
    return {
        "probe_real_delta_percent": 100.0 * float((measured_after / real_bpp - 1).mean()),
        "probe_proxy_down_fraction": float(predicted_down.float().mean()),
        "probe_real_down_fraction": float(actual_down.float().mean()),
    }
