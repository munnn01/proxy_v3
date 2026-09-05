from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.filters import spatial_lowpass
from preprocessing.model import build_preprocessor, preprocessor_from_checkpoint
from preprocessing.proxy_training import mixed_qp_roundtrip, rate_delta_loss, rate_fit_loss
from train import masked_total_variation, validation_accuracy_guard
from train_proxy import run_epoch
from preprocessing.utils import validate_run_directory


def small_swin(**kwargs):
    return build_preprocessor("swin", swin_embed_dim=8, swin_depth=2, swin_num_heads=2,
                              swin_window_size=(2, 2, 2), **kwargs)


def test_spatial_lowpass_preserves_constant_frames_without_temporal_blur():
    clip = torch.zeros(1, 2, 3, 5, 7)
    clip[:, 1] = 1
    assert torch.equal(spatial_lowpass(clip), clip)


def test_gated_smoothing_starts_as_identity_and_receives_gradient():
    model = small_swin(swin_gated_smoothing=True)
    clip = torch.rand(1, 2, 3, 9, 11)
    output = model(clip, 35)
    assert torch.equal(output, clip)
    output.square().mean().backward()
    assert model.smoothing_head.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.smoothing_head.bias.fill_(2)
    output = model(clip, 35)
    assert output.shape == clip.shape
    assert output.min() >= 0 and output.max() <= 1
    assert output.var() < clip.var()


@pytest.mark.parametrize("smoothing", [False, True])
def test_checkpoint_rebuild_preserves_gated_swin_output(smoothing):
    model = small_swin(swin_gated_smoothing=smoothing)
    with torch.no_grad():
        model.to_rgb.bias.fill_(0.2)
        if smoothing:
            model.smoothing_head.bias.fill_(1)
    args = dict(preprocessor="swin", swin_embed_dim=8, swin_depth=2, swin_heads=2,
                swin_window_temporal=2, swin_window_spatial=2, swin_qp_conditioning=True)
    if smoothing:
        args["swin_gated_smoothing"] = True
    restored = preprocessor_from_checkpoint({"args": args, "preprocessor": model.state_dict()})
    clip = torch.rand(1, 2, 3, 8, 8)
    assert torch.equal(model(clip, 40), restored(clip, 40))


def test_temporal_residual_penalty_does_not_penalize_source_motion():
    clip = torch.zeros(1, 2, 3, 3, 3)
    clip[:, 1] = 1
    weight = torch.ones(1, 2, 1, 3, 3)
    assert masked_total_variation(clip, weight) > 0
    assert masked_total_variation(clip, weight, temporal_target=torch.zeros_like(clip)) == 0
    assert masked_total_variation(clip, weight, temporal_weight=0) == 0
    flickering_residual = clip * 0.1
    assert masked_total_variation(clip, weight, temporal_target=flickering_residual) > 0


def test_accuracy_guard_uses_percentage_points_and_checks_every_qp():
    anchor = {"qp30_top1": 0.6, "qp45_top1": 0.4}
    proposed = {"qp30_top1": 0.7, "qp45_top1": 0.39}
    feasible, drops = validation_accuracy_guard(anchor, proposed, [30, 45], 0.5)
    assert not feasible  # A large gain at QP30 cannot hide QP45's one-point drop.
    assert drops[45] == pytest.approx(1)
    assert validation_accuracy_guard(anchor, proposed, [30, 45], 1)[0]
    proposed["qp45_top1"] = float("nan")
    assert not validation_accuracy_guard(anchor, proposed, [30, 45], 1)[0]


def test_log_rate_loss_treats_equal_relative_errors_equally():
    low = rate_fit_loss(torch.tensor([0.11]), torch.tensor([0.1]), "log")
    high = rate_fit_loss(torch.tensor([1.1]), torch.tensor([1.0]), "log")
    assert low == pytest.approx(high, abs=1e-7)


def test_rate_delta_requires_correct_within_clip_direction():
    base = torch.tensor([1.0], requires_grad=True)
    variant = torch.tensor([1.1], requires_grad=True)
    loss = rate_delta_loss(base, variant, torch.tensor([1.0]), torch.tensor([0.8]))
    loss.backward()
    assert base.grad < 0 and variant.grad > 0
    assert rate_delta_loss(torch.tensor([2.0]), torch.tensor([1.6]),
                           torch.tensor([1.0]), torch.tensor([0.8])) < 1e-10


class MeasuredCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.qp = 35
        self.seen = []

    def set_qp(self, qp):
        self.qp = qp

    def forward(self, clips):
        assert not clips.requires_grad
        self.seen.append(self.qp)
        return clips.round(), clips.mean((1, 2, 3, 4)) + self.qp / 100.0


def test_mixed_qp_codec_keeps_sample_order_and_restores_qp():
    codec = MeasuredCodec()
    clips = torch.stack([torch.full((2, 3, 4, 4), v) for v in [0.1, 0.2, 0.3]])
    _, rates = mixed_qp_roundtrip(codec, clips, torch.tensor([45, 30, 45]))
    assert rates.tolist() == pytest.approx([0.55, 0.5, 0.75])
    assert codec.seen == [30, 45]
    assert codec.qp == 35


class DifferentiableProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.1))

    def forward(self, clips, qp):
        qps = torch.as_tensor(qp, device=clips.device)
        return clips * self.scale, self.scale * clips.mean((1, 2, 3, 4)) + qps / 100.0


class FrozenPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.9), requires_grad=False)

    def forward(self, clips, qp):
        return self.scale * clips


def test_cached_pair_training_and_validation_probes_run_with_real_targets():
    clips = torch.rand(4, 2, 3, 4, 4)
    qps = torch.tensor([30, 45, 30, 45])
    rates = clips.mean((1, 2, 3, 4)) + qps / 100
    loader = DataLoader(TensorDataset(clips, clips.round(), rates, qps), batch_size=4)
    args = SimpleNamespace(qps=[30, 45], precomputed_root="cache", amp=False,
                           rate_loss="log", rate_weight=0.1, rate_delta_weight=0.1,
                           pair_strengths=[0.0, 0.3], gradient_probe_batches=1,
                           gradient_probe_step=2/255, clip_grad=1.0)
    proxy, codec, preprocessor = DifferentiableProxy(), MeasuredCodec(), FrozenPreprocessor()
    initial = proxy.scale.detach().clone()
    optimizer = torch.optim.Adam(proxy.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics = run_epoch(loader, proxy, codec, args, torch.device("cpu"), optimizer=optimizer,
                        scaler=scaler, epoch=1, preprocessor=preprocessor)
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert not torch.equal(proxy.scale.detach(), initial)
    assert preprocessor.scale.grad is None
    assert "qp30_rate_mape_percent" in metrics
    validation = run_epoch(loader, proxy, codec, args, torch.device("cpu"), preprocessor=preprocessor)
    assert validation["probe_real_delta_percent"] < 0
    assert validation["probe_real_down_fraction"] == 1.0
    assert codec.qp == 35


def test_new_run_rejects_stale_best_checkpoint(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"old result")
    with pytest.raises(ValueError, match="already contains checkpoints"):
        validate_run_directory(tmp_path, None)
    validate_run_directory(tmp_path, str(tmp_path / "last.pt"))
    with pytest.raises(ValueError, match="original"):
        validate_run_directory(tmp_path / "new", str(tmp_path / "last.pt"))


def test_paired_proxy_calibration_with_actual_h264():
    import shutil
    from preprocessing.standard_codec import StandardVideoCodec

    executable = shutil.which("ffmpeg")
    if not executable:
        executable = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    codec = StandardVideoCodec("h264", 35, ffmpeg=executable, codec_workers=1)
    clips = torch.rand(2, 2, 3, 16, 16)
    qps = torch.tensor([30, 45])
    reconstruction, rates = mixed_qp_roundtrip(codec, clips, qps)
    assert reconstruction.shape == clips.shape and bool((rates > 0).all())
    loader = DataLoader(TensorDataset(clips, reconstruction, rates, qps), batch_size=2)
    args = SimpleNamespace(qps=[30, 45], precomputed_root="cache", amp=False,
                           rate_loss="log", rate_weight=0.1, rate_delta_weight=0.1,
                           pair_strengths=[0.3], gradient_probe_batches=1,
                           gradient_probe_step=2/255, clip_grad=1.0)
    metrics = run_epoch(loader, DifferentiableProxy(), codec, args, torch.device("cpu"))
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert "qp45_variant_rate_mape_percent" in metrics
    assert "probe_real_delta_percent" in metrics
