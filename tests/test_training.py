from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    build_qp_lambda_map,
    compression_loss,
    run_epoch,
    validation_task_bd_rate,
)


def test_hybrid_distortion_combines_processed_and_reconstructed_mse():
    source = torch.zeros(1, 1, 3, 2, 2)
    processed = torch.full_like(source, 0.5)
    reconstruction = torch.ones_like(source)
    bpp = torch.tensor([0.2])
    rd, distortion, reconstruction_mse, processed_mse, rate = compression_loss(
        source,
        processed,
        reconstruction,
        bpp,
        alpha=10.0,
        rate_lambda=0.05,
        reconstruction_weight=0.25,
    )
    assert reconstruction_mse == pytest.approx(1.0)
    assert processed_mse == pytest.approx(0.25)
    assert distortion == pytest.approx(0.4375)
    assert rate == pytest.approx(0.2)
    assert rd == pytest.approx(4.475)


def test_build_qp_lambda_map_expands_scalar():
    assert build_qp_lambda_map([30, 35, 40, 45], [0.05]) == {
        30: 0.05,
        35: 0.05,
        40: 0.05,
        45: 0.05,
    }


def test_build_qp_lambda_map_accepts_per_qp_values():
    assert build_qp_lambda_map(
        [30, 35, 40, 45], [0.048, 0.151, 0.386, 0.576]
    ) == {30: 0.048, 35: 0.151, 40: 0.386, 45: 0.576}


def test_build_qp_lambda_map_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one value per"):
        build_qp_lambda_map([30, 35, 40, 45], [0.05, 0.1])


class _RecordingPreprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_qps: list[int] = []

    def forward(self, clips: torch.Tensor, qp: int) -> torch.Tensor:
        self.seen_qps.append(qp)
        return clips


class _ValidationCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qp = 30

    def set_qp(self, qp: int) -> None:
        self.qp = qp

    def forward(self, clips: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bpp = torch.full((clips.shape[0],), self.qp / 100.0, device=clips.device)
        return clips, bpp


class _Analyzer(nn.Module):
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return torch.zeros((clips.shape[0], 5), device=clips.device)


class _FeatureAnalyzer(nn.Module):
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        score = clips.mean(dim=(1, 2, 3, 4), keepdim=False)
        return torch.stack((score, -score, score * 0, score * 0, score * 0), dim=1)

    def forward_with_features(
        self, clips: torch.Tensor, layers: list[str]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert layers == ["layer4"]
        features = clips.permute(0, 2, 1, 3, 4)
        return self.forward(clips), {"layer4": features}


def test_validation_runs_every_clip_at_every_qp():
    clips = torch.rand(2, 2, 3, 8, 8)
    labels = torch.zeros(2, dtype=torch.long)
    loader = DataLoader(TensorDataset(clips, labels), batch_size=1, shuffle=False)
    preprocessor = _RecordingPreprocessor()
    args = SimpleNamespace(
        amp=False,
        codec_qps=[30, 45],
        alpha=10.0,
        qp_to_rate_lambda={30: 0.05, 45: 0.1},
        accumulation_steps=1,
        clip_grad=1.0,
    )

    metrics = run_epoch(
        loader,
        preprocessor,
        _ValidationCodec(),
        _Analyzer(),
        args,
        torch.device("cpu"),
    )

    assert preprocessor.seen_qps == [30, 45, 30, 45]
    assert metrics["bpp"] == pytest.approx(0.375)
    assert metrics["qp30_bpp"] == pytest.approx(0.30)
    assert metrics["qp45_bpp"] == pytest.approx(0.45)
    assert "qp30_top1" in metrics
    assert "qp45_top1" in metrics


def test_validation_task_bd_rate_uses_top1_and_bpp_curves():
    qps = [30, 35, 40, 45]
    anchor = {}
    proposed = {}
    for qp, bpp, top1 in zip(qps, [0.4, 0.3, 0.2, 0.1], [0.7, 0.6, 0.5, 0.4]):
        anchor[f"qp{qp}_bpp"] = bpp
        anchor[f"qp{qp}_top1"] = top1
        proposed[f"qp{qp}_bpp"] = 0.8 * bpp
        proposed[f"qp{qp}_top1"] = top1
    assert validation_task_bd_rate(anchor, proposed, qps) == pytest.approx(
        -20.0, abs=1e-6
    )


def test_validation_logs_clean_kd_and_feature_terms():
    clips = torch.rand(2, 2, 3, 8, 8)
    labels = torch.zeros(2, dtype=torch.long)
    loader = DataLoader(TensorDataset(clips, labels), batch_size=2, shuffle=False)
    args = SimpleNamespace(
        amp=False,
        codec_qps=[30],
        alpha=10.0,
        qp_to_rate_lambda={30: 0.05},
        accumulation_steps=1,
        clip_grad=1.0,
        kd_weight=0.5,
        kd_temperature=2.0,
        ce_weight=1.0,
        feature_weight=0.05,
        feature_layer="layer4",
        distortion_reconstruction_weight=0.25,
        normalize_rate_by_anchor=False,
    )
    metrics = run_epoch(
        loader,
        _RecordingPreprocessor(),
        _ValidationCodec(),
        _FeatureAnalyzer(),
        args,
        torch.device("cpu"),
    )
    assert metrics["clean_top1"] == pytest.approx(1.0)
    assert metrics["kd_loss"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["feature_loss"] == pytest.approx(0.0, abs=1e-7)
