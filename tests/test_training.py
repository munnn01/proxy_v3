import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    PresetArgumentParser,
    analyzer_view_box,
    build_qp_lambda_map,
    build_rate_weight,
    clean_feature_layers,
    compression_loss,
    masked_total_variation,
    parse_args,
    run_epoch,
    validation_task_bd_rate,
)

PRESET_DIR = Path(__file__).resolve().parents[1] / "presets"


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
    assert metrics["mask_rate"] == pytest.approx(0.0)


def test_analyzer_view_box_follows_the_torchvision_video_preset():
    transform = SimpleNamespace(resize_size=(128, 171), crop_size=(112, 112))
    # The preset stretches 128x128 to 128x171 and then center-crops 112x112, so the
    # analyzer only ever sees the middle 84 of 128 source columns.
    assert analyzer_view_box(transform, 128, 128) == (8, 22, 112, 84)


def test_analyzer_view_box_supports_shorter_side_resizing():
    transform = SimpleNamespace(resize_size=(128,), crop_size=(112,))
    assert analyzer_view_box(transform, 128, 256) == (8, 72, 112, 112)


def test_analyzer_view_box_is_none_without_preset_attributes():
    assert analyzer_view_box(object(), 128, 128) is None
    assert analyzer_view_box(None, 128, 128) is None


def test_rate_weight_protects_the_most_active_region():
    feature = torch.zeros(1, 1, 1, 2, 2)
    feature[0, 0, 0, 0, 0] = 1.0
    weight = build_rate_weight(feature, 1, 2, 2)
    assert weight.shape == (1, 1, 1, 2, 2)
    assert float(weight[0, 0, 0, 0, 0]) == pytest.approx(0.0)
    assert float(weight[0, 0, 0, 1, 1]) == pytest.approx(1.0)


def test_rate_weight_fills_the_border_outside_the_analyzer_box():
    feature = torch.zeros(1, 1, 1, 2, 2)
    feature[0, 0, 0, 0, 0] = 1.0
    weight = build_rate_weight(
        feature, 1, 4, 4, box=(1, 1, 2, 2), outside_weight=0.25
    )
    assert weight.shape == (1, 1, 1, 4, 4)
    # Min-max normalization can only produce 0 or 1 inside for a two-valued feature.
    assert float(weight[0, 0, 0, 1, 1]) == pytest.approx(0.0)
    assert float(weight[0, 0, 0, 2, 2]) == pytest.approx(1.0)
    for row, column in ((0, 0), (0, 3), (3, 0), (3, 3), (0, 2), (2, 0)):
        assert float(weight[0, 0, 0, row, column]) == pytest.approx(0.25)


def test_rate_weight_gamma_shrinks_the_protected_area():
    feature = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 1, 3)
    plain = build_rate_weight(feature, 1, 1, 3, gamma=1.0)
    sharpened = build_rate_weight(feature, 1, 1, 3, gamma=2.0)
    assert float(plain[0, 0, 0, 0, 1]) == pytest.approx(0.5)
    assert float(sharpened[0, 0, 0, 0, 1]) == pytest.approx(0.75)


def test_masked_total_variation_averages_over_weighted_differences():
    target = torch.zeros(1, 1, 1, 2, 2)
    target[0, 0, 0, 0, 1] = 1.0
    ones = torch.ones(1, 1, 1, 2, 2)
    # One horizontal and one vertical step of size 1 over four counted pairs.
    assert masked_total_variation(target, ones) == pytest.approx(0.5)
    assert masked_total_variation(target, torch.zeros_like(ones)) == pytest.approx(0.0)


def test_masked_total_variation_skips_pairs_touching_a_protected_pixel():
    target = torch.zeros(1, 1, 1, 1, 3)
    target[0, 0, 0, 0, 1] = 1.0
    weight = torch.ones(1, 1, 1, 1, 3)
    weight[0, 0, 0, 0, 1] = 0.0
    # Both differences border the protected pixel, so nothing is charged.
    assert masked_total_variation(target, weight) == pytest.approx(0.0)


def test_masked_total_variation_penalizes_temporal_change():
    target = torch.zeros(1, 2, 1, 1, 1)
    target[0, 1] = 1.0
    assert masked_total_variation(target, torch.ones(1, 2, 1, 1, 1)) == pytest.approx(1.0)


def test_clean_feature_layers_deduplicates_and_follows_weights():
    off = SimpleNamespace(feature_weight=0.0, mask_rate_weight=0.0)
    assert clean_feature_layers(off) == []
    both = SimpleNamespace(
        feature_weight=0.05,
        feature_layer="layer4",
        mask_rate_weight=1.0,
        mask_rate_layer="layer4",
    )
    assert clean_feature_layers(both) == ["layer4"]
    split = SimpleNamespace(
        feature_weight=0.05,
        feature_layer="layer4",
        mask_rate_weight=1.0,
        mask_rate_layer="layer3",
    )
    assert clean_feature_layers(split) == ["layer4", "layer3"]


def _mask_rate_args(target: str) -> SimpleNamespace:
    return SimpleNamespace(
        amp=False,
        codec_qps=[30],
        alpha=10.0,
        qp_to_rate_lambda={30: 0.05},
        accumulation_steps=1,
        clip_grad=1.0,
        kd_weight=0.0,
        ce_weight=1.0,
        feature_weight=0.0,
        feature_layer="layer4",
        distortion_reconstruction_weight=1.0,
        normalize_rate_by_anchor=False,
        mask_rate_weight=1.0,
        mask_rate_layer="layer4",
        mask_rate_target=target,
        mask_rate_gamma=1.0,
        mask_rate_outside_weight=1.0,
    )


def _run_mask_rate_epoch(target: str) -> dict[str, float]:
    torch.manual_seed(0)
    clips = torch.rand(2, 2, 3, 8, 8)
    labels = torch.zeros(2, dtype=torch.long)
    loader = DataLoader(TensorDataset(clips, labels), batch_size=2, shuffle=False)
    return run_epoch(
        loader,
        _RecordingPreprocessor(),
        _ValidationCodec(),
        _FeatureAnalyzer(),
        args=_mask_rate_args(target),
        device=torch.device("cpu"),
    )


def test_masked_rate_penalty_is_zero_for_an_identity_residual():
    assert _run_mask_rate_epoch("residual")["mask_rate"] == pytest.approx(0.0)


def test_masked_rate_penalty_charges_detail_in_the_output():
    assert _run_mask_rate_epoch("output")["mask_rate"] > 0.01


def test_preset_lines_drop_comments_and_split_inline_values():
    parser = PresetArgumentParser()
    assert parser.convert_arg_line_to_args("--rate-lambda 0.05  # tuned") == [
        "--rate-lambda",
        "0.05",
    ]
    assert parser.convert_arg_line_to_args("# only a comment") == []
    assert parser.convert_arg_line_to_args("   ") == []


@pytest.mark.parametrize(
    "preset", sorted(PRESET_DIR.glob("*.args")), ids=lambda path: path.name
)
def test_shipped_presets_parse(preset, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            f"@{preset}",
            "--data-root",
            "data",
            "--proxy-checkpoint",
            "proxy.pt",
        ],
    )
    args = parse_args()
    assert args.codec_qps == [30, 35, 40, 45]
    assert args.qp_sampling_weights is None
    assert args.checkpoint_metric == "task_bd_rate"
