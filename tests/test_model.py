import json

import pytest
import torch
from torch import nn

from preprocessing.data import (
    MixedQPBatchSampler,
    PrecomputedCodecDataset,
    stratified_limit_indices,
    stratified_split_indices,
)
from preprocessing.evaluation import (
    bootstrap_bd_rate,
    build_evaluation_dataset,
    calculate_bd_rate,
    calculate_bd_rate_details,
    dataset_sample_path,
)
from preprocessing.model import PaperPreprocessor, VideoTransformerPreprocessor
from preprocessing.standard_codec import ParallelStandardVideoCodec, StandardCodecProxy
from preprocessing.swin import VideoSwinLitePreprocessor


def test_preprocessor_shape_range_and_identity_initialization():
    model = PaperPreprocessor(temporal_frames=4)
    clip = torch.rand(2, 6, 3, 32, 40)
    output = model(clip)
    assert output.shape == clip.shape
    assert output.min() >= 0
    assert output.max() <= 1
    torch.testing.assert_close(output, clip)


def test_preprocessor_backpropagates():
    model = PaperPreprocessor(temporal_frames=3)
    clip = torch.rand(1, 4, 3, 16, 16)
    model(clip).mean().backward()
    assert model.to_rgb.weight.grad is not None


def test_vit_preprocessor_handles_non_patch_multiple_and_starts_as_identity():
    model = VideoTransformerPreprocessor(
        patch_size=4, embed_dim=16, depth=2, num_heads=4
    )
    clip = torch.rand(1, 3, 3, 17, 19)
    output = model(clip)
    assert output.shape == clip.shape
    torch.testing.assert_close(output, clip)


def test_vit_preprocessor_backpropagates_to_residual_head():
    model = VideoTransformerPreprocessor(
        patch_size=4, embed_dim=16, depth=2, num_heads=4
    )
    clip = torch.rand(1, 2, 3, 16, 16)
    model(clip).mean().backward()
    assert model.to_rgb.weight.grad is not None


def test_video_swin_lite_handles_padding_and_starts_as_identity():
    model = VideoSwinLitePreprocessor(
        patch_size=2,
        embed_dim=12,
        depth=2,
        num_heads=3,
        window_size=(2, 2, 2),
    )
    clip = torch.rand(1, 3, 3, 9, 11)
    output = model(clip)
    assert output.shape == clip.shape
    torch.testing.assert_close(output, clip)


def test_video_swin_lite_backpropagates_through_shifted_windows():
    model = VideoSwinLitePreprocessor(
        patch_size=2,
        embed_dim=12,
        depth=2,
        num_heads=3,
        window_size=(2, 2, 2),
    )
    torch.nn.init.normal_(model.to_rgb.weight, std=0.01)
    clip = 0.25 + 0.5 * torch.rand(1, 3, 3, 8, 8)
    model(clip, qp=30).mean().backward()
    assert model.to_rgb.weight.grad is not None
    assert torch.isfinite(model.to_rgb.weight.grad).all()
    shifted_qkv_gradient = model.blocks[1].attention.qkv.weight.grad
    assert shifted_qkv_gradient is not None
    assert torch.isfinite(shifted_qkv_gradient).all()


def test_video_swin_lite_qp_conditioning_changes_nonzero_residual():
    torch.manual_seed(7)
    model = VideoSwinLitePreprocessor(
        patch_size=2,
        embed_dim=12,
        depth=2,
        num_heads=3,
        window_size=(2, 2, 2),
        qp_conditioning=True,
        qp_embed_dim=8,
    )
    torch.nn.init.normal_(model.to_rgb.weight, std=0.01)
    assert model.qp_embedding is not None
    for parameter in model.qp_embedding.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    for film in model.qp_films:
        torch.nn.init.normal_(film.weight, std=0.2)
    clip = 0.25 + 0.5 * torch.rand(1, 3, 3, 8, 8)
    qp30 = model(clip, qp=30)
    qp45 = model(clip, qp=45)
    assert qp30.shape == clip.shape
    assert qp45.shape == clip.shape
    assert not torch.allclose(qp30, qp45)


def test_video_swin_lite_accepts_per_sample_qp():
    model = VideoSwinLitePreprocessor(
        patch_size=2,
        embed_dim=12,
        depth=1,
        num_heads=3,
        window_size=(2, 2, 2),
        qp_conditioning=True,
        qp_embed_dim=8,
    )
    clip = torch.rand(2, 2, 3, 8, 8)
    output = model(clip, qp=torch.tensor([30, 45]))
    assert output.shape == clip.shape
    torch.testing.assert_close(output, clip)


def test_standard_codec_proxy_preserves_clip_shape_and_has_input_gradient():
    proxy = StandardCodecProxy(
        hidden_channels=8,
        latent_channels=12,
        bottleneck_channels=16,
        blocks_per_stage=1,
        film_channels=8,
    )
    torch.nn.init.normal_(proxy.to_rgb.weight, std=0.01)
    clip = torch.rand(2, 3, 3, 17, 19, requires_grad=True)
    reconstruction, bpp = proxy(clip, qp=torch.tensor([30, 45]))
    assert reconstruction.shape == clip.shape
    assert bpp.shape == (2,)
    (reconstruction.mean() + bpp.mean()).backward()
    assert clip.grad is not None
    assert torch.isfinite(clip.grad).all()
    assert clip.grad.abs().sum() > 0
    assert proxy.config["architecture"] == "film_deeper3d_v1"


def test_film_deeper3d_rejects_legacy_shallow_checkpoint(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "proxy_config": {
                "hidden_channels": 8,
                "latent_channels": 12,
                "max_delta": 0.5,
            },
            "proxy": {},
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="retrain the proxy from scratch"):
        StandardCodecProxy.from_checkpoint(checkpoint)


class _FakeStandardCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qp = 35

    def set_qp(self, qp: int) -> None:
        self.qp = qp

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = clip.shape[0]
        return torch.full_like(clip, 0.375), torch.full((batch,), 1.25, device=clip.device)


class _FakeProxy(nn.Module):
    def forward(
        self, clip: torch.Tensor, qp: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del qp
        return clip * 2.0, clip.mean(dim=(1, 2, 3, 4))


def test_parallel_codec_uses_real_forward_values_and_proxy_backward():
    clip = torch.rand(1, 2, 3, 8, 8, requires_grad=True)
    codec = ParallelStandardVideoCodec(_FakeStandardCodec(), _FakeProxy()).train()
    reconstruction, bpp = codec(clip)
    torch.testing.assert_close(reconstruction, torch.full_like(clip, 0.375))
    torch.testing.assert_close(bpp, torch.tensor([1.25]))
    (reconstruction.mean() + bpp.mean()).backward()
    assert clip.grad is not None
    assert torch.count_nonzero(clip.grad) == clip.numel()


def test_parallel_codec_proxy_source_skips_real_forward_values():
    clip = torch.rand(1, 2, 3, 8, 8, requires_grad=True)
    codec = ParallelStandardVideoCodec(_FakeStandardCodec(), _FakeProxy()).train()
    reconstruction, bpp = codec(clip, codec_source="proxy")
    torch.testing.assert_close(reconstruction, clip * 2.0)
    torch.testing.assert_close(bpp, clip.mean(dim=(1, 2, 3, 4)))
    (reconstruction.mean() + bpp.mean()).backward()
    assert clip.grad is not None


def test_frozen_film_deeper3d_proxy_backpropagates_to_preprocessor_input():
    proxy = StandardCodecProxy(
        hidden_channels=8,
        latent_channels=12,
        bottleneck_channels=16,
        blocks_per_stage=1,
        film_channels=8,
    )
    torch.nn.init.normal_(proxy.to_rgb.weight, std=0.01)
    codec = ParallelStandardVideoCodec(_FakeStandardCodec(), proxy).train()
    clip = torch.rand(2, 3, 3, 16, 16, requires_grad=True)
    reconstruction, bpp = codec(clip)
    torch.testing.assert_close(reconstruction, torch.full_like(clip, 0.375))
    torch.testing.assert_close(bpp, torch.full((2,), 1.25))
    (reconstruction.mean() + bpp.mean()).backward()
    assert clip.grad is not None
    assert torch.isfinite(clip.grad).all()
    assert clip.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in proxy.parameters())


def test_stratified_split_has_no_overlap_and_preserves_classes():
    samples = [(None, label) for label in range(3) for _ in range(5)]
    train, val = stratified_split_indices(samples, val_ratio=0.2, seed=42)
    assert not set(train) & set(val)
    assert len(train) == 12
    assert len(val) == 3
    assert {samples[index][1] for index in train} == {0, 1, 2}
    assert {samples[index][1] for index in val} == {0, 1, 2}


def test_stratified_limit_preserves_every_class_when_limit_allows():
    samples = [(None, label) for label in range(4) for _ in range(6)]
    limited = stratified_limit_indices(samples, range(len(samples)), limit=8, seed=42)
    labels = [samples[index][1] for index in limited]
    assert len(limited) == 8
    assert set(labels) == {0, 1, 2, 3}
    assert all(labels.count(label) == 2 for label in range(4))


def test_evaluation_recreates_checkpoint_validation_split_and_limit(tmp_path):
    categories = ["class zero", "class one"]
    for class_name in ("class_zero", "class_one"):
        directory = tmp_path / class_name
        directory.mkdir()
        for index in range(5):
            (directory / f"{index}.mp4").touch()

    full = build_evaluation_dataset(
        categories=categories,
        data_root=tmp_path,
        saved_args={"val_ratio": 0.2, "seed": 7},
    )
    limited = build_evaluation_dataset(
        categories=categories,
        data_root=tmp_path,
        limit=2,
        saved_args={"val_ratio": 0.2, "seed": 7},
    )
    limited_again = build_evaluation_dataset(
        categories=categories,
        data_root=tmp_path,
        limit=2,
        saved_args={"val_ratio": 0.2, "seed": 7},
    )

    assert len(full) == 2
    assert len(limited) == 2
    assert [dataset_sample_path(limited, index) for index in range(2)] == [
        dataset_sample_path(limited_again, index) for index in range(2)
    ]


def test_bd_rate_reports_known_twenty_percent_saving():
    rows = []
    for method, scale in (("anchor", 1.0), ("preprocessed", 0.8)):
        for quality, rate in zip((30.0, 40.0, 50.0, 60.0), (0.1, 0.2, 0.4, 0.8)):
            rows.append(
                {
                    "method": method,
                    "bpp": rate * scale,
                    "quality": quality,
                }
            )

    assert calculate_bd_rate(rows, "quality") == pytest.approx(-20.0, abs=1e-6)


def test_bd_rate_details_report_pchip_overlap_and_effective_points():
    rows = []
    for method, scale in (("anchor", 1.0), ("preprocessed", 0.9)):
        for quality, rate in zip((30.0, 40.0, 50.0), (0.1, 0.2, 0.4)):
            rows.append(
                {"method": method, "bpp": rate * scale, "quality": quality}
            )

    details = calculate_bd_rate_details(rows, "quality")
    assert details["interpolation"] == "pchip"
    assert details["anchor_points"] == 3
    assert details["preprocessed_points"] == 3
    assert details["quality_min"] == pytest.approx(30.0)
    assert details["quality_max"] == pytest.approx(50.0)
    assert details["overlap_span"] == pytest.approx(20.0)
    assert details["bd_rate_percent"] == pytest.approx(-10.0, abs=1e-6)


def test_bd_rate_is_undefined_for_flat_task_accuracy():
    rows = [
        {"method": method, "bpp": rate, "top1": 50.0}
        for method in ("anchor", "preprocessed")
        for rate in (0.1, 0.2, 0.3, 0.4)
    ]
    assert calculate_bd_rate(rows, "top1") is None


def test_paired_bootstrap_preserves_known_rate_scaling():
    rows = []
    qualities = (30.0, 40.0, 50.0, 60.0)
    rates = (0.1, 0.2, 0.4, 0.8)
    for sample_index in range(6):
        for method, scale in (("anchor", 1.0), ("preprocessed", 0.8)):
            for qp, (quality, rate) in enumerate(zip(qualities, rates, strict=True)):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "method": method,
                        "qp": qp,
                        "bpp": rate * scale,
                        "mse": 10.0 ** (-quality / 10.0),
                        "top1": int(qp >= 2),
                    }
                )

    uncertainty = bootstrap_bd_rate(
        rows, "psnr_db", samples=100, confidence_level=0.95, seed=7
    )
    assert uncertainty["samples_valid"] == 100
    assert uncertainty["point_estimate_percent"] == pytest.approx(-20.0, abs=1e-6)
    assert uncertainty["median_percent"] == pytest.approx(-20.0, abs=1e-6)
    assert uncertainty["lower_percent"] == pytest.approx(-20.0, abs=1e-6)
    assert uncertainty["upper_percent"] == pytest.approx(-20.0, abs=1e-6)


def test_paired_bootstrap_rejects_incomplete_operating_points():
    rows = [
        {
            "sample_index": 0,
            "method": "anchor",
            "qp": 30,
            "bpp": 0.2,
            "mse": 0.01,
            "top1": 1,
        },
        {
            "sample_index": 1,
            "method": "preprocessed",
            "qp": 30,
            "bpp": 0.18,
            "mse": 0.01,
            "top1": 1,
        },
    ]
    with pytest.raises(ValueError, match="every video at every method/QP"):
        bootstrap_bd_rate(rows, "psnr_db", samples=10)


def test_precomputed_codec_dataset_reuses_one_uint8_source_for_all_qps(tmp_path):
    qps = [30, 35, 40, 45]
    manifest = {
        "codec": {"qps": qps},
        "splits": {"train": [{"id": "00000000", "source": "clip.mp4", "label": 3}]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    clip_path = tmp_path / "train" / "clips" / "00000000.pt"
    clip_path.parent.mkdir(parents=True)
    source = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
    torch.save({"clip": source}, clip_path)
    for qp in qps:
        target_path = tmp_path / "train" / "recon" / f"qp_{qp}" / "00000000.pt"
        target_path.parent.mkdir(parents=True)
        torch.save({"reconstruction": source, "bpp": torch.tensor(qp / 10)}, target_path)

    dataset = PrecomputedCodecDataset(tmp_path, "train", qps)
    assert len(dataset) == 4
    for index, qp in enumerate(qps):
        clip, reconstruction, bpp, returned_qp = dataset[index]
        assert returned_qp == qp
        torch.testing.assert_close(clip, source.float() / 255.0)
        torch.testing.assert_close(reconstruction, clip)
        torch.testing.assert_close(bpp, torch.tensor(qp / 10))


def test_mixed_qp_sampler_balances_every_batch(tmp_path):
    qps = [30, 35, 40, 45]
    samples = [
        {"id": f"{index:08d}", "source": f"{index}.mp4", "label": 0}
        for index in range(4)
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"codec": {"qps": qps}, "splits": {"train": samples}}),
        encoding="utf-8",
    )
    dataset = PrecomputedCodecDataset(tmp_path, "train", qps)
    sampler = MixedQPBatchSampler(dataset, batch_size=8, seed=7)
    batches = list(sampler)
    assert len(batches) == 2
    assert sorted(index % len(qps) for index in batches[0]) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert sorted(index for batch in batches for index in batch) == list(range(len(dataset)))
