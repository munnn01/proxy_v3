from types import SimpleNamespace
import json
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import train
from preprocessing.feature_distillation import (
    feature_configuration, feature_distillation, validate_feature_resume,
)


def test_legacy_cosine_is_unchanged():
    student, teacher = torch.randn(2, 5, 3, 4, 4), torch.randn(2, 5, 3, 4, 4)
    args = SimpleNamespace(feature_layer="layer4")
    layers, weights, mode = feature_configuration(args)
    loss, _ = feature_distillation({"layer4": student}, {"layer4": teacher}, layers, weights, mode)
    expected = 1 - (F.normalize(student, dim=1) * F.normalize(teacher, dim=1)).sum(dim=1).mean()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)


def test_relative_mse_detects_magnitude_error_cosine_cannot_detect():
    teacher = torch.rand(2, 8, 2, 4, 4) + 0.5
    student = 2 * teacher
    relative, _ = feature_distillation({"f": student}, {"f": teacher}, ["f"], [1], "relative_mse")
    cosine, _ = feature_distillation({"f": student}, {"f": teacher}, ["f"], [1], "cosine")
    assert relative.item() == pytest.approx(1)
    assert cosine.item() == pytest.approx(0, abs=1e-7)


def test_multilayer_relative_mse_is_per_clip_and_teacher_detached():
    teacher = {"a": torch.full((2, 2, 1, 4, 4), 2., requires_grad=True),
               "b": torch.full((2, 8, 2, 2, 2), 10., requires_grad=True)}
    student = {"a": torch.full_like(teacher["a"], 4., requires_grad=True),
               "b": torch.full_like(teacher["b"], 15., requires_grad=True)}
    loss, terms = feature_distillation(student, teacher, ["a", "b"], [0.3, 0.7], "relative_mse")
    assert loss.item() == pytest.approx(0.3 * 1 + 0.7 * .25)
    loss.backward()
    assert all(t.grad is None for t in teacher.values())
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in student.values())
    assert terms["a"].item() == pytest.approx(1.)
    scaled, _ = feature_distillation({k: v.detach() * 100 for k, v in student.items()},
                                     {k: v.detach() * 100 for k, v in teacher.items()},
                                     ["a", "b"], [0.3, 0.7], "relative_mse")
    torch.testing.assert_close(scaled, loss.detach())


def test_zero_energy_and_mixed_clip_scales_remain_finite():
    reference = torch.tensor([[0., 0.], [10., 10.]])
    proposed = torch.tensor([[1e-4, 1e-4], [20., 20.]], requires_grad=True)
    loss, _ = feature_distillation({"f": proposed}, {"f": reference}, ["f"], [1], "relative_mse")
    assert loss.item() == pytest.approx((.01 + 1) / 2)
    loss.backward()
    assert torch.isfinite(proposed.grad).all()


@pytest.mark.parametrize("layers,weights", [([], []), (["a", "a"], [1, 1]),
                         (["a", "b"], [1]), (["a"], [-1]), (["a"], [0]), (["a"], [float("nan")])])
def test_invalid_layer_configuration_rejected(layers, weights):
    with pytest.raises(ValueError):
        feature_configuration(SimpleNamespace(feature_layers=layers, feature_layer_weights=weights))


def test_resume_accepts_legacy_but_rejects_feature_objective_changes():
    saved = {"feature_weight": .05, "feature_layer": "layer4"}
    current = SimpleNamespace(**saved, feature_layers=None, feature_layer_weights=None, feature_loss="cosine")
    validate_feature_resume(saved, current)
    current.feature_loss = "relative_mse"
    with pytest.raises(ValueError, match="init-checkpoint"):
        validate_feature_resume(saved, current)


def test_forward_loss_backpropagates_both_layers_to_preprocessor_only():
    class Analyzer(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(2.), requires_grad=False)

        def forward_with_features(self, clips, layers):
            f = clips.permute(0, 2, 1, 3, 4) * self.gain
            logits = f.mean((1, 2, 3, 4)).unsqueeze(1).expand(-1, 5)
            return logits, {name: f * (i + 1) for i, name in enumerate(layers)}

    class Preprocessor(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(.8))

        def forward(self, x, qp):
            return x * self.gain

    class Codec(nn.Module):
        def forward(self, x, codec_source):
            return x, x.mean((1, 2, 3, 4))

    args = SimpleNamespace(feature_weight=.05, feature_layers=["a", "b"],
                           feature_layer_weights=[.3, .7], feature_loss="relative_mse",
                           alpha=0., qp_to_rate_lambda={30: 0.}, ce_weight=0., kd_weight=0.)
    analyzer, preprocessor = Analyzer(), Preprocessor()
    clips = torch.rand(2, 2, 3, 4, 4)
    _, clean = analyzer.forward_with_features(clips, ["a", "b"])
    losses = train.forward_losses(clips, torch.zeros(2, dtype=torch.long), preprocessor,
                                  Codec(), analyzer, args, False, 30, clean_features=clean)
    assert losses["feature_loss_a"].item() == pytest.approx(.04)
    assert losses["feature_loss_b"].item() == pytest.approx(.04)
    assert losses["total"].item() == pytest.approx(.05 * .04)
    losses["total"].backward()
    assert preprocessor.gain.grad < 0
    assert analyzer.gain.grad is None


def test_initial_validation_only_measures_loaded_weights_without_training(tmp_path, monkeypatch):
    original = nn.Linear(1, 1)
    checkpoint, proxy, output = tmp_path / "initial.pt", tmp_path / "proxy.pt", tmp_path / "run"
    torch.save({"preprocessor": original.state_dict()}, checkpoint)
    torch.save({}, proxy)
    before = checkpoint.read_bytes()
    monkeypatch.setattr(sys, "argv", ["train.py", "--data-root", "unused", "--device", "cpu",
        "--no-amp", "--init-checkpoint", str(checkpoint), "--proxy-checkpoint", str(proxy),
        "--initial-validation-only", "--output-dir", str(output)])
    monkeypatch.setattr(train, "require_ffmpeg", lambda *a: None)
    analyzer = SimpleNamespace(transform=None, categories=[])
    analyzer.to = lambda device: analyzer
    monkeypatch.setattr(train, "FrozenVideoAnalyzer", lambda *a: analyzer)
    monkeypatch.setattr(train, "make_loaders", lambda *a: ("train", "val"))
    monkeypatch.setattr(train, "build_preprocessor", lambda *a, **k: nn.Linear(1, 1))
    monkeypatch.setattr(train.StandardCodecProxy, "from_checkpoint", lambda *a: nn.Identity())
    monkeypatch.setattr(train, "StandardVideoCodec", lambda *a, **k: None)
    monkeypatch.setattr(train, "ParallelStandardVideoCodec", lambda *a: nn.Identity())
    anchor = {f"qp{q}_{key}": value for q, rate, acc in zip([30, 35, 40, 45], [.4, .3, .2, .1], [.7, .6, .5, .4])
              for key, value in (("bpp", rate), ("top1", acc))}
    monkeypatch.setattr(train, "load_or_evaluate_anchor_validation", lambda *a: anchor)
    calls = []

    def epoch(loader, model, *a, **k):
        assert loader == "val" and k.get("optimizer") is None
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[name])
        calls.append(loader)
        return dict(anchor)

    monkeypatch.setattr(train, "run_epoch", epoch)
    train.main()
    report = json.loads((output / "initial_validation.json").read_text())
    assert calls == ["val"]
    assert report["optimizer_updates"] == 0
    assert report["val_metrics"]["task_bd_rate_percent"] == pytest.approx(0., abs=1e-8)
    assert checkpoint.read_bytes() == before
    assert not list(output.glob("*.pt"))


def test_v8_and_legacy_control_differ_only_in_feature_objective(tmp_path, monkeypatch):
    from pathlib import Path
    from kaggle_cells.v7_rate_recovery import RecoveryRun, training_arguments
    from kaggle_cells.v8_feature_ablation import loss_overrides
    project = Path(train.__file__).parent
    monkeypatch.chdir(project)
    saved = vars(train.build_parser().parse_args([
        "@presets/v6_accuracy_rate.args", "--proxy-checkpoint", "old.pt",
        "--resume", "old/last.pt", "--epochs", "15",
    ]))
    run = RecoveryRun(project, tmp_path, tmp_path, tmp_path / "manifest.json",
                      tmp_path / "v6_best.pt", tmp_path / "proxy.pt", tmp_path / "train",
                      tmp_path / "controller", tmp_path / "proxy", tmp_path / "swin", saved, {})
    old = vars(train.build_parser().parse_args(training_arguments(run, run.starting_proxy, loss_overrides("legacy"))))
    new = vars(train.build_parser().parse_args(training_arguments(run, run.starting_proxy, loss_overrides())))
    difference = {key for key in old if old[key] != new[key]}
    assert difference == {"feature_loss", "feature_layers", "feature_layer_weights"}
    assert old["resume"] is None and new["resume"] is None
    assert new["init_checkpoint"] == str(run.checkpoint)
    assert new["epochs"] == 2 and new["lr"] == 1e-5 and new["target_bpp_ratio"] == .95
