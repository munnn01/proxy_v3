"""Proxy replacement must not turn a continuation into a fresh Swin run."""

from copy import deepcopy
import argparse
import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train


def test_ordinary_resume_still_rejects_proxy_changes():
    checkpoint = {"proxy_sha256": "old", "args": {"proxy_checkpoint": "old.pt"}}
    assert not train.validate_resume_proxy(checkpoint, "old.pt", "old", refresh=False)
    with pytest.raises(ValueError, match="frozen proxy weights"):
        train.validate_resume_proxy(checkpoint, "new.pt", "new", refresh=False)
    with pytest.raises(ValueError, match="saved proxy path"):
        train.validate_resume_proxy(checkpoint, "new.pt", "old", refresh=False)
    assert train.validate_resume_proxy(checkpoint, "new.pt", "new", refresh=True)
    with pytest.raises(ValueError, match="newly calibrated"):
        train.validate_resume_proxy(checkpoint, "old.pt", "old", refresh=True)
    with pytest.raises(ValueError, match="proxy_sha256"):
        train.validate_resume_proxy({}, "new.pt", "new", refresh=True)


def test_kaggle_recipe_restores_saved_options_without_old_initialization():
    recipe = Path(train.__file__).parent / "kaggle_cells/v6_refresh_resume.py"
    tree = ast.parse(recipe.read_text(encoding="utf-8"))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "saved_training_cli")
    namespace = {"argparse": argparse, "build_parser": train.build_parser}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(recipe), "exec"), namespace)
    saved = vars(train.build_parser().parse_args([
        "--data-root", "data", "--proxy-checkpoint", "old.pt",
        "--init-checkpoint", "warm.pt", "--swin-qp-conditioning", "--no-amp",
        "--target-bpp-ratio", "0.90", "--codec-qps", "30", "40", "45",
        "--swin-smoothing-max-strength", "0.4", "--lr", "0.000025",
    ]))
    saved["qp_to_anchor_bpp"] = {30: 0.3}  # Derived fields are not CLI arguments.
    overrides = {"init_checkpoint": None, "resume": "last.pt",
                 "refresh_proxy_on_resume": True, "proxy_checkpoint": "new.pt", "epochs": 15}
    cli = namespace["saved_training_cli"](saved, overrides)
    restored = vars(train.build_parser().parse_args(cli))
    for key, value in {**saved, **overrides}.items():
        if key in restored:
            assert restored[key] == value, key
    assert "--no-amp" in cli
    assert "--init-checkpoint" not in cli


@pytest.mark.parametrize("options", [[], ["--resume", "checkpoints/last.pt"]])
def test_refresh_requires_resume_with_direct_rate_control(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["train.py", "--data-root", "data",
                                     "--proxy-checkpoint", "new.pt",
                                     "--refresh-proxy-on-resume", *options])
    with pytest.raises(ValueError, match="requires --resume and --rate-dual-control"):
        train.main()


@pytest.mark.parametrize("invalid", [0.0, -0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["bpp", "proxy_bpp"])
def test_refresh_rejects_invalid_rate_without_resetting_guard(invalid, field):
    state = {"proxy_guard_streak": 2}
    metrics = {"qp30_bpp": 0.1, "qp30_proxy_bpp": 0.1}
    metrics[f"qp30_{field}"] = invalid
    with pytest.raises(ValueError, match="BPP"):
        train.validate_refreshed_proxy(
            state, {"qp30_bpp": 0.1}, metrics, [30],
            maximum_allowed_ratio=0.91, max_underestimate_percent=10, patience=1,
        )
    assert state == {"proxy_guard_streak": 2}


def test_refresh_preserves_real_rate_history_and_resets_only_proxy_guard():
    state = {
        "proxy_guard_streak": 2, "proxy_guard_bad_qps": [30],
        "log_dual_weight_by_qp": {30: 1.4}, "ema_ratio_by_qp": {30: 1.03},
        "last_ratio_by_qp": {30: 1.05}, "last_proxy_drift_percent_by_qp": {30: -13.0},
    }
    original = deepcopy(state)
    updated, drifts = train.validate_refreshed_proxy(
        state, {"qp30_bpp": 0.1}, {"qp30_bpp": 0.11, "qp30_proxy_bpp": 0.105}, [30],
        maximum_allowed_ratio=0.91, max_underestimate_percent=10, patience=1,
    )
    assert state == original
    assert updated["proxy_guard_streak"] == 0
    assert updated["proxy_guard_bad_qps"] == []
    assert drifts[30] == pytest.approx(-4.5454545)
    for key in ("log_dual_weight_by_qp", "ema_ratio_by_qp", "last_ratio_by_qp"):
        assert updated[key] == original[key]


@pytest.mark.parametrize("accepted", [True, False])
def test_main_refresh_checks_before_epoch8_and_restores_adam_state(tmp_path, monkeypatch, accepted):
    """Exercise actual main/resume/checkpoint logic with tiny models and codec metrics."""
    output = tmp_path / "swin"
    output.mkdir()
    new_proxy = tmp_path / "new_proxy.pt"
    torch.save({}, new_proxy)
    resume = output / "last.pt"
    command = [
        "train.py", f"@{Path(train.__file__).parent / 'presets/v6_accuracy_rate.args'}",
        "--data-root", "unused", "--device", "cpu", "--no-amp",
        "--target-bpp-ratio", "0.9", "--epochs", "8",
        "--resume", str(resume), "--proxy-checkpoint", str(new_proxy),
        "--output-dir", str(output), "--refresh-proxy-on-resume",
    ]
    monkeypatch.setattr(sys, "argv", command)
    args = train.parse_args()
    old_args = vars(args).copy()
    old_args["proxy_checkpoint"] = "old_proxy.pt"
    old_args["refresh_proxy_on_resume"] = False
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    for _ in range(7):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).square().sum().backward()
        optimizer.step()
        scheduler.step(1.0)
    initial_weights = {qp: 0.5 * bpp for qp, bpp in zip(args.codec_qps, [0.3, 0.2, 0.1, 0.05])}
    state = train.initialize_rate_dual_state(
        initial_weights, start_epoch=1, target_ratio=0.9, codec="h264",
        train_codec_source="real", kappa=3.0, ema_beta=0.8,
        minimum_weight=0.0001, maximum_weight=10.0,
    )
    train.update_rate_dual_state(
        state, {qp: 1.05 for qp in args.codec_qps}, target_ratio=0.9,
        kappa=3.0, ema_beta=0.8, minimum_weight=0.0001, maximum_weight=10.0,
    )
    state["proxy_guard_streak"] = 2
    state["proxy_guard_bad_qps"] = [30]
    saved = {
        "epoch": 7, "args": old_args, "proxy_sha256": "oldhash",
        "preprocessor": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": {}, "rate_dual_state": state,
        "proxy_refresh_history": [{"after_epoch": 3}],
    }
    torch.save(saved, resume)
    original_bytes = resume.read_bytes()
    anchor = {f"qp{qp}_bpp": 2 * weight for qp, weight in initial_weights.items()}
    anchor.update({f"qp{qp}_top1": 0.5 for qp in args.codec_qps})
    monkeypatch.setattr(train, "require_ffmpeg", lambda *a: None)
    analyzer = SimpleNamespace(transform=None, categories=[])
    analyzer.to = lambda device: analyzer
    monkeypatch.setattr(train, "FrozenVideoAnalyzer", lambda *a: analyzer)
    monkeypatch.setattr(train, "analyzer_view_box", lambda *a: None)
    monkeypatch.setattr(train, "make_loaders", lambda *a: ("train_loader", "val_loader"))
    monkeypatch.setattr(train, "build_preprocessor", lambda *a, **kw: nn.Linear(1, 1))
    monkeypatch.setattr(train.StandardCodecProxy, "from_checkpoint", lambda *a: nn.Identity())
    monkeypatch.setattr(train, "StandardVideoCodec", lambda *a, **kw: None)
    monkeypatch.setattr(train, "ParallelStandardVideoCodec", lambda *a: nn.Identity())
    monkeypatch.setattr(train, "load_or_evaluate_anchor_validation", lambda *a: anchor)
    monkeypatch.setattr(train, "validation_task_bd_rate", lambda *a: -12.0)
    calls = []

    def epoch(loader, preprocessor, codec, analyzer, current_args, device, **kwargs):
        calls.append(loader)
        opt = kwargs.get("optimizer")
        if len(calls) == 1:
            assert opt is None and loader == "val_loader"
            for key, value in saved["preprocessor"].items():
                torch.testing.assert_close(preprocessor.state_dict()[key], value)
        if opt is not None:
            assert calls == ["val_loader", "train_loader"]
            assert opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
            for param, old_param in zip(preprocessor.parameters(), model.parameters()):
                for name in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(opt.state[param][name], optimizer.state[old_param][name])
            assert current_args.qp_to_rate_dual_weight == train.rate_dual_weights(state, args.codec_qps)
            opt.zero_grad()
            preprocessor(torch.ones(1, 1)).square().sum().backward()
            opt.step()
        ratio = 1.05 if len(calls) == 1 else 0.88
        proxy_scale = 0.98 if accepted else 0.80
        metrics = {"loss": 1.0, "monitor_loss": 1.0, "ce_loss": 0.5, "top1": 0.5}
        for qp in args.codec_qps:
            metrics.update({
                f"qp{qp}_bpp": anchor[f"qp{qp}_bpp"] * ratio,
                f"qp{qp}_proxy_bpp": anchor[f"qp{qp}_bpp"] * ratio * proxy_scale,
                f"qp{qp}_proxy_rate_ratio": ratio * proxy_scale,
                f"qp{qp}_top1": 0.5,
            })
        return metrics

    monkeypatch.setattr(train, "run_epoch", epoch)
    if not accepted:
        with pytest.raises(RuntimeError, match="refreshed proxy failed"):
            train.main()
        assert calls == ["val_loader"]
        assert resume.read_bytes() == original_bytes
        assert not (output / "best.pt").exists()
        return
    train.main()
    result = torch.load(resume, weights_only=False)
    assert calls == ["val_loader", "train_loader", "val_loader"]
    assert result["epoch"] == 8
    assert result["scheduler"]["last_epoch"] == saved["scheduler"]["last_epoch"] + 1
    assert all(item["step"] == 8 for item in result["optimizer"]["state"].values())
    assert result["proxy_sha256"] == hashlib.sha256(new_proxy.read_bytes()).hexdigest()
    assert result["proxy_refresh_history"][0] == {"after_epoch": 3}
    event = result["proxy_refresh_history"][-1]
    assert event["after_epoch"] == 7 and event["previous_proxy_sha256"] == "oldhash"
    assert event["previous_guard_streak"] == 2
    assert result["rate_dual_state"]["proxy_guard_streak"] == 0
    assert result["run_status"] == "has_feasible_checkpoint"
