import argparse
import hashlib

import pytest
import torch

from kaggle_cells.v7_rate_recovery import (
    EXPECTED_START_BD_RATE,
    balanced_controller,
    find_recorded_proxy,
    find_start_checkpoint,
    saved_training_cli,
)


def test_balanced_controller_is_deterministic_and_uses_each_class():
    candidates = [f"class_{label}/v{index}.mp4" for label in range(4) for index in range(5)]
    first = balanced_controller(candidates, 12, 345)
    assert first == balanced_controller(candidates, 12, 345)
    assert len(first) == len(set(first)) == 12
    assert {item.split("/")[0] for item in first} == {f"class_{i}" for i in range(4)}


def test_balanced_controller_rejects_insufficient_pool():
    with pytest.raises(ValueError, match="Not enough"):
        balanced_controller(["a/1.mp4"], 2, 1)


def test_start_checkpoint_uses_actual_metric(tmp_path):
    input_root = tmp_path / "input"
    folder = input_root / "dataset" / "v6_calibration_20260905_161558_046557" / "swin_ratio_090"
    folder.mkdir(parents=True)
    path = folder / "best_task_bd_rate.pt"
    torch.save({"preprocessor": {}, "epoch": 9, "args": {},
                "val_metrics": {"task_bd_rate_percent": EXPECTED_START_BD_RATE}}, path)
    selected, payload = find_start_checkpoint(input_root, tmp_path / "working", torch)
    assert selected == path
    assert payload["epoch"] == 9


def test_recorded_proxy_must_match_hash(tmp_path):
    proxy_dir = tmp_path / "proxy_calibrated"
    proxy_dir.mkdir()
    proxy = proxy_dir / "best.pt"
    torch.save({"proxy": {}, "proxy_config": {"architecture": "test"}}, proxy)
    digest = hashlib.sha256(proxy.read_bytes()).hexdigest()
    selected, payload = find_recorded_proxy({"proxy_sha256": digest}, [tmp_path])
    assert selected == proxy
    assert "proxy_config" in payload
    with pytest.raises(FileNotFoundError, match="complete saved run"):
        find_recorded_proxy({"proxy_sha256": "0" * 64}, [tmp_path])


def test_training_cli_resets_resume_and_preserves_explicit_false():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qps", nargs="+", type=int)
    cli = saved_training_cli(
        {"resume": "old.pt", "feature": True, "qps": [30]},
        {"resume": None, "init_checkpoint": "best.pt", "feature": False,
         "qps": [30, 35, 40, 45]},
        parser,
    )
    assert "--resume" not in cli
    assert cli == ["--init-checkpoint", "best.pt", "--no-feature",
                   "--qps", "30", "35", "40", "45"]
