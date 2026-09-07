import json

import pytest

from kaggle_cells.evaluate_best_task_from_inputs import (
    QPS, print_report, select_checkpoint, select_manifest, validation_sources,
)


def test_selects_actual_checkpoint_metric_not_historical_best(tmp_path):
    old, best = tmp_path / "old.pt", tmp_path / "best.pt"
    old.write_bytes(b"old")
    best.write_bytes(b"best")
    payloads = {
        old: {"preprocessor": {}, "epoch": 15, "best_task_bd_rate": -6.687,
              "val_metrics": {"task_bd_rate_percent": -2.733}},
        best: {"preprocessor": {}, "epoch": 12,
               "val_metrics": {"task_bd_rate_percent": -6.687123}},
    }
    assert select_checkpoint([old, best], -6.687, payloads.__getitem__)[0] == best
    with pytest.raises(RuntimeError, match="latest saved Output"):
        select_checkpoint([old], -6.687, payloads.__getitem__)


def test_ambiguous_checkpoint_requires_explicit_path(tmp_path):
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for path in paths:
        path.write_bytes(path.name.encode())
    payload = {"preprocessor": {}, "epoch": 12,
               "val_metrics": {"task_bd_rate_percent": -6.6871}}
    with pytest.raises(RuntimeError, match="--checkpoint"):
        select_checkpoint(paths, -6.687, lambda _: payload)


def manifest():
    return {"groups": {"train": ["class/train.mp4"],
                       "controller": ["class/a.mp4"],
                       "validation_full": ["class/a.mp4", "class/b.mp4"]}}


def test_exact_membership_missing_video_and_overlap(tmp_path):
    (tmp_path / "class").mkdir()
    for name in ("a", "b", "extra"):
        (tmp_path / "class" / f"{name}.mp4").touch()
    data = manifest()
    sources = validation_sources(tmp_path, data)
    assert [relative.as_posix() for relative, _ in sources] == data["groups"]["validation_full"]
    data["groups"]["train"].append("class/a.mp4")
    with pytest.raises(ValueError, match="overlaps"):
        validation_sources(tmp_path, data)
    data = manifest()
    data["groups"]["validation_full"].append("class/missing.mp4")
    with pytest.raises(FileNotFoundError):
        validation_sources(tmp_path, data)


def test_conflicting_manifests_rejected(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps(manifest()))
    data = manifest()
    data["groups"]["validation_full"].append("class/other.mp4")
    second.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="--manifest"):
        select_manifest([first, second])


def test_report_separates_bd_rate_from_accuracy_and_interval(tmp_path, capsys):
    rows = [{"codec": "h264", "method": method, "qp": qp,
             "bpp": 1 if method == "anchor" else 0.85,
             "top1_percent": 60 if method == "anchor" else 59}
            for qp in QPS for method in ("anchor", "preprocessed")]
    (tmp_path / "metrics.json").write_text(json.dumps(rows))
    (tmp_path / "bd_rate.json").write_text(json.dumps({"h264": {
        "task_bd_rate_percent": -11, "task_bootstrap": {
            "lower_percent": -15, "upper_percent": -7, "confidence_level": .95}}}))
    print_report(tmp_path)
    output = capsys.readouterr().out
    assert "Point estimate below -10%: True" in output
    assert "Interval upper bound below -10%: False" in output
    assert "Top1 >= H264 at every QP: False" in output
    assert "BPP <= H264 at every QP: True" in output
    (tmp_path / "metrics.json").write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError, match="seven"):
        print_report(tmp_path)
