"""Kaggle rate-recovery fine-tuning from the validated V6 best-task checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys

RUN_NAME = "v6_calibration_20260905_161558_046557"
EXPECTED_START_BD_RATE = -6.687
CLEAN_DEFAULT = Path(
    "/kaggle/input/datasets/qktttttttttt/kineticscleaned/cleaned_final/"
    "kinetics400_5per/kinetics400_5per/train"
)
TRAIN_QPS = [30, 35, 40, 45]
CONTROLLER_SIZE = 800
TARGET_BPP_RATIO = 0.95
FINETUNE_EPOCHS = 5
FINETUNE_LR = 1e-5


@dataclass(frozen=True)
class RecoveryRun:
    project: Path
    root: Path
    clean: Path
    manifest: Path
    checkpoint: Path
    starting_proxy: Path
    train_dir: Path
    controller_dir: Path
    proxy_dir: Path
    swin_dir: Path
    saved_args: dict
    proxy_config: dict


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_source(clean: Path, member: str) -> tuple[PurePosixPath, Path]:
    relative = PurePosixPath(member)
    if (relative.is_absolute() or len(relative.parts) < 2 or ".." in relative.parts
            or "\\" in member or ":" in member):
        raise ValueError(f"Invalid manifest path: {member}")
    root = clean.resolve(strict=True)
    source = root.joinpath(*relative.parts).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError(f"Missing or unsafe clean video: {member}")
    return relative, source


def balanced_controller(candidates: list[str], size: int, seed: int) -> list[str]:
    """Choose a deterministic class-balanced subset from unused train-pool clips."""
    if size < 1:
        raise ValueError("controller size must be positive")
    by_class: dict[str, list[str]] = {}
    for member in sorted(set(candidates)):
        relative = PurePosixPath(member)
        if len(relative.parts) < 2:
            raise ValueError(f"Invalid class-folder path: {member}")
        by_class.setdefault(relative.parts[0], []).append(member)
    if sum(map(len, by_class.values())) < size:
        raise ValueError("Not enough unused train-pool clips for the recovery controller")
    generator = random.Random(seed)
    labels = sorted(by_class)
    generator.shuffle(labels)
    for members in by_class.values():
        generator.shuffle(members)
    chosen = []
    while len(chosen) < size:
        for label in labels:
            if by_class[label]:
                chosen.append(by_class[label].pop())
                if len(chosen) == size:
                    break
    generator.shuffle(chosen)
    return chosen


def link_members(clean: Path, members: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for member in members:
        relative, source = _safe_source(clean, member)
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)


def find_start_checkpoint(input_root: Path, working: Path, torch_module) -> tuple[Path, dict]:
    candidates = list(input_root.glob(f"**/{RUN_NAME}/swin_ratio_090/best_task_bd_rate.pt"))
    local = working / RUN_NAME / "swin_ratio_090" / "best_task_bd_rate.pt"
    if local.is_file():
        candidates.append(local)
    matches = []
    observations = []
    for path in sorted(set(candidates)):
        payload = torch_module.load(path, map_location="cpu", weights_only=False)
        score = payload.get("val_metrics", {}).get("task_bd_rate_percent")
        observations.append(f"{path}: epoch={payload.get('epoch')}, actual BD-rate={score}")
        if ("preprocessor" in payload and isinstance(score, (int, float))
                and math.isfinite(score) and abs(score - EXPECTED_START_BD_RATE) < 0.0005):
            matches.append((path, payload))
        else:
            del payload
    if not matches:
        raise RuntimeError(
            "Attach the latest Output containing the actual -6.687% best-task checkpoint.\n"
            + "\n".join(observations)
        )
    hashes = {file_sha256(path) for path, _ in matches}
    if len(hashes) != 1:
        raise RuntimeError("Different -6.687% checkpoints found; remove the ambiguous Input.")
    return matches[0]


def find_manifest(input_root: Path, working: Path) -> tuple[Path, dict]:
    paths = list(input_root.glob("**/v5_fixed_split/split_manifest.json"))
    local = working / "v5_fixed_split" / "split_manifest.json"
    if local.is_file():
        paths.append(local)
    if not paths:
        raise FileNotFoundError("Attach checking with v5_fixed_split/split_manifest.json")
    payloads = [(path, json.loads(path.read_text(encoding="utf-8")))
                for path in sorted(set(paths))]
    signatures = {json.dumps(item[1]["groups"], sort_keys=True) for item in payloads}
    if len(signatures) != 1:
        raise RuntimeError("Different fixed-split manifests found in Kaggle Inputs")
    return payloads[0]


def find_recorded_proxy(checkpoint: dict, roots: list[Path]) -> tuple[Path, dict]:
    expected_hash = checkpoint.get("proxy_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Starting checkpoint has no recorded proxy SHA-256")
    candidates = []
    for root in roots:
        if root.is_dir():
            candidates.extend(root.glob("**/best.pt"))
    for path in sorted(set(candidates)):
        if file_sha256(path) != expected_hash:
            continue
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "proxy" not in payload or "proxy_config" not in payload:
            raise ValueError(f"Recorded proxy hash points to a non-proxy checkpoint: {path}")
        return path, payload
    raise FileNotFoundError(
        "The Output contains best_task_bd_rate.pt but not its recorded proxy best.pt. "
        "Attach the complete saved run/output that contains proxy_calibrated/best.pt."
    )


def prepare(project: Path = Path("/kaggle/working/proxy_v3"),
            clean: Path = CLEAN_DEFAULT) -> RecoveryRun:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU in Kaggle Settings")
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
    project = project.resolve(strict=True)
    working, input_root = Path("/kaggle/working"), Path("/kaggle/input")
    checkpoint_path, checkpoint = find_start_checkpoint(input_root, working, torch)
    manifest_path, manifest = find_manifest(input_root, working)
    groups = manifest["groups"]
    train = list(groups["train"])
    pool = set(groups["train_pool"])
    full_validation = set(groups["validation_full"])
    if not set(train) <= pool or pool & full_validation or set(train) & full_validation:
        raise ValueError("Saved train-pool/train/full-validation membership is inconsistent")
    unused = sorted(pool - set(train))
    controller = balanced_controller(unused, CONTROLLER_SIZE, int(manifest.get("seed", 42)) + 303)
    if set(controller) & set(train) or set(controller) & full_validation:
        raise AssertionError("Recovery controller overlaps train or full validation")
    # Resolve every member before creating any split links.
    for member in train + controller:
        _safe_source(clean, member)

    starting_proxy, proxy_payload = find_recorded_proxy(
        checkpoint, [working / RUN_NAME, input_root]
    )
    run_root = working / ("v7_rate_recovery_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_root.mkdir(exist_ok=False)
    split_root = run_root / "split"
    link_members(clean, train, split_root / "train")
    link_members(clean, controller, split_root / "controller_800")
    recovery_manifest = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "method": "controller_from_unused_train_pool_class_balanced",
        "seed": int(manifest.get("seed", 42)) + 303,
        "train": train,
        "controller_800": controller,
        "reserved_validation_full_count": len(full_validation),
        "disjoint": True,
    }
    (run_root / "recovery_split.json").write_text(
        json.dumps(recovery_manifest, indent=2), encoding="utf-8"
    )
    selection = {
        "start_checkpoint": str(checkpoint_path),
        "start_checkpoint_sha256": file_sha256(checkpoint_path),
        "start_epoch": checkpoint.get("epoch"),
        "start_controller_bd_rate_percent": checkpoint["val_metrics"]["task_bd_rate_percent"],
        "recorded_proxy": str(starting_proxy),
        "recorded_proxy_sha256": checkpoint["proxy_sha256"],
        "target_bpp_ratio": TARGET_BPP_RATIO,
        "finetune_lr": FINETUNE_LR,
        "finetune_epochs": FINETUNE_EPOCHS,
    }
    (run_root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    result = RecoveryRun(
        project=project, root=run_root, clean=clean, manifest=manifest_path,
        checkpoint=checkpoint_path, starting_proxy=starting_proxy,
        train_dir=split_root / "train", controller_dir=split_root / "controller_800",
        proxy_dir=run_root / "proxy_calibrated", swin_dir=run_root / "swin_ratio_095",
        saved_args=checkpoint["args"], proxy_config=proxy_payload["proxy_config"],
    )
    print(json.dumps({"run_root": str(result.root), "checkpoint": str(result.checkpoint),
                      "checkpoint_epoch": checkpoint.get("epoch"),
                      "checkpoint_controller_bd_rate": checkpoint["val_metrics"]["task_bd_rate_percent"],
                      "train_videos": len(train), "new_controller_videos": len(controller),
                      "starting_proxy": str(starting_proxy)}, indent=2), flush=True)
    del checkpoint, proxy_payload
    return result


def run_script(run: RecoveryRun, script: str, *arguments, check: bool = True):
    command = [sys.executable, "-u", str(run.project / script), *map(str, arguments)]
    return subprocess.run(command, cwd=run.project, check=check)


def calibrate_proxy(run: RecoveryRun) -> Path:
    saved, config = run.saved_args, run.proxy_config
    video = ["--frames", saved["frames"], "--frame-stride", saved["frame_stride"],
             "--frame-size", saved["frame_size"]]
    architecture = []
    for name in ("hidden_channels", "latent_channels", "bottleneck_channels",
                 "blocks_per_stage", "film_channels", "qp_step_divisor", "max_delta"):
        architecture.extend(["--" + name.replace("_", "-"), config[name]])
    run_script(
        run, "train_proxy.py", "--train-dir", run.train_dir, "--val-dir", run.controller_dir,
        *video, *architecture, "--init-checkpoint", run.starting_proxy,
        "--preprocessor-checkpoint", run.checkpoint,
        "--codec", "h264", "--qps", *TRAIN_QPS,
        "--fps", saved.get("codec_fps", 30), "--preset", saved.get("codec_preset", "medium"),
        "--seed", saved.get("seed", 42), "--rate-loss", "log", "--rate-weight", 0.1,
        "--rate-delta-weight", 0.1, "--pair-strengths", 0, 0.15, 0.35,
        "--gradient-probe-batches", 4, "--gradient-probe-step", 2 / 255,
        "--limit-train", 1000, "--limit-val", 400,
        "--epochs", 3, "--lr", 5e-5, "--batch-size", 4, "--workers", 2,
        "--amp", "--output-dir", run.proxy_dir,
    )
    best = run.proxy_dir / "best.pt"
    if not best.is_file():
        raise FileNotFoundError("Proxy calibration did not produce best.pt")
    import torch
    payload = torch.load(best, map_location="cpu", weights_only=False)
    print("Calibrated proxy validation:")
    print(json.dumps(payload["val_metrics"], indent=2), flush=True)
    return best


def saved_training_cli(saved: dict, overrides: dict, parser) -> list[str]:
    values = {**saved, **overrides}
    result = []
    for action in parser._actions:
        value = values.get(action.dest)
        if action.dest == "help" or not action.option_strings or value is None:
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            result.append(option if value else action.option_strings[1])
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(option)
        else:
            result.append(option)
            result.extend(map(str, value if isinstance(value, (list, tuple)) else [value]))
    return result


def fine_tune(run: RecoveryRun, proxy: Path) -> tuple[Path, bool]:
    os.chdir(run.project)
    sys.path.insert(0, str(run.project))
    import train as training
    parser = importlib.reload(training).build_parser()
    overrides = {
        "data_root": None, "train_dir": str(run.train_dir), "val_dir": str(run.controller_dir),
        "limit_train": None, "limit_val": None, "controller_limit_val": None,
        "init_checkpoint": str(run.checkpoint), "resume": None,
        "refresh_proxy_on_resume": False, "proxy_checkpoint": str(proxy),
        "preprocessor": "swin", "swin_qp_conditioning": True,
        "swin_gated_smoothing": True, "swin_smoothing_max_strength": 0.5,
        "max_residual": 0.10, "codec": "h264", "codec_qps": TRAIN_QPS,
        "train_codec_source": "real", "rate_lambda": [0.0], "alpha": 10.0,
        "distortion_reconstruction_weight": 1.0, "ce_weight": 1.0,
        "kd_weight": 0.5, "kd_temperature": 2.0,
        "feature_weight": 0.05, "feature_layer": "layer4",
        "max_top1_drop_pp": 0.0, "mask_rate_weight": 0.25,
        "mask_rate_target": "output", "mask_rate_layer": "layer4",
        "mask_rate_outside_weight": 0.0, "mask_rate_temporal_weight": 0.25,
        "mask_rate_temporal_target": "residual", "dual_rate_control": False,
        "rate_dual_control": True, "target_bpp_ratio": TARGET_BPP_RATIO,
        "rate_dual_parity_lambda": 0.05, "rate_dual_kappa": 3.0,
        "rate_dual_ema_beta": 0.8, "rate_dual_min": 0.0001, "rate_dual_max": 10.0,
        "rate_dual_max_proxy_underestimate_percent": 10.0,
        "rate_dual_proxy_guard_patience": 1, "dual_feasibility_tolerance": 0.005,
        "normalize_rate_by_anchor": False, "qp_sampling_weights": None,
        "epochs": FINETUNE_EPOCHS, "batch_size": 1, "accumulation_steps": 4,
        "lr": FINETUNE_LR, "optimizer": "adamw", "weight_decay": 0.01,
        "clip_grad": 1.0, "workers": 2, "amp": True, "device": "cuda",
        "checkpoint_metric": "task_bd_rate", "output_dir": str(run.swin_dir),
        "smoke_test": False,
    }
    arguments = saved_training_cli(run.saved_args, overrides, parser)
    outcome = run_script(run, "train.py", *arguments, check=False)
    import torch
    last = run.swin_dir / "last.pt"
    if not last.is_file():
        raise RuntimeError(f"Fine-tuning failed before writing last.pt (exit {outcome.returncode})")
    last_payload = torch.load(last, map_location="cpu", weights_only=False)
    if last_payload.get("val_metrics", {}).get("rate_dual_proxy_guard_abort"):
        raise RuntimeError("Proxy guard aborted rate recovery; do not evaluate this run")
    if int(last_payload.get("epoch", 0)) != FINETUNE_EPOCHS:
        raise RuntimeError(
            f"Fine-tuning stopped at epoch {last_payload.get('epoch')} of {FINETUNE_EPOCHS}"
        )
    feasible = run.swin_dir / "best_feasible.pt"
    diagnostic = run.swin_dir / "best_task_bd_rate.pt"
    candidate = feasible if feasible.is_file() else diagnostic
    if not candidate.is_file():
        raise FileNotFoundError("Fine-tuning produced no candidate checkpoint")
    print({"train_exit_code": outcome.returncode, "candidate": str(candidate),
           "feasible": feasible.is_file()}, flush=True)
    if not feasible.is_file():
        print("No feasible checkpoint: the selected best-task file is diagnostic only.", flush=True)
    return candidate, feasible.is_file()


def evaluate_candidate(run: RecoveryRun, candidate: Path) -> Path:
    import torch
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    score = float(payload["val_metrics"]["task_bd_rate_percent"])
    from kaggle_cells import evaluate_best_task_from_inputs as evaluation
    evaluation = importlib.reload(evaluation)
    return evaluation.main([
        "--checkpoint", str(candidate), "--manifest", str(run.manifest),
        "--clean", str(run.clean), "--expected-controller-bd-rate", str(score),
    ])
