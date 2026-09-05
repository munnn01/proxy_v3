# %% Cell 1: Clone/update the GitHub source and install dependencies.
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import sys

PROJECT = Path("/kaggle/working/proxy_v3")
if PROJECT.exists():
    subprocess.run(["git", "-C", str(PROJECT), "pull", "--ff-only", "origin", "main"], check=True)
else:
    subprocess.run(["git", "clone", "-q", "--branch", "main",
                    "https://github.com/munnn01/proxy_v3.git", str(PROJECT)], check=True)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(PROJECT / "requirements.txt")], check=True)

# %% Cell 2: Automatically split the cleaned dataset and configure this run.
import torch
from torchvision.models.video import R3D_18_Weights
from preprocessing.data import (
    VideoFolderDataset,
    stratified_limit_indices,
    stratified_split_indices,
)

DATA = Path("/kaggle/input/datasets/qktttttttttt/kineticscleaned/cleaned_final/kinetics400_5per/kinetics400_5per/train")
SEED = 42
VAL_RATIO = 0.2
TRAIN_LIMIT = 2000  # Set to None to use the complete training pool.
CONTROLLER_LIMIT = 400
RUN_NAME = "v6_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_ROOT = Path("/kaggle/working") / RUN_NAME
SPLIT_ROOT = RUN_ROOT / "split"
TRAIN_DIR = SPLIT_ROOT / "train"
VAL_DIR = SPLIT_ROOT / "controller"
FULL_VAL_DIR = SPLIT_ROOT / "validation_full"
TEST_DIR = None  # Optional separate, untouched class-folder test directory.

categories = R3D_18_Weights.DEFAULT.meta["categories"]
source = VideoFolderDataset(DATA, categories, train=False)
train_pool, val_ids = stratified_split_indices(source.samples, VAL_RATIO, SEED)
train_ids = stratified_limit_indices(source.samples, train_pool, TRAIN_LIMIT, SEED + 101)
controller_ids = stratified_limit_indices(source.samples, val_ids, CONTROLLER_LIMIT, SEED + 202)

def link_split(indices, destination_root):
    destination_root.mkdir(parents=True, exist_ok=True)
    for index in indices:
        source_path, _ = source.samples[index]
        destination = destination_root / source_path.relative_to(DATA)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source_path.resolve())

link_split(train_ids, TRAIN_DIR)
link_split(controller_ids, VAL_DIR)
link_split(val_ids, FULL_VAL_DIR)
print({"train": len(train_ids), "controller": len(controller_ids),
       "validation_full": len(val_ids), "run_root": str(RUN_ROOT)})

CACHE = RUN_ROOT / "cache"
BASE = RUN_ROOT / "proxy_base"
WARM = RUN_ROOT / "swin_warmup"
CALIBRATED = RUN_ROOT / "proxy_calibrated"
TARGETS = [0.90]  # For separate ablations: [0.95, 0.90, 0.85].
PROXY_EPOCHS, WARMUP_EPOCHS, CALIBRATION_EPOCHS, TRAIN_EPOCHS = 20, 5, 3, 15
MAX_TOP1_DROP_PP = 0.0

VIDEO = ["--frames", 16, "--frame-stride", 2, "--frame-size", 128]
SPLIT = ["--train-dir", TRAIN_DIR, "--val-dir", VAL_DIR, "--seed", SEED, "--val-ratio", VAL_RATIO]

def run(script, *arguments, check=True):
    command = [sys.executable, "-u", str(PROJECT / script), *map(str, arguments)]
    return subprocess.run(command, cwd=PROJECT, check=check)

# %% Cell 3: Cache H.264 targets, then fit the initial proxy with relative-rate loss.
run("precompute_codec.py", *SPLIT, *VIDEO,
    "--codec", "h264", "--qps", 30, 35, 40, 45,
    "--fps", 30, "--preset", "medium", "--codec-io", "pipe",
    "--codec-workers", 2, "--output-dir", CACHE)
run("train_proxy.py", "--precomputed-root", CACHE, *VIDEO,
    "--codec", "h264", "--qps", 30, 35, 40, 45,
    "--rate-loss", "log", "--rate-weight", 0.1,
    "--epochs", PROXY_EPOCHS, "--batch-size", 4, "--workers", 2,
    "--output-dir", BASE)

# %% Cell 4: Produce Swin outputs and recalibrate the proxy on these paired variants.
run("train.py", "@presets/v3_masked_rate.args", *SPLIT, *VIDEO,
    "--proxy-checkpoint", BASE / "best.pt",
    "--swin-gated-smoothing", "--max-residual", 0.10,
    "--mask-rate-weight", 0.25, "--mask-rate-outside-weight", 0,
    "--mask-rate-temporal-target", "residual", "--mask-rate-temporal-weight", 0.25,
    "--epochs", WARMUP_EPOCHS, "--batch-size", 1, "--accumulation-steps", 4,
    "--workers", 2, "--output-dir", WARM)
# last.pt supplies the current distribution for calibration; it is not a final result.
run("train_proxy.py", "--precomputed-root", CACHE, *VIDEO,
    "--init-checkpoint", BASE / "best.pt",
    "--preprocessor-checkpoint", WARM / "last.pt",
    "--rate-loss", "log", "--rate-weight", 0.1, "--rate-delta-weight", 0.1,
    "--pair-strengths", 0, 0.15, 0.35,
    "--gradient-probe-batches", 4, "--gradient-probe-step", 2 / 255,
    "--limit-train", 1000, "--limit-val", 400,
    "--epochs", CALIBRATION_EPOCHS, "--lr", 5e-5,
    "--batch-size", 4, "--workers", 2, "--output-dir", CALIBRATED)
calibration = torch.load(CALIBRATED / "best.pt", map_location="cpu", weights_only=False)
print(json.dumps(calibration["val_metrics"], indent=2))
del calibration

# %% Cell 5: Train fresh target runs; best.pt must pass BOTH real BPP and Top-1 guards.
candidates = []
for target in TARGETS:
    destination = RUN_ROOT / f"swin_ratio_{target:.2f}"
    assert not any(destination.glob("*.pt")), "Use a new RUN_ROOT or explicitly resume the old run."
    outcome = run("train.py", "@presets/v6_accuracy_rate.args", *SPLIT, *VIDEO,
        "--proxy-checkpoint", CALIBRATED / "best.pt", "--init-checkpoint", WARM / "last.pt",
        "--target-bpp-ratio", target, "--max-top1-drop-pp", MAX_TOP1_DROP_PP,
        "--epochs", TRAIN_EPOCHS, "--batch-size", 1, "--accumulation-steps", 4,
        "--workers", 2, "--output-dir", destination, check=False)
    print({"target": target, "exit_code": outcome.returncode, "directory": str(destination)})
    if outcome.returncode != 0:
        print("Run failed or a guard stopped it. Inspect the traceback and last.pt diagnostics.")
    checkpoint = destination / "best.pt"
    if checkpoint.is_file():
        candidates.append(checkpoint)
assert candidates, "No feasible checkpoint. Inspect proxy drift and Top-1 drops; do not report last.pt as a winner."

# %% Cell 6: Measure complete development curves; report success only from real H.264.
import csv
import math

def summarize(directory):
    bd = json.loads((directory / "bd_rate.json").read_text())["h264"]
    with (directory / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    anchors = {int(row["qp"]): row for row in rows if row["method"] == "anchor"}
    proposed = [row for row in rows if row["method"] == "preprocessed"]
    drop = max(float(anchors[int(row["qp"])]["top1_percent"]) - float(row["top1_percent"])
               for row in proposed)
    ratio = max(float(row["bpp"]) / float(anchors[int(row["qp"])]["bpp"]) for row in proposed)
    point = bd["task_bd_rate_percent"]
    upper = bd["task_bootstrap"].get("upper_percent")
    return {"bd_rate": point, "ci_upper": upper, "max_top1_drop_pp": drop,
            "max_bpp_ratio": ratio,
            "accuracy_ok": drop <= MAX_TOP1_DROP_PP + 1e-9,
            "rate_ok": ratio <= 1.0,
            "target_met": (point is not None and point < -10 and ratio <= 1.0
                           and drop <= MAX_TOP1_DROP_PP + 1e-9),
            "confident_below_minus10": (upper is not None and upper < -10 and ratio <= 1.0
                                       and drop <= MAX_TOP1_DROP_PP + 1e-9)}

reports = []
for checkpoint in candidates:
    destination = RUN_ROOT / "evaluation" / checkpoint.parent.name
    run("evaluate_real_codec.py", "--checkpoint", checkpoint, "--test-dir", FULL_VAL_DIR,
        *VIDEO, "--codecs", "h264", "--qps", 30, 32, 35, 37, 40, 42, 45,
        "--bootstrap-samples", 2000, "--device", "cuda", "--output-dir", destination)
    report = {"checkpoint": str(checkpoint), **summarize(destination)}
    reports.append(report)
print(json.dumps(reports, indent=2))
eligible = [r for r in reports if r["accuracy_ok"] and r["bd_rate"] is not None
            and math.isfinite(r["bd_rate"]) and r["max_bpp_ratio"] <= 1.0]
assert eligible, "Full validation has no candidate preserving both accuracy and per-QP BPP."
winner = min(eligible, key=lambda row: row["bd_rate"])
print("Development winner:", winner)
# FULL_VAL_DIR overlaps controller data, so it is development validation, not an untouched test.
if TEST_DIR is not None:
    destination = RUN_ROOT / "evaluation" / "heldout_test"
    run("evaluate_real_codec.py", "--checkpoint", winner["checkpoint"], "--test-dir", TEST_DIR,
        *VIDEO, "--codecs", "h264", "--qps", 30, 32, 35, 37, 40, 42, 45,
        "--bootstrap-samples", 2000, "--device", "cuda", "--output-dir", destination)
    print("Held-out test:", summarize(destination))
