# %% Cell 1: Load the UPDATED source archive attached as a Kaggle Input.
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

archives = list(Path("/kaggle/input").rglob("proxy_v3_v6_source.zip"))
if archives:
    assert len(archives) == 1, "Select one version of proxy_v3_v6_source.zip."
    with zipfile.ZipFile(archives[0]) as archive:
        manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
        source_files = {name: archive.read(name) for name in manifest}
else:
    manifests = [path for path in Path("/kaggle/input").rglob("SOURCE_MANIFEST.json")
                 if (path.parent / "presets/v6_accuracy_rate.args").is_file()]
    assert len(manifests) == 1, "Attach the source ZIP or its extracted files as a Kaggle Input."
    manifest = json.loads(manifests[0].read_text())
    source_files = {name: (manifests[0].parent / name).read_bytes() for name in manifest}
PROJECT = Path("/kaggle/working/proxy_v3_v6")
source_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
marker = PROJECT / ".source_sha256"
if PROJECT.exists():
    assert marker.is_file() and marker.read_text() == source_digest, "Use a new PROJECT directory for different source."
else:
    PROJECT.mkdir(parents=True)
    for name, data in source_files.items():
        assert hashlib.sha256(data).hexdigest() == manifest[name], f"Source checksum mismatch: {name}"
        destination = PROJECT / name
        assert destination.resolve().is_relative_to(PROJECT.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    marker.write_text(source_digest)
assert (PROJECT / "presets/v6_accuracy_rate.args").is_file()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(PROJECT / "requirements.txt")], check=True)

# %% Cell 2: Reuse the fixed, cleaned split; never silently resplit old experiments.
import torch

SPLIT_ROOT = Path("/kaggle/working/v5_fixed_split")
TRAIN_DIR = SPLIT_ROOT / "train"
VAL_DIR = SPLIT_ROOT / "controller"
FULL_VAL_DIR = SPLIT_ROOT / "validation_full"
TEST_DIR = None  # Optional separate, untouched class-folder test directory.
for folder in (TRAIN_DIR, VAL_DIR, FULL_VAL_DIR):
    assert folder.is_dir(), f"Missing {folder}. Run fixed_split_after_cleaning.ipynb first."
assert torch.cuda.is_available(), "Enable a Kaggle GPU."
subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)

RUN_ROOT = Path("/kaggle/working/v6_task_rate")
CACHE = RUN_ROOT / "cache"
BASE = RUN_ROOT / "proxy_base"
WARM = RUN_ROOT / "swin_warmup"
CALIBRATED = RUN_ROOT / "proxy_calibrated"
TARGETS = [0.90]  # For separate ablations: [0.95, 0.90, 0.85].
PROXY_EPOCHS, WARMUP_EPOCHS, CALIBRATION_EPOCHS, TRAIN_EPOCHS = 20, 5, 3, 15
MAX_TOP1_DROP_PP = 0.0

VIDEO = ["--frames", 16, "--frame-stride", 2, "--frame-size", 128]
SPLIT = ["--train-dir", TRAIN_DIR, "--val-dir", VAL_DIR, "--seed", 42, "--val-ratio", 0.2]

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
