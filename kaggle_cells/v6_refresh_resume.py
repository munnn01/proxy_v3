# %% Cell 1: Continue the current working run after a proxy guard stop.
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import torch

PROJECT = Path("/kaggle/working/proxy_v3")
RUN_ROOT = Path("/kaggle/working/v6_calibration_20260905_161558_046557")
FINAL_DIR = RUN_ROOT / "swin_ratio_090"
RESUME_PT = FINAL_DIR / "last.pt"
CALIBRATION_EPOCHS = 3
TOTAL_SWIN_EPOCHS = 15  # Total epoch number, not the number of additional epochs.

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
import train as training
build_parser = importlib.reload(training).build_parser

def run_script(script, *arguments):
    return subprocess.run(
        [sys.executable, "-u", str(PROJECT / script), *map(str, arguments)],
        cwd=PROJECT, check=True,
    )

if not RESUME_PT.is_file():
    raise FileNotFoundError(
        f"Missing {RESUME_PT}. Restore the latest run first; do not replace it "
        "with an older Input checkpoint. This recipe continues the working run."
    )
checkpoint = torch.load(RESUME_PT, map_location="cpu", weights_only=False)
saved_args = checkpoint["args"]
completed_epoch = int(checkpoint["epoch"])
if TOTAL_SWIN_EPOCHS <= completed_epoch:
    raise ValueError(f"Set TOTAL_SWIN_EPOCHS above the saved epoch {completed_epoch}.")
OLD_PROXY = Path(checkpoint.get("proxy_checkpoint", saved_args["proxy_checkpoint"]))
if hashlib.sha256(OLD_PROXY.read_bytes()).hexdigest() != checkpoint.get("proxy_sha256"):
    raise ValueError("The starting proxy differs from the frozen proxy recorded in last.pt.")
for name in ("train_dir", "val_dir"):
    if not saved_args.get(name) or not Path(saved_args[name]).is_dir():
        raise FileNotFoundError(f"Restore the original fixed split at {saved_args.get(name)} first.")

proxy_payload = torch.load(OLD_PROXY, map_location="cpu", weights_only=False)
proxy_config = proxy_payload["proxy_config"]
cache_hint = proxy_payload.get("args", {}).get("precomputed_root")
CACHE = Path(cache_hint) if cache_hint else None
if CACHE is None or not (CACHE / "manifest.json").is_file():
    caches = sorted({p for p in Path("/kaggle/input").rglob("codec_cache")
                     if (p / "manifest.json").is_file()})
    if len(caches) != 1:
        raise RuntimeError(f"Expected the checking Input codec_cache; found {caches}.")
    CACHE = caches[0]

REFRESH_ROOT = RUN_ROOT / (
    f"refresh_after_epoch_{completed_epoch:03d}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
REFRESH_ROOT.mkdir(parents=True, exist_ok=False)
SNAPSHOT = REFRESH_ROOT / f"swin_epoch_{completed_epoch:03d}.pt"
shutil.copy2(RESUME_PT, SNAPSHOT)
SNAPSHOT_SHA256 = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
NEW_PROXY_DIR = REFRESH_ROOT / "proxy_calibrated"
VIDEO_ARGS = ["--frames", saved_args["frames"], "--frame-stride", saved_args["frame_stride"],
              "--frame-size", saved_args["frame_size"]]
PROXY_MODEL_ARGS = []
for name in ("hidden_channels", "latent_channels", "bottleneck_channels", "blocks_per_stage",
             "film_channels", "qp_step_divisor", "max_delta"):
    PROXY_MODEL_ARGS.extend(["--" + name.replace("_", "-"), proxy_config[name]])
del checkpoint, proxy_payload
print(f"Saved Swin epoch: {completed_epoch}; next Swin epoch: {completed_epoch + 1}")
print("Starting calibrated proxy:", OLD_PROXY)
print("Refresh output:", NEW_PROXY_DIR)

# %% Cell 2: Refine the existing proxy on outputs of the saved Swin epoch.
run_script(
    "train_proxy.py", "--precomputed-root", CACHE, *VIDEO_ARGS, *PROXY_MODEL_ARGS,
    "--init-checkpoint", OLD_PROXY,
    "--preprocessor-checkpoint", SNAPSHOT,
    "--codec", saved_args["codec"], "--qps", *saved_args["codec_qps"],
    "--fps", saved_args["codec_fps"], "--preset", saved_args["codec_preset"],
    "--seed", saved_args["seed"], "--val-ratio", saved_args["val_ratio"],
    "--rate-loss", "log", "--rate-weight", 0.1, "--rate-delta-weight", 0.1,
    "--pair-strengths", 0, 0.15, 0.35,
    "--gradient-probe-batches", 4, "--gradient-probe-step", 2 / 255,
    "--limit-train", 1000, "--limit-val", 400,
    "--epochs", CALIBRATION_EPOCHS, "--lr", 5e-5,
    "--batch-size", 4, "--workers", 2, "--amp", "--output-dir", NEW_PROXY_DIR,
)
calibrated = torch.load(NEW_PROXY_DIR / "best.pt", map_location="cpu", weights_only=False)
print(json.dumps(calibrated["val_metrics"], indent=2))
del calibrated

# %% Cell 3: Resume the next Swin epoch with its saved configuration and training state.
def saved_training_cli(saved, overrides):
    """Serialize only supported options, including explicit False boolean settings."""
    values = {**saved, **overrides}
    result = []
    for action in build_parser()._actions:
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

if hashlib.sha256(RESUME_PT.read_bytes()).hexdigest() != SNAPSHOT_SHA256:
    raise RuntimeError("last.pt changed during calibration. Refresh from the latest Swin epoch.")
resume_arguments = saved_training_cli(saved_args, {
    "init_checkpoint": None,
    "resume": str(RESUME_PT),
    "refresh_proxy_on_resume": True,
    "proxy_checkpoint": str(NEW_PROXY_DIR / "best.pt"),
    "output_dir": str(FINAL_DIR),
    "epochs": TOTAL_SWIN_EPOCHS,
    "workers": 2,
    "device": "cuda",
    "smoke_test": False,
})
# Pre-training real-codec validation must pass the existing guard before Swin updates.
# A failed validation leaves last.pt unchanged. A later guard stop saves the new epoch.
run_script("train.py", *resume_arguments)
