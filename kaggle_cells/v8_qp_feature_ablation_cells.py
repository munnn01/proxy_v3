# Kaggle Inputs: full V7 Output, V6 -6.687% best-task checkpoint,
# checking/v5_fixed_split/split_manifest.json, and kineticscleaned.
# Enable GPU + Internet. Run cells in order. This is a new two-epoch experiment.

# %% 1. Update code and import helpers
from pathlib import Path
import subprocess, sys, importlib

PROJECT = Path("/kaggle/working/proxy_v3")
if PROJECT.exists():
    subprocess.run(["git", "-C", str(PROJECT), "pull", "--ff-only", "origin", "main"], check=True)
else:
    subprocess.run(["git", "clone", "https://github.com/munnn01/proxy_v3.git", str(PROJECT)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT / "requirements.txt")], check=True)
sys.path.insert(0, str(PROJECT))
importlib.invalidate_caches()
from preprocessing import feature_distillation
from kaggle_cells import v7_rate_recovery, v8_feature_ablation
importlib.reload(feature_distillation)
importlib.reload(v7_rate_recovery)
v8 = importlib.reload(v8_feature_ablation)

# %% 2. Select inputs and per-QP feature coefficients
# Auto-find the original V6 checkpoint, V7 proxy, and exact V7 controller split.
RUN = v8.prepare()
FEATURE_BY_QP = {30: 0.03, 35: 0.04, 40: 0.06, 45: 0.07}
FEATURE_WEIGHTS = [FEATURE_BY_QP[q] for q in [30, 35, 40, 45]]
print("Feature weights:", FEATURE_BY_QP)
print("Swin input:", RUN.checkpoint)
print("Calibrated proxy input:", RUN.starting_proxy)
print("Output:", RUN.root)

# %% 3. Measure the starting V6 checkpoint on the same 800-video controller
BASELINE_JSON = v8.baseline(RUN, feature_weights_by_qp=FEATURE_WEIGHTS)

# %% 4. Fine-tune two epochs with a fresh optimizer and rate-dual controller
CANDIDATE_PT, IS_FEASIBLE = v8.train(RUN, feature_weights_by_qp=FEATURE_WEIGHTS)
print("Candidate:", CANDIDATE_PT, "Feasible:", IS_FEASIBLE)

# %% 5. Compare on the controller, then evaluate real H.264 at all seven QPs
import json, torch
import pandas as pd
from IPython.display import display, FileLink

initial = json.loads(BASELINE_JSON.read_text())["val_metrics"]
payload = torch.load(CANDIDATE_PT, map_location="cpu", weights_only=False)
candidate = payload["val_metrics"]
print("Controller initial BD-rate (%):", initial["task_bd_rate_percent"])
print("Controller candidate BD-rate (%):", candidate["task_bd_rate_percent"])
print("Feasible:", IS_FEASIBLE, "Epoch:", payload["epoch"])
display(pd.DataFrame([
    {"QP": q, "feature_weight": candidate[f"qp{q}_feature_weight"],
     "initial_BPP_ratio": initial[f"qp{q}_bpp_ratio"],
     "candidate_BPP_ratio": candidate[f"qp{q}_bpp_ratio"],
     "initial_Top1_%": 100 * initial[f"qp{q}_top1"],
     "candidate_Top1_%": 100 * candidate[f"qp{q}_top1"]}
    for q in [30, 35, 40, 45]
]))
del payload
if not IS_FEASIBLE:
    print("Diagnostic checkpoint: BPP/Top-1 constraints were not met on the controller.")
EVAL_DIR = v7_rate_recovery.evaluate_candidate(RUN, CANDIDATE_PT)
for name in ("metrics.csv", "per_video_metrics.csv", "bd_rate.json"):
    display(FileLink(str(EVAL_DIR / name)))
