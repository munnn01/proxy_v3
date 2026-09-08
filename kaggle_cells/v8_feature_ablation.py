"""A short feature-loss ablation reusing the finished V7 split and calibrated proxy."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import json

from kaggle_cells import v7_rate_recovery as recovery

V7_NAME = "v7_rate_recovery_20260907_163149_952110"


def prepare(source_root=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU")
    working, inputs = Path("/kaggle/working"), Path("/kaggle/input")
    if source_root is None:
        candidates = [working / V7_NAME, *sorted(inputs.glob(f"**/{V7_NAME}"))]
        candidates = [p for p in candidates if (p / "recovery_split.json").is_file()
                      and (p / "proxy_calibrated/best.pt").is_file()]
        if not candidates:
            raise FileNotFoundError("Attach the V7 Output with recovery_split.json and proxy_calibrated/best.pt")
        signatures = {(recovery.file_sha256(p / "recovery_split.json"),
                       recovery.file_sha256(p / "proxy_calibrated/best.pt")) for p in candidates}
        if len(signatures) != 1:
            raise ValueError("Different V7 outputs found; pass source_root explicitly")
        source_root = candidates[0]
    source_root = Path(source_root)
    split = json.loads((source_root / "recovery_split.json").read_text())
    selection = json.loads((source_root / "selection.json").read_text())
    checkpoint_path, checkpoint = recovery.find_start_checkpoint(inputs, working, torch)
    if recovery.file_sha256(checkpoint_path) != selection["start_checkpoint_sha256"]:
        raise ValueError("The V6 checkpoint differs from the starting checkpoint recorded by V7")
    manifest_path, manifest = recovery.find_manifest(inputs, working)
    if recovery.file_sha256(manifest_path) != split["source_manifest_sha256"]:
        raise ValueError("Fixed split manifest differs from the one used in V7")
    train, controller = split["train"], split["controller_800"]
    if (set(train) != set(manifest["groups"]["train"])
            or len(train) != len(set(train)) or len(controller) != 800 or len(set(controller)) != 800
            or not set(controller) <= set(manifest["groups"]["train_pool"])
            or set(train) & set(controller)
            or set(controller) & set(manifest["groups"]["validation_full"])):
        raise ValueError("V7 split membership is inconsistent")
    proxy_path = source_root / "proxy_calibrated/best.pt"
    proxy = torch.load(proxy_path, map_location="cpu", weights_only=False)
    original_shape = checkpoint["args"]
    if (original_shape.get("preprocessor") != "swin" or not original_shape.get("swin_qp_conditioning")
            or not original_shape.get("swin_gated_smoothing")
            or original_shape.get("swin_smoothing_max_strength") != 0.5
            or original_shape.get("max_residual") != 0.10):
        raise ValueError("This ablation requires the existing V6 Swin/QP/smoothing configuration unchanged")
    clean = recovery.CLEAN_DEFAULT
    for member in train + controller:
        recovery._safe_source(clean, member)
    root = working / ("v8_feature_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    recovery.link_members(clean, train, root / "split/train")
    recovery.link_members(clean, controller, root / "split/controller_800")
    run = recovery.RecoveryRun(
        project=Path(__file__).resolve().parents[1], root=root, clean=clean, manifest=manifest_path,
        checkpoint=checkpoint_path, starting_proxy=proxy_path,
        train_dir=root / "split/train", controller_dir=root / "split/controller_800",
        proxy_dir=proxy_path.parent, swin_dir=root / "feature_relative_mse",
        saved_args=checkpoint["args"], proxy_config=proxy["proxy_config"],
    )
    (root / "selection.json").write_text(json.dumps({
        "source_v7": str(source_root), "start_checkpoint": str(checkpoint_path),
        "start_checkpoint_sha256": recovery.file_sha256(checkpoint_path),
        "proxy": str(proxy_path), "proxy_sha256": recovery.file_sha256(proxy_path),
        "source_split_sha256": recovery.file_sha256(source_root / "recovery_split.json"),
        "controller": "same 800 clips as V7", "no_new_proxy_training": True,
    }, indent=2))
    print("Starting Swin:", checkpoint_path)
    print("Reused V7 proxy:", proxy_path)
    print("Output:", root)
    return run


def loss_overrides(mode="relative_mse"):
    common = {"epochs": 2, "lr": 1e-5, "target_bpp_ratio": 0.95,
              "validate_initial": False, "initial_validation_only": False}
    if mode == "legacy":
        return {**common, "feature_loss": "cosine", "feature_layers": ["layer4"],
                "feature_layer_weights": [1.0], "feature_weight": 0.05}
    if mode != "relative_mse":
        raise ValueError("mode must be relative_mse or legacy")
    return {**common, "feature_loss": "relative_mse", "feature_layers": ["layer3", "layer4"],
            "feature_layer_weights": [0.3, 0.7], "feature_weight": 0.05}


def baseline(run):
    arguments = recovery.training_arguments(run, run.starting_proxy, {
        **loss_overrides(), "initial_validation_only": True,
    })
    recovery.run_script(run, "train.py", *arguments)
    path = run.swin_dir / "initial_validation.json"
    print("Initial Swin on the V7 controller:", path)
    return path


def train(run, mode="relative_mse"):
    if mode == "legacy":
        run = replace(run, swin_dir=run.root / "feature_legacy")
    candidate, feasible = recovery.fine_tune(run, run.starting_proxy, {
        **loss_overrides(mode), "validate_initial": not (run.swin_dir / "initial_validation.json").is_file(),
    })
    return candidate, feasible
