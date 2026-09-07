"""Evaluate a saved Swin checkpoint on the original full validation split."""
from pathlib import Path, PurePosixPath
from datetime import datetime
import argparse
import hashlib
import json
import math
import subprocess
import sys

RUN_NAME = "v6_calibration_20260905_161558_046557"
CLEAN_DEFAULT = (
    "/kaggle/input/datasets/qktttttttttt/kineticscleaned/cleaned_final/"
    "kinetics400_5per/kinetics400_5per/train"
)
QPS = [30, 32, 35, 37, 40, 42, 45]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_checkpoint(candidates, expected_score, loader):
    matches = []
    observed = []
    for path in sorted(set(candidates)):
        payload = loader(path)
        score = payload.get("val_metrics", {}).get("task_bd_rate_percent")
        epoch = payload.get("epoch")
        observed.append(f"{path}: epoch={epoch}, actual validation BD-rate={score}")
        if ("preprocessor" in payload and isinstance(score, (int, float))
                and math.isfinite(score) and abs(score - expected_score) < 0.0005):
            matches.append((path, score, epoch))
        del payload
    if not matches:
        raise RuntimeError(
            f"No Swin checkpoint with actual BD-rate rounded to {expected_score:.3f}%. "
            "Attach the latest saved Output containing swin_ratio_090/best_task_bd_rate.pt. "
            "An older Input version may contain a different checkpoint.\n" + "\n".join(observed)
        )
    if len({sha256(path) for path, _, _ in matches}) > 1:
        raise RuntimeError("Different checkpoints match this score; specify --checkpoint explicitly.")
    return matches[0]


def select_manifest(paths):
    if not paths:
        raise FileNotFoundError("Attach checking with v5_fixed_split/split_manifest.json.")
    ordered = sorted(set(paths))
    manifests = [read_json(path) for path in ordered]
    signatures = {
        json.dumps({key: sorted(values) for key, values in m["groups"].items()}, sort_keys=True)
        for m in manifests
    }
    if len(signatures) != 1:
        raise RuntimeError("Different split manifests found; specify --manifest explicitly.")
    return ordered[0], manifests[0]


def validation_sources(clean, manifest):
    groups = manifest["groups"]
    members = groups["validation_full"]
    if not members or len(set(members)) != len(members):
        raise ValueError("Full validation must be nonempty with no duplicate paths.")
    if set(groups["train"]) & set(members):
        raise ValueError("Train overlaps full validation.")
    if not set(groups["controller"]) <= set(members):
        raise ValueError("Controller is not a subset of the saved full validation.")
    root = clean.resolve(strict=True)
    sources = []
    for member in members:
        relative = PurePosixPath(member)
        if (relative.is_absolute() or len(relative.parts) < 2
                or ".." in relative.parts or "\\" in member or ":" in member):
            raise ValueError(f"Invalid relative video path: {member}")
        source = root.joinpath(*relative.parts).resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Video is outside the clean dataset or missing: {member}")
        sources.append((relative, source))
    return sources


def print_report(output):
    rows = [r for r in read_json(output / "metrics.json") if r["codec"] == "h264"]
    indexed = {(r["method"], int(r["qp"])): r for r in rows}
    expected = {(method, qp) for method in ("anchor", "preprocessed") for qp in QPS}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("Expected both methods at all seven H.264 QPs.")
    accuracy_ok = bpp_ok = True
    print("\n QP | H264 BPP | Swin BPP | BPP change | H264 Top1 | Swin Top1 | Top1 delta (pp)")
    for qp in QPS:
        anchor, swin = indexed["anchor", qp], indexed["preprocessed", qp]
        for row in (anchor, swin):
            if not (math.isfinite(row["bpp"]) and row["bpp"] > 0
                    and math.isfinite(row["top1_percent"]) and 0 <= row["top1_percent"] <= 100):
                raise ValueError(f"Invalid BPP/Top1 at QP {qp}.")
        delta = swin["top1_percent"] - anchor["top1_percent"]
        change = 100 * (swin["bpp"] / anchor["bpp"] - 1)
        accuracy_ok &= delta >= -1e-9
        bpp_ok &= change <= 1e-9
        print(f"{qp:3} | {anchor['bpp']:.6f} | {swin['bpp']:.6f} | {change:+8.3f}% | "
              f"{anchor['top1_percent']:8.3f}% | {swin['top1_percent']:8.3f}% | {delta:+.3f}")
    result = read_json(output / "bd_rate.json")["h264"]
    score = result["task_bd_rate_percent"]
    interval = result["task_bootstrap"]
    upper = interval.get("upper_percent")
    print("\nTask BD-rate (%):", score)
    print("Bootstrap interval (%):", interval.get("lower_percent"), "to", upper)
    print("Point estimate below -10%:", score is not None and math.isfinite(score) and score < -10)
    print("Interval upper bound below -10%:", upper is not None and math.isfinite(upper) and upper < -10)
    print("Top1 >= H264 at every QP:", accuracy_ok)
    print("BPP <= H264 at every QP:", bpp_ok)
    print("Development validation: includes controller clips used for checkpoint selection.")
    print("Full bootstrap diagnostics:", json.dumps(interval))
    print("Results:", output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clean", type=Path, default=Path(CLEAN_DEFAULT))
    parser.add_argument("--expected-controller-bd-rate", type=float, default=-6.687)
    args = parser.parse_args(argv)
    import torch

    project = Path(__file__).resolve().parents[1]
    input_root = Path("/kaggle/input")
    working = Path("/kaggle/working")
    candidates = ([args.checkpoint] if args.checkpoint else
                  list(input_root.glob(f"**/{RUN_NAME}/swin_ratio_090/best_task_bd_rate.pt")))
    local = working / RUN_NAME / "swin_ratio_090/best_task_bd_rate.pt"
    if args.checkpoint is None and local.is_file():
        candidates.append(local)
    checkpoint, controller_score, epoch = select_checkpoint(
        candidates, args.expected_controller_bd_rate,
        lambda path: torch.load(path, map_location="cpu", weights_only=False),
    )
    print(f"Checkpoint: {checkpoint}\nEpoch: {epoch}\nController BD-rate: {controller_score:.6f}%", flush=True)
    manifests = ([args.manifest] if args.manifest else
                 list(input_root.glob("**/v5_fixed_split/split_manifest.json")))
    local_manifest = working / "v5_fixed_split/split_manifest.json"
    if args.manifest is None and local_manifest.is_file():
        manifests.append(local_manifest)
    manifest_path, manifest = select_manifest(manifests)
    sources = validation_sources(args.clean, manifest)
    output = working / ("eval_best_task_7qp_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    full_val = output / "validation_full"
    for relative, source in sources:
        target = full_val.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["args"]
    # Ensure the analyzer label mapping does not silently omit manifest members.
    sys.path.insert(0, str(project))
    from preprocessing.analyzer import FrozenVideoAnalyzer
    from preprocessing.data import VideoFolderDataset
    analyzer = FrozenVideoAnalyzer(saved.get("analyzer", "r3d_18"))
    dataset = VideoFolderDataset(full_val, analyzer.categories, train=False)
    if len(dataset) != len(sources):
        raise ValueError("Analyzer category mapping omitted validation videos.")
    del analyzer, dataset
    command = [sys.executable, "-u", str(project / "evaluate_real_codec.py"),
               "--checkpoint", str(checkpoint), "--test-dir", str(full_val),
               "--codecs", "h264", "--qps", *map(str, QPS),
               "--bootstrap-samples", "2000", "--bootstrap-seed", "2026",
               "--confidence-level", "0.95", "--device", "cuda", "--output-dir", str(output)]
    for flag, key, default in (("frames", "frames", 16), ("frame-stride", "frame_stride", 2),
                               ("frame-size", "frame_size", 128), ("fps", "codec_fps", 30),
                               ("preset", "codec_preset", "medium")):
        command.extend(["--" + flag, str(saved.get(key, default))])
    provenance = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                  "epoch": epoch, "controller_bd_rate_percent": controller_score,
                  "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                  "clean": str(args.clean), "videos": len(sources), "command": command,
                  "split_role": "development_validation_including_controller"}
    (output / "selection.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Full validation: {len(sources)} videos\nOutput: {output}", flush=True)
    subprocess.run(command, cwd=project, check=True)
    config = read_json(output / "evaluation_config.json")
    if config["videos"] != len(sources):
        raise ValueError("Evaluator video count differs from the manifest.")
    print_report(output)
    return output


if __name__ == "__main__":
    main()
