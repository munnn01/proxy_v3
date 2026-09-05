"""Evaluate anchor and preprocessed clips with real H.264/H.265 codecs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import torch
from torch.nn import functional as F
from tqdm import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from preprocessing import FrozenVideoAnalyzer, StandardVideoCodec, build_preprocessor
from preprocessing.evaluation import (
    bootstrap_bd_rate,
    build_evaluation_dataset,
    calculate_bd_rate_details,
    dataset_sample_path,
)
from preprocessing.utils import topk_correct, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--test-dir")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--val-ratio",
        type=float,
        help="override checkpoint validation ratio when val/ is absent",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="override checkpoint split seed when val/ is absent",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        choices=("h264", "h265"),
        default=["h264", "h265"],
    )
    parser.add_argument(
        "--qps",
        nargs="+",
        type=int,
        default=[30, 32, 35, 37, 40, 42, 45],
        help="dense operating points improve BD-rate reliability",
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--preset")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include-clean-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="measure frozen-analyzer accuracy before the codec and show it on plots",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="paired video-level bootstrap repetitions; use 0 to disable",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/real_codec")
    return parser.parse_args()


def format_bd_rate(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.2f}%"


def format_bd_rate_interval(uncertainty: dict[str, object] | None) -> str:
    if not uncertainty:
        return ""
    lower = uncertainty.get("lower_percent")
    upper = uncertainty.get("upper_percent")
    if lower is None or upper is None:
        return ""
    confidence = 100.0 * float(uncertainty["confidence_level"])
    return f" ({confidence:.0f}% CI {float(lower):+.2f}% to {float(upper):+.2f}%)"


def save_rate_accuracy_plot(
    path: Path,
    rows: list[dict[str, float | int | str]],
    codec: str,
    task_bd_rate: float | None,
    psnr_bd_rate: float | None,
    clean_top1_percent: float | None = None,
    task_uncertainty: dict[str, object] | None = None,
) -> None:
    colors = {"anchor": "#E45756", "preprocessed": "#4C78A8"}
    labels = {"anchor": "Anchor", "preprocessed": "Video Swin preprocessor"}
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    codec_rows = [row for row in rows if row["codec"] == codec]
    for method in ("anchor", "preprocessed"):
        method_rows = [row for row in codec_rows if row["method"] == method]
        qp_rows = sorted(method_rows, key=lambda row: int(row["qp"]))
        rate_rows = sorted(method_rows, key=lambda row: float(row["bpp"]))
        color = colors[method]
        axes[0].plot(
            [row["qp"] for row in qp_rows],
            [row["bpp"] for row in qp_rows],
            marker="o",
            linewidth=2,
            color=color,
            label=labels[method],
        )
        axes[1].plot(
            [row["bpp"] for row in rate_rows],
            [row["top1_percent"] for row in rate_rows],
            marker="o",
            linewidth=2,
            color=color,
            label=labels[method],
        )
        axes[2].plot(
            [row["bpp"] for row in rate_rows],
            [row["psnr_db"] for row in rate_rows],
            marker="o",
            linewidth=2,
            color=color,
            label=labels[method],
        )
        for row in rate_rows:
            axes[1].annotate(
                f"QP {int(row['qp'])}",
                (float(row["bpp"]), float(row["top1_percent"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    axes[0].set(xlabel="QP", ylabel="Bitrate (BPP)", title="QP - BPP")
    axes[1].set(
        xlabel="Bitrate (BPP)",
        ylabel="Top-1 accuracy (%)",
        title="Task rate-accuracy",
    )
    if clean_top1_percent is not None:
        axes[1].axhline(
            clean_top1_percent,
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label=f"Clean reference ({clean_top1_percent:.2f}%)",
        )
    axes[2].set(
        xlabel="Bitrate (BPP)",
        ylabel="PSNR (dB)",
        title="Classical rate-distortion",
    )
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(
        f"{codec.upper()} | Task BD-rate: {format_bd_rate(task_bd_rate)}"
        f"{format_bd_rate_interval(task_uncertainty)} | "
        f"PSNR BD-rate: {format_bd_rate(psnr_bd_rate)}",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_top1_bd_rate_plot(
    path: Path,
    rows: list[dict[str, float | int | str]],
    codec: str,
    task_bd_rate: float | None,
    clean_top1_percent: float | None = None,
    task_uncertainty: dict[str, object] | None = None,
) -> None:
    """Write a presentation-ready Top-1/BPP curve with one BD-rate value."""

    colors = {"anchor": "#E45756", "preprocessed": "#4C78A8"}
    labels = {"anchor": "Anchor", "preprocessed": "Video Swin Lite v2"}
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    codec_rows = [row for row in rows if row["codec"] == codec]
    for method in ("anchor", "preprocessed"):
        rate_rows = sorted(
            (row for row in codec_rows if row["method"] == method),
            key=lambda row: float(row["bpp"]),
        )
        axis.plot(
            [row["bpp"] for row in rate_rows],
            [row["top1_percent"] for row in rate_rows],
            marker="o",
            markersize=7,
            linewidth=2.2,
            color=colors[method],
            label=labels[method],
        )
        for row in rate_rows:
            axis.annotate(
                f"QP {int(row['qp'])}",
                (float(row["bpp"]), float(row["top1_percent"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    if clean_top1_percent is not None:
        axis.axhline(
            clean_top1_percent,
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label=f"Clean reference ({clean_top1_percent:.2f}%)",
        )
    axis.set(
        xlabel="Bitrate (BPP)",
        ylabel="Top-1 accuracy (%)",
        title=(
            f"{codec.upper()} Top-1 BD-rate: {format_bd_rate(task_bd_rate)}"
            f"{format_bd_rate_interval(task_uncertainty)}"
        ),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def real_codec_roundtrip(
    clip: torch.Tensor,
    codec: str,
    qp: int,
    fps: float,
    *,
    preset: str = "medium",
    ffmpeg: str = "ffmpeg",
) -> tuple[torch.Tensor, float]:
    module = StandardVideoCodec(
        codec, qp, fps=fps, preset=preset, ffmpeg=ffmpeg
    )
    reconstruction, bpp = module(clip.unsqueeze(0))
    return reconstruction[0], float(bpp[0])


def main() -> None:
    args = parse_args()
    if not args.qps or len(set(args.qps)) != len(args.qps):
        raise ValueError("--qps must contain at least one unique value")
    if any(qp < 0 or qp > 51 for qp in args.qps):
        raise ValueError("--qps values must lie in [0, 51]")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must lie strictly between zero and one")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    codec_fps = args.fps if args.fps is not None else float(saved_args.get("codec_fps", 30.0))
    codec_preset = args.preset or saved_args.get("codec_preset", "medium")
    analyzer_name = saved_args.get("analyzer", "r3d_18")
    preprocessor_kind = saved_args.get("preprocessor", "cnn")
    analyzer = FrozenVideoAnalyzer(analyzer_name).to(device).eval()
    preprocessor = build_preprocessor(
        preprocessor_kind,
        temporal_frames=int(saved_args.get("temporal_frames", 8)),
        patch_size=int(saved_args.get("vit_patch_size", 8)),
        embed_dim=int(saved_args.get("vit_embed_dim", 96)),
        depth=int(saved_args.get("vit_depth", 4)),
        num_heads=int(saved_args.get("vit_heads", 4)),
        swin_patch_size=int(saved_args.get("swin_patch_size", 4)),
        swin_embed_dim=int(saved_args.get("swin_embed_dim", 48)),
        swin_depth=int(saved_args.get("swin_depth", 4)),
        swin_num_heads=int(saved_args.get("swin_heads", 4)),
        swin_window_size=(
            int(saved_args.get("swin_window_temporal", 4)),
            int(saved_args.get("swin_window_spatial", 8)),
            int(saved_args.get("swin_window_spatial", 8)),
        ),
        # Checkpoints created before QP conditioning have no FiLM parameters.
        swin_qp_conditioning=bool(saved_args.get("swin_qp_conditioning", False)),
        swin_qp_embed_dim=int(saved_args.get("swin_qp_embed_dim", 64)),
        swin_gated_smoothing=bool(saved_args.get("swin_gated_smoothing", False)),
        swin_smoothing_max_strength=float(saved_args.get("swin_smoothing_max_strength", 0.5)),
        max_residual=float(saved_args.get("max_residual", 0.25)),
    ).to(device).eval()
    preprocessor.load_state_dict(checkpoint["preprocessor"])

    dataset = build_evaluation_dataset(
        data_root=args.data_root,
        test_dir=args.test_dir,
        split=args.split,
        categories=analyzer.categories,
        frames=args.frames,
        stride=args.frame_stride,
        size=args.frame_size,
        limit=args.limit,
        val_ratio=args.val_ratio,
        seed=args.seed,
        saved_args=saved_args,
    )
    totals: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: {"videos": 0, "bpp": 0.0, "mse": 0.0, "top1": 0, "top5": 0}
    )
    clean_totals = {"videos": 0, "top1": 0, "top5": 0}
    per_video_rows: list[dict[str, float | int | str]] = []
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_video_path = output / "per_video_metrics.csv"
    per_video_fields = [
        "sample_index",
        "source",
        "label",
        "codec",
        "qp",
        "method",
        "bpp",
        "mse",
        "top1",
        "top5",
    ]
    with per_video_path.open("w", newline="", encoding="utf-8") as stream:
        per_video_writer = csv.DictWriter(stream, fieldnames=per_video_fields)
        per_video_writer.writeheader()
        iterator = enumerate(tqdm(dataset, desc="real codec evaluation"))
        for sample_index, (clip, label) in iterator:
            source_path = str(dataset_sample_path(dataset, sample_index))
            source = clip.to(device).unsqueeze(0)
            label_tensor = torch.tensor([label], device=device)
            if args.include_clean_reference:
                with torch.no_grad():
                    clean_logits = analyzer(source)
                clean_totals["videos"] += 1
                clean_totals["top1"] += topk_correct(clean_logits, label_tensor, 1)
                clean_totals["top5"] += topk_correct(clean_logits, label_tensor, 5)
            for qp in args.qps:
                # A QP-conditioned preprocessor emits a distinct clip for each
                # operating point. Legacy/CNN/ViT preprocessors simply ignore QP.
                with torch.no_grad():
                    proposed = preprocessor(source, qp)[0].cpu()
                for codec in args.codecs:
                    for method, input_clip in (
                        ("anchor", clip),
                        ("preprocessed", proposed),
                    ):
                        decoded, bpp = real_codec_roundtrip(
                            input_clip,
                            codec,
                            qp,
                            codec_fps,
                            preset=codec_preset,
                            ffmpeg=args.ffmpeg,
                        )
                        decoded_device = decoded.to(device).unsqueeze(0)
                        with torch.no_grad():
                            logits = analyzer(decoded_device)
                        mse = float(F.mse_loss(decoded_device, source))
                        top1 = topk_correct(logits, label_tensor, 1)
                        top5 = topk_correct(logits, label_tensor, 5)
                        key = (codec, qp, method)
                        row = totals[key]
                        row["videos"] += 1
                        row["bpp"] += bpp
                        row["mse"] += mse
                        row["top1"] += top1
                        row["top5"] += top5
                        sample_row: dict[str, float | int | str] = {
                            "sample_index": sample_index,
                            "source": source_path,
                            "label": int(label),
                            "codec": codec,
                            "qp": qp,
                            "method": method,
                            "bpp": bpp,
                            "mse": mse,
                            "top1": top1,
                            "top5": top5,
                        }
                        per_video_writer.writerow(sample_row)
                        per_video_rows.append(
                            {
                                "sample_index": sample_index,
                                "codec": codec,
                                "qp": qp,
                                "method": method,
                                "bpp": bpp,
                                "mse": mse,
                                "top1": top1,
                            }
                        )

    rows = []
    for (codec, qp, method), values in sorted(totals.items()):
        count = int(values["videos"])
        mse = values["mse"] / count
        rows.append(
            {
                "codec": codec,
                "qp": qp,
                "method": method,
                "videos": count,
                "bpp": values["bpp"] / count,
                "mse": mse,
                "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
                "top1": values["top1"] / count,
                "top1_percent": 100.0 * values["top1"] / count,
                "top5": values["top5"] / count,
            }
        )
    clean_metrics = None
    if clean_totals["videos"]:
        clean_count = int(clean_totals["videos"])
        clean_metrics = {
            "videos": clean_count,
            "top1": clean_totals["top1"] / clean_count,
            "top1_percent": 100.0 * clean_totals["top1"] / clean_count,
            "top5": clean_totals["top5"] / clean_count,
        }
        write_json(output / "clean_metrics.json", clean_metrics)
        print(
            f"[clean] Top-1={clean_metrics['top1_percent']:.2f}% "
            f"Top-5={100.0 * clean_metrics['top5']:.2f}%"
        )
    write_json(output / "metrics.json", rows)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output / 'metrics.csv'}")
    print(f"wrote {per_video_path}")

    write_json(
        output / "evaluation_config.json",
        {
            "schema_version": 1,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "videos": len(dataset),
            "codecs": list(args.codecs),
            "qps": list(args.qps),
            "frames": args.frames,
            "frame_stride": args.frame_stride,
            "frame_size": args.frame_size,
            "codec_fps": codec_fps,
            "codec_preset": codec_preset,
            "stream_scope": "one_elementary_stream_per_clip",
            "bpp_includes_elementary_stream_headers": True,
            "keyint": "clip_frame_count",
            "scene_cut": False,
            "analyzer": analyzer_name,
            "preprocessor": preprocessor_kind,
            "bd_rate_interpolation": "pchip",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "confidence_level": args.confidence_level,
        },
    )

    bd_rates = {}
    for codec in args.codecs:
        codec_rows = [row for row in rows if row["codec"] == codec]
        codec_sample_rows = [
            row for row in per_video_rows if row["codec"] == codec
        ]
        task_details = calculate_bd_rate_details(codec_rows, "top1_percent")
        psnr_details = calculate_bd_rate_details(codec_rows, "psnr_db")
        task_bd_rate = task_details["bd_rate_percent"]
        psnr_bd_rate = psnr_details["bd_rate_percent"]
        task_uncertainty = bootstrap_bd_rate(
            codec_sample_rows,
            "top1_percent",
            samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.bootstrap_seed,
        )
        psnr_uncertainty = bootstrap_bd_rate(
            codec_sample_rows,
            "psnr_db",
            samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.bootstrap_seed + 1,
        )
        bd_rates[codec] = {
            "task_bd_rate_percent": task_bd_rate,
            "psnr_bd_rate_percent": psnr_bd_rate,
            "task_details": task_details,
            "psnr_details": psnr_details,
            "task_bootstrap": task_uncertainty,
            "psnr_bootstrap": psnr_uncertainty,
        }
        plot_path = output / f"{codec}_top1_bpp_bd_rate.png"
        save_rate_accuracy_plot(
            plot_path,
            rows,
            codec,
            task_bd_rate,
            psnr_bd_rate,
            None if clean_metrics is None else float(clean_metrics["top1_percent"]),
            task_uncertainty,
        )
        top1_plot_path = output / f"{codec}_top1_bd_rate.png"
        save_top1_bd_rate_plot(
            top1_plot_path,
            rows,
            codec,
            task_bd_rate,
            None if clean_metrics is None else float(clean_metrics["top1_percent"]),
            task_uncertainty,
        )
        print(
            f"[{codec}] task BD-rate={format_bd_rate(task_bd_rate)}"
            f"{format_bd_rate_interval(task_uncertainty)} "
            f"PSNR BD-rate={format_bd_rate(psnr_bd_rate)}"
        )
        print(f"wrote {plot_path}")
        print(f"wrote {top1_plot_path}")
    write_json(output / "bd_rate.json", bd_rates)
    print(f"wrote {output / 'bd_rate.json'}")


if __name__ == "__main__":
    main()
