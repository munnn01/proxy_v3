"""Visualize source -> preprocessor -> standard codec -> analyzer for one clip."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch.nn import functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from preprocessing import FrozenVideoAnalyzer, StandardVideoCodec, build_preprocessor
from preprocessing.evaluation import build_evaluation_dataset, dataset_sample_path
from preprocessing.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--test-dir")
    parser.add_argument("--split", default="val")
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--codec", choices=("h264", "h265"))
    parser.add_argument("--codec-qp", type=int)
    parser.add_argument("--codec-fps", type=float)
    parser.add_argument("--codec-preset")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--frame-size", type=int)
    parser.add_argument("--show-frames", type=int, default=4)
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/visualization")
    return parser.parse_args()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    count = min(length, count)
    return np.linspace(0, length - 1, count, dtype=int).tolist()


def top1_prediction(
    logits: torch.Tensor, categories: list[str], target: int
) -> dict[str, float | str | bool]:
    probabilities = logits.softmax(dim=1)[0]
    index = int(probabilities.argmax())
    return {
        "label": categories[index],
        "probability": float(probabilities[index]),
        "correct": index == target,
        "top1_accuracy": float(index == target),
    }


def clip_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    mse = float(F.mse_loss(candidate.float(), reference.float()))
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    mae = float(F.l1_loss(candidate.float(), reference.float()))
    return {"mse": mse, "mae": mae, "psnr_db": psnr}


def tensor_image(frame: torch.Tensor) -> np.ndarray:
    return frame.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def error_map(reference: torch.Tensor, candidate: torch.Tensor) -> np.ndarray:
    return (candidate - reference).abs().mean(dim=0).detach().cpu().numpy()


def save_comparison_figure(
    path: Path,
    source: torch.Tensor,
    processed: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    indices: list[int],
    class_name: str,
    bpp: float,
    reconstruction_psnr: float,
    prediction: str,
    top1_accuracy: float,
) -> None:
    preprocessor_errors = [error_map(source[index], processed[index]) for index in indices]
    codec_errors = [error_map(source[index], reconstructed[index]) for index in indices]
    error_scale = max(
        float(np.percentile(np.concatenate([x.ravel() for x in preprocessor_errors]), 99)),
        float(np.percentile(np.concatenate([x.ravel() for x in codec_errors]), 99)),
        1e-6,
    )

    columns = [
        "Source",
        "Preprocessed",
        "Standard-codec reconstruction",
        "Preprocessor |delta|",
        "End-to-end |error|",
    ]
    figure, axes = plt.subplots(
        len(indices), len(columns), figsize=(15, 3 * len(indices)), squeeze=False
    )
    for row, frame_index in enumerate(indices):
        axes[row, 0].imshow(tensor_image(source[frame_index]))
        axes[row, 1].imshow(tensor_image(processed[frame_index]))
        axes[row, 2].imshow(tensor_image(reconstructed[frame_index]))
        heatmap = axes[row, 3].imshow(
            preprocessor_errors[row], cmap="magma", vmin=0.0, vmax=error_scale
        )
        axes[row, 4].imshow(codec_errors[row], cmap="magma", vmin=0.0, vmax=error_scale)
        axes[row, 0].set_ylabel(f"Frame {frame_index}")
        for column, title in enumerate(columns):
            if row == 0:
                axes[row, column].set_title(title)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    figure.suptitle(
        f"Target: {class_name} | reconstruction: {prediction} | "
        f"Top-1 acc: {top1_accuracy:.0%} | {bpp:.4f} bpp | "
        f"{reconstruction_psnr:.2f} dB",
        y=0.995,
    )
    # Reserve a dedicated strip outside the image grid so the colorbar never
    # obscures the final end-to-end error column.
    figure.subplots_adjust(left=0.04, right=0.88, bottom=0.02, top=0.94, wspace=0.04, hspace=0.08)
    error_axes = axes[:, 4].ravel().tolist()
    colorbar_bottom = min(axis.get_position().y0 for axis in error_axes)
    colorbar_top = max(axis.get_position().y1 for axis in error_axes)
    colorbar_axis = figure.add_axes([0.91, colorbar_bottom, 0.015, colorbar_top - colorbar_bottom])
    figure.colorbar(heatmap, cax=colorbar_axis, label="Mean RGB error")
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def uint8_bgr(frame: torch.Tensor) -> np.ndarray:
    rgb = tensor_image(frame)
    return cv2.cvtColor(np.round(rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def colored_error(reference: torch.Tensor, candidate: torch.Tensor, scale: float) -> np.ndarray:
    error = np.clip(error_map(reference, candidate) / max(scale, 1e-6), 0.0, 1.0)
    return cv2.applyColorMap(np.round(error * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(output, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
    return output


def save_comparison_video(
    path: Path,
    source: torch.Tensor,
    processed: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    fps: float,
    bpp: float,
    top1_accuracy: float,
) -> bool:
    all_errors = torch.cat(
        ((processed - source).abs().flatten(), (reconstructed - source).abs().flatten())
    )
    error_scale = max(float(torch.quantile(all_errors.float(), 0.99)), 1e-6)
    height, width = source.shape[-2:]
    header_height = 32
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 5, height + header_height),
    )
    if not writer.isOpened():
        return False

    labels = ["Source", "Preprocessed", "Reconstructed", "Preprocessor delta", "Total error"]
    for index in range(source.shape[0]):
        panels = [
            uint8_bgr(source[index]),
            uint8_bgr(processed[index]),
            uint8_bgr(reconstructed[index]),
            colored_error(source[index], processed[index], error_scale),
            colored_error(source[index], reconstructed[index], error_scale),
        ]
        panel = np.concatenate(
            [add_label(image, label) for image, label in zip(panels, labels, strict=True)], axis=1
        )
        header = np.zeros((header_height, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"Frame {index + 1}/{source.shape[0]} | Top-1 acc {top1_accuracy:.0%} | "
            f"standard codec {bpp:.4f} bpp",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        writer.write(np.concatenate((header, panel), axis=0))
    writer.release()
    return True


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    frames = args.frames or int(saved_args.get("frames", 16))
    stride = args.frame_stride or int(saved_args.get("frame_stride", 2))
    size = args.frame_size or int(saved_args.get("frame_size", 128))
    analyzer_name = saved_args.get("analyzer", "r3d_18")
    codec_name = args.codec or saved_args.get("codec", checkpoint.get("codec", "h264"))
    qp = args.codec_qp if args.codec_qp is not None else int(checkpoint.get("codec_qp", 35))
    codec_fps = (
        args.codec_fps
        if args.codec_fps is not None
        else float(saved_args.get("codec_fps", 30.0))
    )
    codec_preset = args.codec_preset or saved_args.get("codec_preset", "medium")

    analyzer = FrozenVideoAnalyzer(analyzer_name).to(device).eval()
    dataset = build_evaluation_dataset(
        data_root=args.data_root,
        test_dir=args.test_dir,
        split=args.split,
        categories=analyzer.categories,
        frames=frames,
        stride=stride,
        size=size,
        limit=args.limit,
        val_ratio=args.val_ratio,
        seed=args.seed,
        saved_args=saved_args,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample index {args.sample_index} is outside [0, {len(dataset) - 1}]")
    source, label = dataset[args.sample_index]
    source_path = dataset_sample_path(dataset, args.sample_index)

    preprocessor = build_preprocessor(
        saved_args.get("preprocessor", "cnn"),
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
        swin_qp_conditioning=bool(saved_args.get("swin_qp_conditioning", False)),
        swin_qp_embed_dim=int(saved_args.get("swin_qp_embed_dim", 64)),
        max_residual=float(saved_args.get("max_residual", 0.25)),
    ).to(device).eval()
    preprocessor.load_state_dict(checkpoint["preprocessor"])
    codec = StandardVideoCodec(
        codec_name,
        qp,
        fps=codec_fps,
        preset=codec_preset,
        ffmpeg=args.ffmpeg,
    ).to(device).eval()
    source_batch = source.unsqueeze(0).to(device)
    use_amp = bool(args.amp and device.type == "cuda")
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp),
    ):
        processed_batch = preprocessor(source_batch, qp)
        reconstructed_batch, bpp = codec(processed_batch)
        source_logits = analyzer(source_batch)
        processed_logits = analyzer(processed_batch)
        reconstructed_logits = analyzer(reconstructed_batch)

    processed = processed_batch[0].float().cpu()
    reconstructed = reconstructed_batch[0].float().cpu()
    metrics = {
        "source_video": str(source_path),
        "sample_index": args.sample_index,
        "target": analyzer.categories[label],
        "analyzer": analyzer_name,
        "codec": codec_name,
        "codec_qp": qp,
        "measured_bpp": float(bpp.float().mean()),
        "preprocessor_change": clip_metrics(source, processed),
        "reconstruction": clip_metrics(source, reconstructed),
        "top1": {
            "source": top1_prediction(source_logits.float(), analyzer.categories, label),
            "preprocessed": top1_prediction(
                processed_logits.float(), analyzer.categories, label
            ),
            "reconstructed": top1_prediction(
                reconstructed_logits.float(), analyzer.categories, label
            ),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "pipeline-comparison.png"
    video_path = output_dir / "pipeline-comparison.mp4"
    metrics_path = output_dir / "pipeline-metrics.json"
    indices = evenly_spaced_indices(frames, args.show_frames)
    save_comparison_figure(
        figure_path,
        source,
        processed,
        reconstructed,
        indices=indices,
        class_name=analyzer.categories[label],
        bpp=metrics["measured_bpp"],
        reconstruction_psnr=metrics["reconstruction"]["psnr_db"],
        prediction=metrics["top1"]["reconstructed"]["label"],
        top1_accuracy=metrics["top1"]["reconstructed"]["top1_accuracy"],
    )
    video_saved = args.save_video and save_comparison_video(
        video_path,
        source,
        processed,
        reconstructed,
        fps=args.video_fps,
        bpp=metrics["measured_bpp"],
        top1_accuracy=metrics["top1"]["reconstructed"]["top1_accuracy"],
    )
    metrics["outputs"] = {
        "figure": str(figure_path),
        "video": str(video_path) if video_saved else None,
    }
    write_json(metrics_path, metrics)
    print(f"wrote {figure_path}")
    if video_saved:
        print(f"wrote {video_path}")
    elif args.save_video:
        print("warning: OpenCV MP4 writer unavailable; PNG and JSON were still written")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
