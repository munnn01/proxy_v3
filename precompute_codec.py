"""Pre-compute deterministic uint8 H.264/H.265 targets for proxy training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from torchvision.models.video import R3D_18_Weights

from preprocessing import StandardVideoCodec
from preprocessing.data import VideoFolderDataset, stratified_split_indices
from preprocessing.standard_codec import require_ffmpeg
from preprocessing.utils import save_checkpoint, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument("--data-root", help="root containing train/ and optionally val/")
    data.add_argument("--train-dir")
    data.add_argument("--val-dir")
    data.add_argument("--val-ratio", type=float, default=0.1)
    data.add_argument("--frames", type=int, default=16)
    data.add_argument("--frame-stride", type=int, default=2)
    data.add_argument("--frame-size", type=int, default=128)
    data.add_argument("--limit-train", type=int)
    data.add_argument("--limit-val", type=int)

    codec = parser.add_argument_group("codec")
    codec.add_argument("--codec", choices=("h264", "h265"), default="h264")
    codec.add_argument("--qps", type=int, nargs="+", default=[30, 35, 40, 45])
    codec.add_argument("--fps", type=float, default=30.0)
    codec.add_argument("--preset", default="medium")
    codec.add_argument("--ffmpeg", default="ffmpeg")
    codec.add_argument("--codec-io", choices=("pipe", "png"), default="pipe")
    codec.add_argument("--codec-workers", type=int, default=2)
    codec.add_argument("--ffmpeg-threads", type=int, default=1)
    codec.add_argument(
        "--verify-pipe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare raw pipe against the legacy PNG path before caching",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="precomputed_codec/h264")
    return parser.parse_args()


def resolve_directories(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.train_dir:
        return Path(args.train_dir), Path(args.val_dir) if args.val_dir else None
    if not args.data_root:
        raise ValueError("provide --data-root or --train-dir")
    root = Path(args.data_root)
    train = root / "train" if (root / "train").is_dir() else root
    validation = root / "val"
    return train, validation if validation.is_dir() else None


def make_splits(
    args: argparse.Namespace,
) -> dict[str, tuple[VideoFolderDataset, list[int]]]:
    train_dir, val_dir = resolve_directories(args)
    categories = list(R3D_18_Weights.DEFAULT.meta["categories"])
    options = {
        "frames": args.frames,
        "stride": args.frame_stride,
        "size": args.frame_size,
        "train": False,
    }
    train_source = VideoFolderDataset(train_dir, categories, **options)
    if val_dir is not None:
        val_source = VideoFolderDataset(val_dir, categories, **options)
        train_indices = list(range(len(train_source)))
        val_indices = list(range(len(val_source)))
    else:
        train_indices, val_indices = stratified_split_indices(
            train_source.samples, args.val_ratio, args.seed
        )
        val_source = train_source
    if args.limit_train is not None:
        train_indices = train_indices[: args.limit_train]
    if args.limit_val is not None:
        val_indices = val_indices[: args.limit_val]
    return {
        "train": (train_source, train_indices),
        "val": (val_source, val_indices),
    }


def uint8_clip(clip: torch.Tensor) -> torch.Tensor:
    return clip.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)


def dataset_fingerprint(
    splits: dict[str, tuple[VideoFolderDataset, list[int]]]
) -> str:
    digest = hashlib.sha256()
    for split in sorted(splits):
        dataset, indices = splits[split]
        for index in indices:
            path, label = dataset.samples[index]
            digest.update(f"{split}\0{path.resolve()}\0{label}\n".encode("utf-8"))
    return digest.hexdigest()


def verify_pipe(codec: StandardVideoCodec, clip: torch.Tensor, qps: list[int]) -> None:
    pipe = codec
    png = StandardVideoCodec(
        codec.codec,
        qps[0],
        fps=codec.fps,
        preset=codec.preset,
        ffmpeg=codec.ffmpeg,
        io_backend="png",
        codec_workers=1,
        ffmpeg_threads=codec.ffmpeg_threads,
    )
    sample = clip.unsqueeze(0)
    for qp in qps:
        pipe.set_qp(qp)
        png.set_qp(qp)
        pipe_reconstruction, pipe_bpp = pipe(sample)
        png_reconstruction, png_bpp = png(sample)
        pipe_pixels = uint8_clip(pipe_reconstruction)
        png_pixels = uint8_clip(png_reconstruction)
        max_difference = int(
            (pipe_pixels.to(torch.int16) - png_pixels.to(torch.int16)).abs().max()
        )
        bpp_difference = abs(float(pipe_bpp.item()) - float(png_bpp.item()))
        if max_difference != 0 or bpp_difference > 1e-7:
            raise RuntimeError(
                "raw pipe verification failed at "
                f"QP {qp}: max pixel difference={max_difference}, "
                f"BPP difference={bpp_difference:.10f}; use --codec-io png"
            )
    print(f"[verify] raw pipe is pixel/BPP exact for QPs {qps}")


def first_decodable_clip(
    splits: dict[str, tuple[VideoFolderDataset, list[int]]]
) -> torch.Tensor:
    for dataset, indices in splits.values():
        for index in indices:
            try:
                clip, _ = dataset[index]
                return clip
            except RuntimeError as error:
                print(f"[data] verification sample skipped: {error}")
    raise RuntimeError("no decodable video is available for pipe verification")


def flush_batch(
    output_dir: Path,
    split: str,
    batch: list[tuple[str, torch.Tensor]],
    codec: StandardVideoCodec,
    qps: list[int],
) -> None:
    if not batch:
        return
    for sample_id, clip in batch:
        save_checkpoint(
            output_dir / split / "clips" / f"{sample_id}.pt",
            {"clip": uint8_clip(clip)},
        )

    clips = torch.stack([clip for _, clip in batch])
    for qp in qps:
        codec.set_qp(qp)
        reconstructions, bpps = codec(clips)
        for (sample_id, _), reconstruction, bpp in zip(
            batch, reconstructions, bpps, strict=True
        ):
            save_checkpoint(
                output_dir
                / split
                / "recon"
                / f"qp_{qp}"
                / f"{sample_id}.pt",
                {
                    "reconstruction": uint8_clip(reconstruction),
                    "bpp": bpp.cpu(),
                },
            )


def cache_split(
    output_dir: Path,
    split: str,
    dataset: VideoFolderDataset,
    indices: list[int],
    codec: StandardVideoCodec,
    qps: list[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    pending: list[tuple[str, torch.Tensor]] = []
    iterator = tqdm(indices, desc=f"precompute {split}")
    for source_index in iterator:
        source_path, label = dataset.samples[source_index]
        sample_id = f"{source_index:08d}"
        clip_path = output_dir / split / "clips" / f"{sample_id}.pt"
        target_paths = [
            output_dir / split / "recon" / f"qp_{qp}" / f"{sample_id}.pt"
            for qp in qps
        ]
        if clip_path.is_file() and all(path.is_file() for path in target_paths):
            records.append(
                {"id": sample_id, "source": str(source_path), "label": int(label)}
            )
            continue
        try:
            clip, _ = dataset[source_index]
        except RuntimeError as error:
            skipped.append(str(source_path))
            print(f"[data] skipped corrupt video: {error}")
            continue
        records.append(
            {"id": sample_id, "source": str(source_path), "label": int(label)}
        )
        pending.append((sample_id, clip))
        if len(pending) == codec.codec_workers:
            flush_batch(output_dir, split, pending, codec, qps)
            pending.clear()
    flush_batch(output_dir, split, pending, codec, qps)
    return records, skipped


def main() -> None:
    args = parse_args()
    if not args.qps or len(set(args.qps)) != len(args.qps):
        raise ValueError("--qps must contain distinct values")
    if any(qp < 0 or qp > 51 for qp in args.qps):
        raise ValueError("--qps must contain values in [0, 51]")
    if args.frame_size % 2:
        raise ValueError("--frame-size must be even for yuv420p H.264/H.265")
    require_ffmpeg(args.ffmpeg)
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    splits = make_splits(args)
    config = {
        "version": 1,
        "data": {
            "train_root": str(splits["train"][0].root.resolve()),
            "val_root": str(splits["val"][0].root.resolve()),
            "limit_train": args.limit_train,
            "limit_val": args.limit_val,
            "fingerprint": dataset_fingerprint(splits),
        },
        "codec": {
            "name": args.codec,
            "qps": list(args.qps),
            "fps": args.fps,
            "preset": args.preset,
            "io_backend": args.codec_io,
            "codec_workers": args.codec_workers,
            "ffmpeg_threads": args.ffmpeg_threads,
        },
        "video": {
            "frames": args.frames,
            "frame_stride": args.frame_stride,
            "frame_size": args.frame_size,
        },
        "split": {"val_ratio": args.val_ratio, "seed": args.seed},
    }
    config_path = output_dir / "config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                f"cache configuration differs from {config_path}; choose a new --output-dir"
            )
    else:
        write_json(config_path, config)

    codec = StandardVideoCodec(
        args.codec,
        args.qps[0],
        fps=args.fps,
        preset=args.preset,
        ffmpeg=args.ffmpeg,
        io_backend=args.codec_io,
        codec_workers=args.codec_workers,
        ffmpeg_threads=args.ffmpeg_threads,
    )
    if args.codec_io == "pipe" and args.verify_pipe:
        verify_pipe(codec, first_decodable_clip(splits), list(args.qps))

    manifest: dict[str, Any] = {**config, "splits": {}, "skipped": {}}
    for split, (dataset, indices) in splits.items():
        records, skipped = cache_split(
            output_dir, split, dataset, indices, codec, list(args.qps)
        )
        manifest["splits"][split] = records
        manifest["skipped"][split] = skipped
        print(f"[cache] {split}: {len(records)} clips, {len(skipped)} skipped")
    write_json(output_dir / "manifest.json", manifest)
    print(f"[cache] ready: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
