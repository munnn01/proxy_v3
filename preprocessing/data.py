"""Class-folder video loading with Kinetics-400 label alignment."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import cv2
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def stratified_split_indices(
    samples: Sequence[tuple[Path, int]], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split sample indices per class while keeping at least one training video."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(samples):
        by_label[label].append(index)

    generator = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_label.values():
        generator.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            continue
        val_count = max(1, round(len(indices) * val_ratio))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    if not val_indices:
        raise RuntimeError("automatic validation split produced no samples")
    generator.shuffle(train_indices)
    generator.shuffle(val_indices)
    return train_indices, val_indices


def stratified_limit_indices(
    samples: Sequence[tuple[Path, int]],
    indices: Sequence[int],
    limit: int | None,
    seed: int,
) -> list[int]:
    """Apply a deterministic, approximately class-balanced subset limit.

    A plain ``indices[:limit]`` can omit rare classes even after a stratified
    split. Round-robin selection gives every represented class one sample
    before taking a second sample from any class.
    """

    selected_pool = list(indices)
    if limit is None or limit >= len(selected_pool):
        return selected_pool
    if limit < 1:
        raise ValueError("dataset limits must be positive")

    by_label: dict[int, list[int]] = defaultdict(list)
    for index in selected_pool:
        by_label[int(samples[index][1])].append(index)
    generator = random.Random(seed)
    for label_indices in by_label.values():
        generator.shuffle(label_indices)

    labels = sorted(by_label)
    generator.shuffle(labels)
    balanced: list[int] = []
    while len(balanced) < limit:
        added = False
        for label in labels:
            if by_label[label]:
                balanced.append(by_label[label].pop())
                added = True
                if len(balanced) == limit:
                    break
        if not added:
            break
    generator.shuffle(balanced)
    return balanced


class VideoFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    """Read ``root/class_name/video.*`` and align names to analyzer categories."""

    def __init__(
        self,
        root: str | Path,
        categories: Sequence[str],
        *,
        frames: int = 16,
        stride: int = 2,
        size: int = 128,
        train: bool = False,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.frames = frames
        self.stride = stride
        self.size = size
        self.train = train
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {self.root}")

        category_lookup = {normalize_label(name): index for index, name in enumerate(categories)}
        samples: list[tuple[Path, int]] = []
        skipped: set[str] = set()
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            relative = path.relative_to(self.root)
            if len(relative.parts) < 2:
                skipped.add("<videos directly under root>")
                continue
            class_name = relative.parts[0]
            key = normalize_label(class_name)
            if key not in category_lookup:
                skipped.add(class_name)
                continue
            samples.append((path, category_lookup[key]))

        if skipped:
            preview = ", ".join(sorted(skipped)[:8])
            print(f"[data] skipped folders not present in Kinetics-400 labels: {preview}")
        if limit is not None:
            samples = samples[:limit]
        if not samples:
            raise RuntimeError(
                f"no labeled videos found below {self.root}; expected root/class_name/video.mp4"
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _indices(self, frame_count: int) -> list[int]:
        span = (self.frames - 1) * self.stride + 1
        maximum_start = max(0, frame_count - span)
        start = random.randint(0, maximum_start) if self.train else maximum_start // 2
        return [min(start + i * self.stride, max(frame_count - 1, 0)) for i in range(self.frames)]

    def _decode(self, path: Path) -> torch.Tensor:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        frames: list[torch.Tensor] = []
        last: torch.Tensor | None = None
        for index in self._indices(frame_count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if ok:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                last = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
            if last is None:
                capture.release()
                raise RuntimeError(f"no decodable frames in {path}")
            frames.append(last.clone())
        capture.release()
        return torch.stack(frames)

    def _spatial_transform(self, clip: torch.Tensor) -> torch.Tensor:
        _, _, height, width = clip.shape
        resize_to = max(self.size, round(self.size * max(height, width) / min(height, width)))
        if height <= width:
            new_height, new_width = self.size, resize_to
        else:
            new_height, new_width = resize_to, self.size
        clip = F.interpolate(
            clip, size=(new_height, new_width), mode="bilinear", align_corners=False
        )
        max_top, max_left = new_height - self.size, new_width - self.size
        if self.train:
            top = random.randint(0, max_top) if max_top else 0
            left = random.randint(0, max_left) if max_left else 0
        else:
            top, left = max_top // 2, max_left // 2
        clip = clip[..., top : top + self.size, left : left + self.size]
        if self.train and random.random() < 0.5:
            clip = clip.flip(-1)
        return clip.contiguous()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            return self._spatial_transform(self._decode(path)), label
        except RuntimeError:
            if not self.train:
                raise
            # Corrupt web videos are common in Kinetics mirrors. Retry another sample.
            replacement = random.randrange(len(self.samples))
            if replacement == index:
                replacement = (replacement + 1) % len(self.samples)
            return self.__getitem__(replacement)


class PrecomputedCodecDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]
):
    """Read deterministic uint8 clips and codec targets without invoking FFmpeg."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        qps: Sequence[int] | None = None,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"precomputed manifest does not exist: {manifest_path}; "
                "run precompute_codec.py first"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if split not in self.manifest.get("splits", {}):
            raise ValueError(f"split {split!r} is not present in {manifest_path}")
        available_qps = [int(value) for value in self.manifest["codec"]["qps"]]
        self.qps = available_qps if qps is None else [int(value) for value in qps]
        missing = sorted(set(self.qps).difference(available_qps))
        if missing:
            raise ValueError(
                f"QPs {missing} are absent from the cache; available QPs: {available_qps}"
            )
        self.split = split
        self.samples = list(self.manifest["splits"][split])
        self.items = [
            (sample, qp) for sample in self.samples for qp in self.qps
        ]
        if not self.items:
            raise RuntimeError(f"precomputed split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        sample, qp = self.items[index]
        sample_id = sample["id"]
        clip_payload = torch.load(
            self.root / self.split / "clips" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        target_payload = torch.load(
            self.root / self.split / "recon" / f"qp_{qp}" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        clip = clip_payload["clip"].float().div_(255.0)
        reconstruction = target_payload["reconstruction"].float().div_(255.0)
        bpp = torch.as_tensor(target_payload["bpp"], dtype=torch.float32)
        return clip, reconstruction, bpp, qp


class MixedQPBatchSampler(Sampler[list[int]]):
    """Build deterministic batches containing a balanced mixture of cached QPs."""

    def __init__(
        self,
        dataset: PrecomputedCodecDataset,
        batch_size: int,
        *,
        seed: int = 42,
    ) -> None:
        if batch_size < len(dataset.qps):
            raise ValueError(
                f"batch_size must be >= the number of QPs ({len(dataset.qps)})"
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        qp_count = len(self.dataset.qps)
        buckets = [
            [
                sample_index * qp_count + qp_index
                for sample_index in range(len(self.dataset.samples))
            ]
            for qp_index in range(qp_count)
        ]
        for bucket in buckets:
            generator.shuffle(bucket)
        positions = [0] * qp_count
        remaining = len(self.dataset)
        while remaining:
            order = list(range(qp_count))
            generator.shuffle(order)
            batch: list[int] = []
            while len(batch) < self.batch_size and remaining:
                made_progress = False
                for qp_index in order:
                    if len(batch) == self.batch_size:
                        break
                    position = positions[qp_index]
                    if position >= len(buckets[qp_index]):
                        continue
                    batch.append(buckets[qp_index][position])
                    positions[qp_index] += 1
                    remaining -= 1
                    made_progress = True
                if not made_progress:
                    break
            yield batch
