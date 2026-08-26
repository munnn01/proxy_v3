"""Dataset and Bjontegaard helpers shared by final evaluation scripts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Subset

from .data import (
    VideoFolderDataset,
    stratified_limit_indices,
    stratified_split_indices,
)


def build_evaluation_dataset(
    *,
    categories: Sequence[str],
    data_root: str | Path | None = None,
    test_dir: str | Path | None = None,
    split: str = "val",
    frames: int = 16,
    stride: int = 2,
    size: int = 128,
    limit: int | None = None,
    val_ratio: float | None = None,
    seed: int | None = None,
    saved_args: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build a physical test split or reproduce the training validation split.

    An explicit ``test_dir`` or an existing ``data_root/split`` directory is
    evaluated directly.  Otherwise ``data_root`` is treated as the training
    class-folder tree and the same deterministic stratified split used by
    ``train.py`` is recreated in memory.
    """

    if test_dir is not None:
        dataset = VideoFolderDataset(
            Path(test_dir),
            categories,
            frames=frames,
            stride=stride,
            size=size,
            train=False,
            limit=limit,
        )
        print(f"[eval data] explicit directory: {len(dataset)} videos")
        return dataset

    if data_root is None:
        raise ValueError("provide --data-root or --test-dir")

    root = Path(data_root)
    split_root = root / split
    if split_root.is_dir():
        dataset = VideoFolderDataset(
            split_root,
            categories,
            frames=frames,
            stride=stride,
            size=size,
            train=False,
            limit=limit,
        )
        print(f"[eval data] physical {split!r} split: {len(dataset)} videos")
        return dataset

    if split not in {"val", "validation"}:
        raise FileNotFoundError(
            f"requested split directory does not exist: {split_root}; "
            "automatic splitting is supported only for val/validation"
        )

    train_root = root / "train" if (root / "train").is_dir() else root
    source = VideoFolderDataset(
        train_root,
        categories,
        frames=frames,
        stride=stride,
        size=size,
        train=False,
    )
    checkpoint_args = saved_args or {}
    effective_ratio = (
        float(val_ratio)
        if val_ratio is not None
        else float(checkpoint_args.get("val_ratio", 0.2))
    )
    effective_seed = (
        int(seed) if seed is not None else int(checkpoint_args.get("seed", 42))
    )
    _, validation_indices = stratified_split_indices(
        source.samples, effective_ratio, effective_seed
    )
    validation_indices = stratified_limit_indices(
        source.samples,
        validation_indices,
        limit,
        effective_seed + 202,
    )
    dataset = Subset(source, validation_indices)
    print(
        "[eval data] no physical validation directory; "
        f"recreated stratified validation={len(dataset)} "
        f"ratio={effective_ratio:.3f} seed={effective_seed}"
    )
    return dataset


def dataset_sample_path(dataset: Dataset, index: int) -> Path:
    """Return the source path through any nested ``Subset`` wrappers."""

    current: Dataset = dataset
    current_index = index
    while isinstance(current, Subset):
        current_index = int(current.indices[current_index])
        current = current.dataset
    samples = getattr(current, "samples", None)
    if samples is None:
        raise TypeError("evaluation dataset does not expose source samples")
    return Path(samples[current_index][0])


def _prepare_rd_curve(
    rows: Sequence[Mapping[str, Any]], method: str, quality_key: str
) -> tuple[np.ndarray, np.ndarray]:
    points = sorted(
        (
            (float(row["bpp"]), float(row[quality_key]))
            for row in rows
            if row["method"] == method
        ),
        key=lambda point: point[0],
    )
    if not points:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    rates = np.asarray([point[0] for point in points], dtype=np.float64)
    qualities = np.asarray([point[1] for point in points], dtype=np.float64)
    if np.any(rates <= 0) or not np.all(np.isfinite(rates)):
        raise ValueError("BD-rate requires finite positive BPP values")

    # Accuracy measured on a finite validation set can move down by one sample.
    # A monotone envelope removes dominated higher-rate points before fitting.
    qualities = np.maximum.accumulate(qualities)
    quality_to_rate: dict[float, float] = {}
    for quality, rate in zip(qualities, rates, strict=True):
        key = float(quality)
        quality_to_rate[key] = min(quality_to_rate.get(key, math.inf), float(rate))
    unique_qualities = np.asarray(sorted(quality_to_rate), dtype=np.float64)
    unique_rates = np.asarray(
        [quality_to_rate[quality] for quality in unique_qualities], dtype=np.float64
    )
    return unique_qualities, unique_rates


def calculate_bd_rate(
    rows: Sequence[Mapping[str, Any]], quality_key: str
) -> float | None:
    """Return preprocessed-vs-anchor BD-rate percentage over shared quality."""

    anchor_quality, anchor_rate = _prepare_rd_curve(rows, "anchor", quality_key)
    proposed_quality, proposed_rate = _prepare_rd_curve(
        rows, "preprocessed", quality_key
    )
    if len(anchor_quality) < 2 or len(proposed_quality) < 2:
        return None

    quality_min = max(float(anchor_quality.min()), float(proposed_quality.min()))
    quality_max = min(float(anchor_quality.max()), float(proposed_quality.max()))
    if quality_max <= quality_min:
        return None

    def average_log_rate(quality: np.ndarray, rate: np.ndarray) -> float:
        degree = min(3, len(quality) - 1)
        polynomial = np.polyfit(quality, np.log(rate), degree)
        integral = np.polyint(polynomial)
        area = np.polyval(integral, quality_max) - np.polyval(integral, quality_min)
        return float(area / (quality_max - quality_min))

    anchor_average = average_log_rate(anchor_quality, anchor_rate)
    proposed_average = average_log_rate(proposed_quality, proposed_rate)
    return float((math.exp(proposed_average - anchor_average) - 1.0) * 100.0)
