"""Shared checkpoint, metric, and reproducibility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += value * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    predictions = logits.topk(k, dim=1).indices
    return int(predictions.eq(labels[:, None]).any(dim=1).sum().item())


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_run_directory(output_dir: str | Path, resume: str | None) -> None:
    """Prevent stale best checkpoints from making a new, infeasible run look valid."""
    output = Path(output_dir).resolve()
    if resume is not None:
        if Path(resume).resolve().parent != output:
            raise ValueError("--resume must use its original --output-dir; use --init-checkpoint for a new run")
    elif output.is_dir() and any(output.glob("*.pt")):
        raise ValueError(f"{output} already contains checkpoints; choose a new --output-dir or --resume")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
