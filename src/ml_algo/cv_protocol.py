from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    id: int
    train_indices: np.ndarray
    val_indices: np.ndarray


def split_purged_embargo(
    timestamps: Sequence[pd.Timestamp],
    *,
    purge: int,
    embargo: int,
    k: int,
) -> List[Fold]:
    if k < 2:
        raise ValueError("k must be at least 2")
    n = len(timestamps)
    if n == 0:
        raise ValueError("timestamps cannot be empty")
    indices = np.arange(n)
    fold_size = n // k
    folds: List[Fold] = []
    for fold_id in range(k):
        start = fold_id * fold_size
        end = n if fold_id == k - 1 else (fold_id + 1) * fold_size
        val_idx = indices[start:end]
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        purge_start = max(0, start - purge)
        train_mask[purge_start:start] = False
        embargo_end = min(n, end + embargo)
        train_mask[end:embargo_end] = False
        train_idx = indices[train_mask]
        folds.append(Fold(fold_id, train_idx, val_idx))
    return folds


def _fold_boundaries(n: int, k: int) -> List[tuple[int, int]]:
    base, remainder = divmod(n, k)
    sizes = [base + (1 if i < remainder else 0) for i in range(k)]
    bounds: List[tuple[int, int]] = []
    start = 0
    for size in sizes:
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds


def split_cpcv(
    timestamps: Sequence[pd.Timestamp],
    *,
    purge: int,
    embargo: int,
    k: int,
    h: int,
) -> List[Fold]:
    """
    Generate Cross-Validated Purged Combinatorial folds (CPCV).

    Args:
        timestamps: ordered timestamps aligned with samples.
        purge: number of samples to purge before each validation block.
        embargo: number of samples to embargo after each validation block.
        k: number of base contiguous folds.
        h: number of folds held out for validation per combination.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    if not 0 < h < k:
        raise ValueError("h must be in (0, k)")
    n = len(timestamps)
    if n == 0:
        raise ValueError("timestamps cannot be empty")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")
    indices = np.arange(n)
    boundaries = _fold_boundaries(n, k)
    folds: List[Fold] = []
    fold_id = 0
    for combo in combinations(range(k), h):
        train_mask = np.ones(n, dtype=bool)
        val_indices_parts: List[np.ndarray] = []
        for base_idx in combo:
            start, end = boundaries[base_idx]
            val_slice = indices[start:end]
            val_indices_parts.append(val_slice)
            train_mask[start:end] = False
            purge_start = max(0, start - purge)
            train_mask[purge_start:start] = False
            embargo_end = min(n, end + embargo)
            train_mask[end:embargo_end] = False
        train_indices = indices[train_mask]
        if train_indices.size == 0:
            raise ValueError("Configuration leaves no training samples; adjust purge/embargo or fold counts.")
        val_indices = np.concatenate(val_indices_parts)
        folds.append(Fold(fold_id, train_indices, val_indices))
        fold_id += 1
    return folds


def no_information_overlap(folds: Iterable[Fold]) -> bool:
    for fold in folds:
        if np.intersect1d(fold.train_indices, fold.val_indices).size > 0:
            return False
    return True


def fold_artifact_path(base_dir: Path | str, fold: Fold, artifact_name: str) -> Path:
    base = Path(base_dir) / f"fold_{fold.id}"
    return base / artifact_name
