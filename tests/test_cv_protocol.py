from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from ml_algo.cv_protocol import (
    fold_artifact_path,
    no_information_overlap,
    split_cpcv,
    split_purged_embargo,
)


def test_split_purged_embargo_no_overlap() -> None:
    timestamps = pd.date_range("2024-01-01", periods=100, freq="min")
    folds = split_purged_embargo(timestamps, purge=2, embargo=2, k=5)
    assert len(folds) == 5
    assert no_information_overlap(folds)
    for fold in folds:
        assert len(fold.val_indices) > 0
        assert np.all(fold.val_indices < 100)


def test_fold_artifact_path_includes_fold_id(tmp_path: Path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=20, freq="min")
    fold = split_purged_embargo(timestamps, purge=1, embargo=1, k=4)[0]
    path = fold_artifact_path(tmp_path, fold, "index.bin")
    assert f"fold_{fold.id}" in str(path)


def test_split_cpcv_combinations() -> None:
    timestamps = pd.date_range("2024-01-01", periods=60, freq="min")
    folds = split_cpcv(timestamps, purge=2, embargo=2, k=4, h=1)
    # Expect one fold per base segment
    assert len(folds) == 4
    assert all(no_information_overlap([fold]) for fold in folds)
    folds_h2 = split_cpcv(timestamps, purge=2, embargo=2, k=4, h=2)
    # C(4,2) = 6
    assert len(folds_h2) == 6
    for fold in folds_h2:
        assert no_information_overlap([fold])
