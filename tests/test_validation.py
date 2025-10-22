from __future__ import annotations

import json
import numpy as np
import pandas as pd

from ml_algo.cv_protocol import split_cpcv
from ml_algo.model_catboost import CatBoostConfig
from ml_algo.validation import cross_validate_catboost


def test_cross_validate_catboost_reports_metrics() -> None:
    rng = np.random.default_rng(123)
    n_samples = 120
    features = pd.DataFrame(
        rng.normal(size=(n_samples, 4)),
        columns=["f1", "f2", "f3", "f4"],
    )
    logits = 0.8 * features["f1"] - 0.6 * features["f2"]
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = pd.Series((probs > 0.5).astype(int))
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="min")
    folds = split_cpcv(timestamps, purge=1, embargo=1, k=4, h=1)
    report = cross_validate_catboost(
        features,
        labels,
        folds,
        config=CatBoostConfig(iterations=30, depth=3, learning_rate=0.1),
    )
    summary = report.aggregate()
    assert "accuracy_mean" in summary
    assert len(report.fold_results) == len(folds)


def test_validation_report_serialization(tmp_path) -> None:
    rng = np.random.default_rng(321)
    features = pd.DataFrame(rng.normal(size=(40, 3)), columns=["f1", "f2", "f3"])
    labels = pd.Series(rng.integers(0, 2, size=40))
    timestamps = pd.date_range("2024-02-01", periods=40, freq="3min")
    folds = split_cpcv(timestamps, purge=1, embargo=1, k=2, h=1)
    report = cross_validate_catboost(
        features,
        labels,
        folds,
        config=CatBoostConfig(iterations=10, depth=2, learning_rate=0.2),
    )
    path = tmp_path / "report.json"
    report.to_json(path)
    payload = json.loads(path.read_text())
    assert "folds" in payload and payload["folds"]
    assert "aggregate" in payload
