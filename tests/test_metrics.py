from __future__ import annotations

import numpy as np

from ml_algo.metrics import (
    MetricReport,
    accuracy_score,
    binary_predictions,
    expected_calibration_error,
    f1_score,
    roc_auc_score,
)


def test_accuracy_and_f1() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    report = MetricReport(accuracy=acc, f1=f1, roc_auc=0.7, ece=0.05)
    as_dict = report.as_dict()
    assert as_dict["accuracy"] == acc
    assert as_dict["f1"] == f1


def test_roc_auc_and_ece() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([0.1, 0.8, 0.6, 0.4, 0.9, 0.3])
    auc = roc_auc_score(y_true, scores)
    ece = expected_calibration_error(y_true, scores, n_bins=5)
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= ece <= 1.0


def test_binary_predictions_threshold() -> None:
    probs = np.array([0.2, 0.4, 0.6, 0.9])
    preds = binary_predictions(probs, threshold=0.5)
    assert preds.tolist() == [0, 0, 1, 1]

