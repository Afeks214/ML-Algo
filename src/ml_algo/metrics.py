from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_binary(labels: np.ndarray) -> None:
    unique = np.unique(labels)
    if not np.array_equal(unique, [0]) and not np.array_equal(unique, [1]) and not np.array_equal(unique, [0, 1]):
        raise ValueError("Input labels must be binary (0/1).")


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return float(np.mean(y_true_arr == y_pred_arr))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("Shapes of y_true and y_score must match.")
    _validate_binary(y_true_arr)
    pos = np.sum(y_true_arr == 1)
    neg = np.sum(y_true_arr == 0)
    if pos == 0 or neg == 0:
        raise ValueError("Both positive and negative labels are required to compute ROC-AUC.")
    order = np.argsort(-y_score_arr)
    y_sorted = y_true_arr[order]
    tp_cumsum = np.cumsum(y_sorted == 1)
    fp_cumsum = np.cumsum(y_sorted == 0)
    tpr = tp_cumsum / pos
    fpr = fp_cumsum / neg
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return float(np.trapz(tpr, fpr))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("Shapes of y_true and y_prob must match.")
    _validate_binary(y_true_arr)
    if not np.all((y_prob_arr >= 0.0) & (y_prob_arr <= 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if n_bins <= 1:
        raise ValueError("n_bins must be greater than 1.")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = y_true_arr.shape[0]
    ece = 0.0
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob_arr >= left) & (y_prob_arr <= right)
        else:
            mask = (y_prob_arr >= left) & (y_prob_arr < right)
        count = np.count_nonzero(mask)
        if count == 0:
            continue
        bin_acc = float(np.mean(y_true_arr[mask]))
        bin_conf = float(np.mean(y_prob_arr[mask]))
        ece += (count / total) * abs(bin_acc - bin_conf)
    return float(ece)


def binary_predictions(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")
    y_prob_arr = np.asarray(y_prob, dtype=float)
    return (y_prob_arr >= threshold).astype(int)


@dataclass(frozen=True)
class MetricReport:
    accuracy: float
    f1: float
    roc_auc: float
    ece: float
    threshold: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "ece": self.ece,
            "threshold": self.threshold,
        }

