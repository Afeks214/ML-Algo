from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from ml_algo.calibration import CalibrationResult
from ml_algo.cv_protocol import Fold
from ml_algo.metrics import (
    MetricReport,
    accuracy_score,
    binary_predictions,
    expected_calibration_error,
    f1_score,
    roc_auc_score,
)
from ml_algo.model_catboost import CatBoostConfig, CatBoostModel, IsotonicCalibrator


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> MetricReport:
    y_pred = binary_predictions(y_prob, threshold=threshold)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
    try:
        ece = expected_calibration_error(y_true, y_prob)
    except ValueError:
        ece = float("nan")
    return MetricReport(
        accuracy=accuracy,
        f1=f1,
        roc_auc=roc_auc,
        ece=ece,
        threshold=threshold,
    )


@dataclass(frozen=True)
class FoldEvaluation:
    fold_id: int
    metrics: MetricReport
    calibration: CalibrationResult | None
    val_indices: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "metrics": self.metrics.as_dict(),
            "calibration": self.calibration.as_dict() if self.calibration else None,
            "val_indices": self.val_indices.tolist(),
        }


@dataclass(frozen=True)
class ValidationReport:
    fold_results: List[FoldEvaluation]

    def aggregate(self) -> dict[str, float]:
        if not self.fold_results:
            return {}
        keys = ["accuracy", "f1", "roc_auc", "ece"]
        summary: dict[str, float] = {}
        for key in keys:
            values = np.array(
                [
                    getattr(result.metrics, key)
                    for result in self.fold_results
                    if getattr(result.metrics, key) == getattr(result.metrics, key)  # filter NaN by equality check
                ],
                dtype=float,
            )
            if values.size == 0:
                summary[f"{key}_mean"] = float("nan")
                summary[f"{key}_std"] = float("nan")
            else:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        return summary

    def to_dict(self) -> dict[str, object]:
        return {
            "folds": [result.as_dict() for result in self.fold_results],
            "aggregate": self.aggregate(),
        }

    def to_json(self, path: Path, *, indent: int = 2) -> None:
        payload = self.to_dict()
        path = Path(path)
        path.write_text(json.dumps(payload, indent=indent))


def evaluate_fold(
    features: pd.DataFrame,
    labels: pd.Series,
    fold: Fold,
    config: CatBoostConfig,
    *,
    calibration_method: str = "isotonic",
) -> FoldEvaluation:
    train_X = features.iloc[fold.train_indices]
    train_y = labels.iloc[fold.train_indices]
    val_X = features.iloc[fold.val_indices]
    val_y = labels.iloc[fold.val_indices]

    model = CatBoostModel(config)
    model.fit(train_X, train_y, eval_set=(val_X, val_y))
    val_probs = model.predict_proba(val_X)[:, 1]
    metrics = _compute_metrics(val_y.to_numpy(), val_probs)

    calibration: CalibrationResult | None = None
    if calibration_method == "isotonic":
        try:
            calibrator = IsotonicCalibrator().fit(val_probs, val_y.to_numpy())
            calibrated_probs = calibrator.transform(val_probs)
            calibrated_ece = expected_calibration_error(val_y.to_numpy(), calibrated_probs)
            calibration = CalibrationResult(
                method="isotonic",
                before_ece=metrics.ece,
                after_ece=calibrated_ece,
            )
        except ValueError:
            calibration = CalibrationResult(
                method="isotonic",
                before_ece=metrics.ece,
                after_ece=None,
            )
    elif calibration_method == "none":
        calibration = None
    else:
        raise ValueError(f"Unsupported calibration_method: {calibration_method}")

    return FoldEvaluation(
        fold_id=fold.id,
        metrics=metrics,
        calibration=calibration,
        val_indices=fold.val_indices,
    )


def cross_validate_catboost(
    features: pd.DataFrame,
    labels: pd.Series,
    folds: Sequence[Fold],
    config: CatBoostConfig | None = None,
    *,
    calibration_method: str = "isotonic",
) -> ValidationReport:
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if not folds:
        raise ValueError("folds cannot be empty")
    cfg = config or CatBoostConfig()
    results: List[FoldEvaluation] = []
    for fold in folds:
        results.append(
            evaluate_fold(
                features,
                labels,
                fold,
                cfg,
                calibration_method=calibration_method,
            )
        )
    return ValidationReport(results)
