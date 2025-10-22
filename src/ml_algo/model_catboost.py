from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier, Pool  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None  # type: ignore
    Pool = None  # type: ignore


@dataclass
class CatBoostConfig:
    depth: int = 6
    learning_rate: float = 0.05
    iterations: int = 1000
    l2_leaf_reg: float = 3.0
    subsample: float = 0.8
    loss_function: str = "Logloss"
    random_seed: int = 42
    early_stopping_rounds: int | None = 50
    eval_fraction: float = 0.2
    class_weights: Sequence[float] | None = None
    task_type: str = "CPU"
    devices: str | None = None
    boosting_type: str = "Ordered"
    bootstrap_type: str | None = None
    grow_policy: str | None = None
    od_type: str | None = "Iter"
    allow_writing_files: bool = False


class IsotonicCalibrator:
    """
    Simple isotonic regression calibrator implemented via Pool-Adjacent Violators Algorithm.

    Works for binary classification probabilities in [0, 1].
    """

    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    @staticmethod
    def _validate_inputs(probs: np.ndarray, labels: np.ndarray) -> None:
        if probs.ndim != 1:
            raise ValueError("probs must be 1D array")
        if labels.ndim != 1:
            raise ValueError("labels must be 1D array")
        if probs.shape[0] != labels.shape[0]:
            raise ValueError("probs and labels must have the same length")
        if not np.all((probs >= 0.0) & (probs <= 1.0)):
            raise ValueError("probabilities must lie in [0, 1]")
        unique = np.unique(labels)
        if not np.array_equal(unique, [0]) and not np.array_equal(unique, [1]) and not np.array_equal(unique, [0, 1]):
            raise ValueError("labels must be binary (0/1) for isotonic calibration")

    @staticmethod
    def _pava(labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Pool-Adjacent Violators Algorithm with block tracking."""
        blocks: list[tuple[float, float, int]] = []  # (weight_sum, value_sum, length)
        for label, weight in zip(labels, weights, strict=True):
            blocks.append((weight, weight * label, 1))
            while len(blocks) >= 2:
                w0, s0, l0 = blocks[-2]
                w1, s1, l1 = blocks[-1]
                if s0 / w0 <= s1 / w1:
                    break
                merged = (w0 + w1, s0 + s1, l0 + l1)
                blocks.pop()
                blocks[-1] = merged
        calibrated = np.empty(labels.shape[0], dtype=np.float64)
        start = 0
        for weight_sum, value_sum, length in blocks:
            avg = value_sum / weight_sum
            calibrated[start : start + length] = avg
            start += length
        return calibrated

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        sample_weight: Iterable[float] | None = None,
    ) -> "IsotonicCalibrator":
        probs_arr = np.asarray(probs, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.float64)
        self._validate_inputs(probs_arr, labels_arr)
        weights = np.ones_like(labels_arr) if sample_weight is None else np.asarray(list(sample_weight), dtype=np.float64)
        if weights.shape != labels_arr.shape:
            raise ValueError("sample_weight must match labels shape")
        order = np.argsort(probs_arr)
        sorted_probs = probs_arr[order]
        sorted_labels = labels_arr[order]
        sorted_weights = weights[order]
        calibrated_values = self._pava(sorted_labels, sorted_weights)
        self._x = sorted_probs
        self._y = calibrated_values
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self._x is None or self._y is None:
            raise RuntimeError("Calibrator is not fitted")
        probs_arr = np.asarray(probs, dtype=np.float64)
        return np.interp(
            probs_arr,
            self._x,
            self._y,
            left=float(self._y[0]),
            right=float(self._y[-1]),
        )

    def to_dict(self) -> dict[str, list[float] | str]:
        if self._x is None or self._y is None:
            raise RuntimeError("Calibrator is not fitted")
        return {
            "method": "isotonic",
            "x": self._x.tolist(),
            "y": self._y.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[float] | str]) -> "IsotonicCalibrator":
        calibrator = cls()
        x = np.asarray(payload["x"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)
        calibrator._x = x
        calibrator._y = y
        return calibrator


class CatBoostModel:
    def __init__(self, config: CatBoostConfig) -> None:
        if CatBoostClassifier is None:  # pragma: no cover - runtime check
            raise ImportError(
                "catboost is required to instantiate CatBoostModel. "
                "Install catboost or run with train_model=False."
            )
        if config.task_type.upper() == "GPU":
            os.environ.setdefault("GPU_DETERMINISTIC", "1")
        self.config = config
        params: dict[str, object] = {
            "depth": config.depth,
            "learning_rate": config.learning_rate,
            "iterations": config.iterations,
            "l2_leaf_reg": config.l2_leaf_reg,
            "subsample": config.subsample,
            "loss_function": config.loss_function,
            "random_seed": config.random_seed,
            "task_type": config.task_type,
            "boosting_type": config.boosting_type,
            "allow_writing_files": config.allow_writing_files,
            "verbose": False,
        }
        if config.class_weights is not None:
            params["class_weights"] = config.class_weights
        if config.devices is not None:
            params["devices"] = config.devices
        if config.bootstrap_type is not None:
            params["bootstrap_type"] = config.bootstrap_type
        if config.grow_policy is not None:
            params["grow_policy"] = config.grow_policy
        if config.od_type is not None:
            params["od_type"] = config.od_type

        self.model = CatBoostClassifier(**params)
        self.calibrator: IsotonicCalibrator | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[int] | np.ndarray,
        eval_set: tuple[pd.DataFrame | np.ndarray, Sequence[int]] | None = None,
        sample_weight: Sequence[float] | None = None,
    ) -> None:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y)
        pool = Pool(X_arr, label=y_arr)
        if eval_set is not None:
            eval_X, eval_y = eval_set
            eval_pool = Pool(
                eval_X.values if isinstance(eval_X, pd.DataFrame) else eval_X,
                label=np.asarray(eval_y),
            )
        else:
            eval_pool = None
        sample_weight_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self.model.fit(
            pool,
            eval_set=eval_pool,
            sample_weight=sample_weight_arr,
            early_stopping_rounds=self.config.early_stopping_rounds,
        )

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_arr)

    def predict_proba_calibrated(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        if probs.ndim != 2 or probs.shape[1] < 2:
            return probs
        if self.calibrator is None:
            return probs
        calibrated_pos = self.calibrator.transform(probs[:, 1])
        calibrated_pos = np.clip(calibrated_pos, 0.0, 1.0)
        calibrated = probs.copy()
        calibrated[:, 1] = calibrated_pos
        if calibrated.shape[1] == 2:
            calibrated[:, 0] = 1.0 - calibrated_pos
        return calibrated

    def set_calibrator(self, calibrator: IsotonicCalibrator | None) -> None:
        self.calibrator = calibrator

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "catboost.cbm"
        config_path = directory / "config.json"
        self.model.save_model(str(model_path))
        config_path.write_text(json.dumps(asdict(self.config)))
        if self.calibrator is not None:
            calibrator_path = directory / "calibrator.json"
            calibrator_path.write_text(json.dumps(self.calibrator.to_dict()))

    @classmethod
    def load(cls, directory: Path | str) -> "CatBoostModel":
        if CatBoostClassifier is None:  # pragma: no cover
            raise ImportError("catboost must be installed to load CatBoostModel")
        directory = Path(directory)
        config_path = directory / "config.json"
        model_path = directory / "catboost.cbm"
        config = CatBoostConfig(**json.loads(config_path.read_text()))
        instance = cls(config)
        instance.model.load_model(str(model_path))
        calibrator_path = directory / "calibrator.json"
        if calibrator_path.exists():
            payload = json.loads(calibrator_path.read_text())
            if payload.get("method") == "isotonic":
                instance.calibrator = IsotonicCalibrator.from_dict(payload)
        return instance
