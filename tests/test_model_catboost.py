from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

catboost = pytest.importorskip('catboost')  # type: ignore

from ml_algo.model_catboost import (
    CatBoostConfig,
    CatBoostModel,
    IsotonicCalibrator,
)


def test_catboost_model_train_and_predict(tmp_path):
    X = pd.DataFrame({'a': [0, 1, 0, 1], 'b': [1, 1, 0, 0]})
    y = np.array([0, 1, 0, 1])
    config = CatBoostConfig(iterations=10, depth=2, learning_rate=0.1)
    model = CatBoostModel(config)
    model.fit(X, y)
    preds = model.predict_proba(X)
    assert preds.shape == (4, 2)
    calibrator = IsotonicCalibrator().fit(preds[:, 1], y)
    model.set_calibrator(calibrator)
    calibrated = model.predict_proba_calibrated(X)
    assert calibrated.shape == preds.shape
    save_dir = tmp_path / 'model'
    model.save(save_dir)
    restored = CatBoostModel.load(save_dir)
    restored_preds = restored.predict_proba(X)
    assert np.allclose(preds, restored_preds, atol=1e-6)
    restored_calibrated = restored.predict_proba_calibrated(X)
    assert np.allclose(calibrated, restored_calibrated, atol=1e-6)

