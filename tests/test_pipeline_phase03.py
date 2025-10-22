from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from ml_algo.data_ingest import GapPolicy
from ml_algo.pipeline import FEATURE_COLUMNS_PHASE3, Phase3Result, run_phase3
from ml_algo.robust_scaling import TylerConfig


def write_sample_csv(tmp_path: Path, nrows: int = 1500) -> Path:
    df = pd.read_csv("data/raw/@NQ - 5 min - ETH.csv", nrows=nrows)
    df = df.rename(
        columns={
            "Date": "ts",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_run_phase3_returns_whitened_features(tmp_path: Path) -> None:
    src = write_sample_csv(tmp_path)
    result = run_phase3(
        sources=[src],
        timezone="America/New_York",
        bar_sizes=["5min"],
        gap_policy=GapPolicy(max_gap_minutes=45),
        tyler_config=TylerConfig(rho=0.2, tol=1e-6, max_iter=400),
    )
    assert isinstance(result, Phase3Result)
    assert set(FEATURE_COLUMNS_PHASE3).issubset(result.features.columns)
    assert result.whitened.shape[1] == len(FEATURE_COLUMNS_PHASE3)
    assert result.tyler.converged

    assert np.isfinite(result.whitened_array).all()
    scatter = (result.whitened_array.T @ result.whitened_array) / result.whitened_array.shape[0]
    scatter_norm = scatter * (
        result.whitened_array.shape[1] / np.trace(scatter)
    )
    deviation = np.linalg.norm(scatter_norm - np.eye(scatter.shape[0]), ord="fro")
    assert deviation < 5.0
