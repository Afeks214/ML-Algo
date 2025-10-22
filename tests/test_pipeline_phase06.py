from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_algo.ann_index import AnnConfig
from ml_algo.data_ingest import GapPolicy
from ml_algo.kernels import KernelEnsembleParams
from ml_algo.model_catboost import CatBoostConfig
from ml_algo.pipeline import Phase6Result, run_phase3, run_phase4, run_phase5, run_phase6
from ml_algo.robust_scaling import TylerConfig


def write_sample_csv(tmp_path: Path, nrows: int = 600) -> Path:
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
    path = tmp_path / "phase6_sample.csv"
    df.to_csv(path, index=False)
    return path


def test_run_phase6_trains_and_exports(tmp_path: Path) -> None:
    src = write_sample_csv(tmp_path)
    phase3 = run_phase3(
        sources=[src],
        timezone="America/New_York",
        bar_sizes=["5min"],
        gap_policy=GapPolicy(max_gap_minutes=45),
        tyler_config=TylerConfig(rho=0.2, tol=1e-6, max_iter=400),
    )
    phase4 = run_phase4(
        phase3,
        ann_config=AnnConfig(k_cand=16, ef_search=16, ef_search_max=64, nprobe=4, nprobe_max=16),
        k_final=8,
    )
    ha_close = phase3.ha["ha_close"].to_numpy()
    labels = pd.Series((np.roll(ha_close, -1) > ha_close).astype(int)[:-1], index=phase3.ha.index[:-1])
    labels = labels.reindex(phase3.ha.index, fill_value=0)
    kernel_params = KernelEnsembleParams()
    phase5 = run_phase5(
        phase4,
        labels=labels,
        kernel_params=kernel_params,
        train_model=False,
    )
    artifact_dir = tmp_path / "artifacts"
    result = run_phase6(
        phase5,
        catboost_config=CatBoostConfig(iterations=50, depth=3, learning_rate=0.1, eval_fraction=0.2),
        artifact_dir=artifact_dir,
    )
    assert isinstance(result, Phase6Result)
    assert result.model.calibrator is not None
    assert artifact_dir.exists()
    assert (artifact_dir / "catboost.cbm").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert "train_catboost" in (result.timings_ms or {})
    assert result.metrics.accuracy >= 0.0
    assert result.metrics.roc_auc >= 0.0
