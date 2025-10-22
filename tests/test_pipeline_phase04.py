from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_algo.ann_index import AnnConfig
from ml_algo.data_ingest import GapPolicy
from ml_algo.pipeline import Phase4Result, run_phase3, run_phase4
from ml_algo.robust_scaling import TylerConfig


def write_sample_csv(tmp_path: Path, nrows: int = 400) -> Path:
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
    path = tmp_path / "phase4_sample.csv"
    df.to_csv(path, index=False)
    return path


def test_run_phase4_reranks_candidates(tmp_path: Path) -> None:
    src = write_sample_csv(tmp_path)
    phase3 = run_phase3(
        sources=[src],
        timezone="America/New_York",
        bar_sizes=["5min"],
        gap_policy=GapPolicy(max_gap_minutes=45),
        tyler_config=TylerConfig(rho=0.2, tol=1e-6, max_iter=400),
    )
    ann_cfg = AnnConfig(k_cand=16, ef_search=16, ef_search_max=64, nprobe=4, nprobe_max=16)
    result = run_phase4(phase3, ann_cfg, k_final=8)
    assert isinstance(result, Phase4Result)
    assert result.reranked_ids.shape == (phase3.whitened.shape[0], 8)
    assert np.isfinite(result.reranked_dists).all()
    assert result.recall >= 0.9
    assert not result.tuner.param_bumps  # exact backend hits target recall
    assert "ann_search" in result.timings_ms
