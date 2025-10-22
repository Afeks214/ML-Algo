from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml_algo.ann_index import AnnConfig
from ml_algo.data_ingest import GapPolicy
from ml_algo.kernels import KernelEnsembleParams
from ml_algo.pipeline import run_phase3, run_phase4, run_phase5, run_phase6
from ml_algo.robust_scaling import TylerConfig
from ml_algo.model_catboost import CatBoostConfig


def make_labels_from_ha_close(ha_close: np.ndarray) -> pd.Series:
    # Binary label: 1 if next HA close > current, else 0
    shifted = np.roll(ha_close, -1)
    y = (shifted > ha_close).astype(int)
    # Last row has no next value; set to 0
    y[-1] = 0
    return pd.Series(y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local training on a sample of the dataset")
    parser.add_argument("--source", type=str, default="data/raw/@NQ - 5 min - ETH.csv", help="Path to OHLCV CSV")
    parser.add_argument("--nrows", type=int, default=2000, help="Number of rows to sample from CSV")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to save model and metrics")
    parser.add_argument("--iterations", type=int, default=300, help="CatBoost iterations for quick local run")
    args = parser.parse_args()

    # Prepare a small sample CSV to avoid loading entire file for quick validation runs
    sample_path = Path("data/tmp/local_sample.csv")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    df_sample = pd.read_csv(args.source, nrows=args.nrows)
    df_sample = df_sample.rename(
        columns={
            "Date": "ts",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df_sample.to_csv(sample_path, index=False)

    # Phase 3: ingest -> HA -> Tyler
    phase3 = run_phase3(
        sources=[sample_path],
        timezone="America/New_York",
        bar_sizes=["5min"],
        gap_policy=GapPolicy(max_gap_minutes=45, fill_volume=False, forward_fill=False),
        tyler_config=TylerConfig(rho=0.2, tol=1e-6, max_iter=400, center=True),
    )

    # Phase 4: ANN exact backend + hyperbolic rerank
    ann_cfg = AnnConfig(backend="exact", k_cand=16, ef_search=16, ef_search_max=64, nprobe=4, nprobe_max=16)
    phase4 = run_phase4(phase3, ann_cfg, k_final=8)

    # Labels: next-bar HA close up/down
    ha_close = phase3.ha["ha_close"].to_numpy()
    labels = make_labels_from_ha_close(ha_close)
    labels.index = phase3.ha.index

    # Phase 5: feature assembly
    kernel_params = KernelEnsembleParams()  # defaults per SPEC/model.yaml
    phase5 = run_phase5(phase4, labels=labels, kernel_params=kernel_params, train_model=False)

    # Phase 6: CatBoost training + calibration + artifact export
    ts = int(time.time())
    out_dir = Path(args.artifact_dir) if args.artifact_dir else Path(f"artifacts/local_run_{ts}")
    cb_cfg = CatBoostConfig(
        iterations=int(args.iterations),
        depth=6,
        learning_rate=0.05,
        task_type="CPU",
        random_seed=42,
        allow_writing_files=False,
    )
    phase6 = run_phase6(
        phase5,
        catboost_config=cb_cfg,
        eval_fraction=0.2,
        calibration_method="isotonic",
        artifact_dir=str(out_dir),
    )

    summary = {
        "artifact_dir": str(out_dir),
        "metrics": phase6.metrics.as_dict(),
        "ann_recall": float(phase4.recall),
        "timings_ms": phase6.timings_ms,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

