from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from ml_algo.kernels import KernelEnsembleParams, compute_kernel_scores, compute_tau_time_diff

if TYPE_CHECKING:
    from ml_algo.pipeline import Phase4Result


@dataclass(frozen=True)
class FeatureAssemblerConfig:
    include_ha: Sequence[str] = ("ha_open", "ha_high", "ha_low", "ha_close", "volume")


def neighborhood_geometry(distances: np.ndarray) -> dict[str, float]:
    return {
        "dL_mean": float(np.mean(distances)),
        "dL_median": float(np.median(distances)),
        "dL_std": float(np.std(distances)),
        "dL_min": float(np.min(distances)),
        "dL_max": float(np.max(distances)),
        "density": float(1.0 / max(np.mean(distances), 1e-6)),
    }


def ann_stats(candidate_dists: np.ndarray, reranked_dists: np.ndarray) -> dict[str, float]:
    return {
        "ann_base_min": float(np.min(candidate_dists)),
        "ann_base_max": float(np.max(candidate_dists)),
        "ann_rerank_min": float(np.min(reranked_dists)),
        "ann_rerank_max": float(np.max(reranked_dists)),
    }


def assemble_features(
    phase4: "Phase4Result",
    labels: pd.Series | np.ndarray,
    kernel_params: KernelEnsembleParams,
    config: FeatureAssemblerConfig | None = None,
) -> pd.DataFrame:
    cfg = config or FeatureAssemblerConfig()
    ha_df = phase4.phase3.ha
    label_array = np.asarray(labels)
    rows: list[dict[str, float]] = []

    for idx in range(len(ha_df)):
        neighbor_ids = phase4.reranked_ids[idx]
        dists = phase4.reranked_dists[idx]
        targets = label_array[neighbor_ids]
        tau = compute_tau_time_diff(idx, neighbor_ids)
        kernel_scores = compute_kernel_scores(targets, dists, tau, kernel_params)
        row = {}
        row.update(neighborhood_geometry(dists))
        row.update(ann_stats(phase4.candidate_dists[idx], dists))
        row.update(kernel_scores)
        for col in cfg.include_ha:
            row[col] = float(ha_df.iloc[idx][col])
        rows.append(row)
    feature_df = pd.DataFrame(rows, index=ha_df.index)
    return feature_df
