from __future__ import annotations

import numpy as np

from ml_algo.kernels import (
    KernelEnsembleParams,
    LocallyPeriodicParams,
    PeriodicParams,
    RationalQuadraticParams,
    compute_kernel_scores,
    compute_tau_time_diff,
)


def test_kernel_scores_produce_expected_keys() -> None:
    targets = np.array([1.0, 0.0, 1.0, 0.0])
    distances = np.array([0.1, 0.2, 0.5, 0.8])
    taus = np.array([0.0, 1.0, 2.0, 3.0])
    params = KernelEnsembleParams(
        rq=RationalQuadraticParams(alpha=1.2, ell=0.5),
        periodic=PeriodicParams(period=10.0, ell=0.8),
        locally_periodic=LocallyPeriodicParams(ell_se=0.3, period=10.0, ell_per=0.9),
        mkl_weights=(0.3, 0.3, 0.4),
        weight_floor=1e-9,
    )
    scores = compute_kernel_scores(targets, distances, taus, params)
    expected_keys = {
        "score_rq",
        "weightsum_rq",
        "eff_k_rq",
        "score_periodic",
        "weightsum_periodic",
        "eff_k_periodic",
        "score_locally_periodic",
        "weightsum_locally_periodic",
        "eff_k_locally_periodic",
        "score_mkl",
    }
    assert expected_keys.issubset(scores.keys())
    assert scores["weightsum_rq"] >= 1e-9
    assert scores["eff_k_rq"] >= 1


def test_compute_tau_time_diff_matches_index_distance() -> None:
    neighbors = [2, 5, 7]
    tau = compute_tau_time_diff(3, neighbors)
    assert np.allclose(tau, np.array([1.0, 2.0, 4.0]))
