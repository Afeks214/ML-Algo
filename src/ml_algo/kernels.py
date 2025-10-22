from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RationalQuadraticParams:
    alpha: float = 1.0
    ell: float = 1.0


@dataclass(frozen=True)
class PeriodicParams:
    period: float = 390.0  # trading day minutes
    ell: float = 1.0


@dataclass(frozen=True)
class LocallyPeriodicParams:
    ell_se: float = 1.0
    period: float = 390.0
    ell_per: float = 1.0


@dataclass(frozen=True)
class KernelEnsembleParams:
    rq: RationalQuadraticParams = RationalQuadraticParams()
    periodic: PeriodicParams = PeriodicParams()
    locally_periodic: LocallyPeriodicParams = LocallyPeriodicParams()
    mkl_weights: tuple[float, float, float] | None = None
    weight_floor: float = 1e-12


def rq(r: FloatArray, params: RationalQuadraticParams) -> FloatArray:
    r2 = np.square(r)
    denom = 2.0 * params.alpha * params.ell * params.ell
    return np.power(1.0 + r2 / denom, -params.alpha)


@lru_cache(maxsize=2048)
def _sin_cache(key: tuple[float, float]) -> float:
    period, tau = key
    return math.sin(math.pi * tau / period)


def periodic(tau: FloatArray, params: PeriodicParams) -> FloatArray:
    tau_arr = np.asarray(tau, dtype=np.float64)
    sin_term = np.sin(math.pi * tau_arr / params.period)
    return np.exp(-2.0 * np.square(sin_term) / (params.ell * params.ell))


def locally_periodic(
    r: FloatArray,
    tau: FloatArray,
    params: LocallyPeriodicParams,
) -> FloatArray:
    se = np.exp(-(np.square(r)) / (2.0 * params.ell_se * params.ell_se))
    per = periodic(tau, PeriodicParams(period=params.period, ell=params.ell_per))
    return se * per


def nadaraya_watson_score(
    targets: FloatArray,
    weights: FloatArray,
    *,
    weight_floor: float = 1e-12,
) -> dict[str, float]:
    w = np.clip(weights, 0.0, None)
    weightsum = float(np.maximum(np.sum(w), weight_floor))
    eff_k = int(np.count_nonzero(w > 0.0))
    score = float(np.dot(w, targets) / weightsum)
    return {"score": score, "weightsum": weightsum, "eff_k": max(eff_k, 1)}


def compute_kernel_scores(
    targets: FloatArray,
    distances: FloatArray,
    taus: FloatArray,
    params: KernelEnsembleParams,
) -> dict[str, float]:
    results: dict[str, float] = {}

    rq_weights = rq(distances, params.rq)
    rq_stats = nadaraya_watson_score(targets, rq_weights, weight_floor=params.weight_floor)
    results.update(
        {
            "score_rq": rq_stats["score"],
            "weightsum_rq": rq_stats["weightsum"],
            "eff_k_rq": float(rq_stats["eff_k"]),
        }
    )

    per_weights = periodic(taus, params.periodic)
    per_stats = nadaraya_watson_score(targets, per_weights, weight_floor=params.weight_floor)
    results.update(
        {
            "score_periodic": per_stats["score"],
            "weightsum_periodic": per_stats["weightsum"],
            "eff_k_periodic": float(per_stats["eff_k"]),
        }
    )

    lp_weights = locally_periodic(distances, taus, params.locally_periodic)
    lp_stats = nadaraya_watson_score(targets, lp_weights, weight_floor=params.weight_floor)
    results.update(
        {
            "score_locally_periodic": lp_stats["score"],
            "weightsum_locally_periodic": lp_stats["weightsum"],
            "eff_k_locally_periodic": float(lp_stats["eff_k"]),
        }
    )

    if params.mkl_weights:
        lam = np.array(params.mkl_weights, dtype=np.float64)
        lam = lam / np.maximum(np.sum(lam), 1e-12)
        kernel_scores = np.array(
            [rq_stats["score"], per_stats["score"], lp_stats["score"]],
            dtype=np.float64,
        )
        mkl_score = float(np.dot(lam, kernel_scores))
        results["score_mkl"] = mkl_score
    return results


def compute_tau_time_diff(
    query_index: int,
    neighbor_indices: Iterable[int],
) -> FloatArray:
    neighbor_idx = np.asarray(list(neighbor_indices), dtype=np.int64)
    tau = np.abs(neighbor_idx - query_index).astype(np.float64)
    return tau
