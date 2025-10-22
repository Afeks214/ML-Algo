from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TylerConfig:
    """Configuration parameters for Regularized Tyler whitening."""

    rho: float = 0.1
    tol: float = 1e-6
    max_iter: int = 500
    center: bool = True
    spatial_median_tol: float = 1e-6
    spatial_median_max_iter: int = 500


@dataclass(frozen=True)
class TylerArtifacts:
    """Result of fitting the Regularized Tyler estimator."""

    shape: FloatArray
    location: FloatArray
    whitener: FloatArray
    iterations: int
    converged: bool


def fit(
    X: FloatArray,
    *,
    config: TylerConfig | None = None,
) -> TylerArtifacts:
    """
    Fit the Regularized Tyler scatter estimator.

    Args:
        X: Input observations (n_samples, n_features).
        config: Optional Tyler configuration.

    Returns:
        TylerArtifacts containing the scatter matrix, location, whitener, and diagnostics.
    """
    if X.ndim != 2:
        raise ValueError("Input X must be 2D array (n_samples, n_features)")
    n_samples, n_features = X.shape
    if n_samples == 0:
        raise ValueError("Cannot fit Tyler estimator on empty input")
    cfg = config or TylerConfig()
    if not (0.0 < cfg.rho <= 1.0):
        raise ValueError("rho must lie in (0, 1]")
    if cfg.tol <= 0.0:
        raise ValueError("tol must be positive")
    if cfg.max_iter <= 0:
        raise ValueError("max_iter must be positive")

    if cfg.center:
        location = spatial_median(
            X,
            tol=cfg.spatial_median_tol,
            max_iter=cfg.spatial_median_max_iter,
        )
    else:
        location = np.zeros(n_features, dtype=np.float64)

    Z = X.astype(np.float64, copy=False) - location
    Sigma = np.eye(n_features, dtype=np.float64)

    iterations = 0
    converged = False
    for iterations in range(1, cfg.max_iter + 1):
        Sigma_next = tyler_mm_iter(Sigma, Z, cfg.rho)
        delta = np.linalg.norm(Sigma_next - Sigma, ord="fro")
        Sigma = Sigma_next
        if delta <= cfg.tol:
            converged = True
            break

    Sigma = ensure_pd(Sigma)
    whitener = inv_sqrt(Sigma)
    return TylerArtifacts(
        shape=Sigma,
        location=location,
        whitener=whitener,
        iterations=iterations,
        converged=converged,
    )


def transform(X: FloatArray, artifacts: TylerArtifacts) -> FloatArray:
    """
    Apply whitening transform using fitted Tyler artifacts.

    Args:
        X: Observations (n_samples, n_features).
        artifacts: Result from `fit`.

    Returns:
        Whitened observations (n_samples, n_features).
    """
    if X.ndim != 2:
        raise ValueError("Input X must be 2D array")
    centered = X.astype(np.float64, copy=False) - artifacts.location
    return centered @ artifacts.whitener.T


def fit_transform(
    X: FloatArray,
    *,
    config: TylerConfig | None = None,
) -> Tuple[FloatArray, TylerArtifacts]:
    """
    Fit Tyler whitening and transform inputs in a single call.

    Returns:
        (Whitened observations, TylerArtifacts)
    """
    artifacts = fit(X, config=config)
    Z = transform(X, artifacts)
    return Z, artifacts


def tyler_mm_iter(Sigma_k: FloatArray, Z: FloatArray, rho: float) -> FloatArray:
    """One MM iteration of the Regularized Tyler estimator."""
    n_samples, n_features = Z.shape
    Sigma_k = ensure_pd(Sigma_k)
    try:
        solve = np.linalg.solve(Sigma_k, Z.T)
    except np.linalg.LinAlgError as err:
        logger.warning("Tyler iteration encountered singular matrix; adding jitter: %s", err)
        jitter = np.eye(n_features, dtype=np.float64) * 1e-6
        solve = np.linalg.solve(Sigma_k + jitter, Z.T)
    mahal_sq = np.sum(Z.T * solve, axis=0)
    mahal_sq = np.clip(mahal_sq, 1e-12, np.inf)
    weights = (n_features / n_samples) * (1.0 / mahal_sq)
    A = (Z.T * weights) @ Z
    Sigma_next = (1.0 - rho) * A + rho * np.eye(n_features, dtype=np.float64)
    Sigma_next = trace_normalize(Sigma_next)
    return ensure_pd(Sigma_next)


def spatial_median(X: FloatArray, *, tol: float = 1e-6, max_iter: int = 500) -> FloatArray:
    """Compute the spatial (geometric) median using Weiszfeld's algorithm."""
    if X.ndim != 2:
        raise ValueError("Input must be 2D array")
    median = np.median(X, axis=0).astype(np.float64, copy=True)
    eps = 1e-9
    for _ in range(max_iter):
        diff = X - median
        distances = np.linalg.norm(diff, axis=1)
        mask = distances > eps
        if not np.any(mask):
            break
        inv_dist = np.divide(1.0, distances, out=np.zeros_like(distances), where=mask)
        numerator = (inv_dist[:, None] * X).sum(axis=0)
        denominator = np.sum(inv_dist)
        if denominator <= eps:
            break
        next_median = numerator / denominator
        if np.linalg.norm(next_median - median) <= tol:
            median = next_median
            break
        median = next_median
    return median


def inv_sqrt(Sigma: FloatArray) -> FloatArray:
    """Compute the inverse square root of a PD matrix."""
    Sigma = ensure_pd(Sigma)
    eigenvalues, eigenvectors = eigh(Sigma)
    eigenvalues = np.clip(eigenvalues, 1e-12, np.inf)
    inv_sqrt_vals = 1.0 / np.sqrt(eigenvalues)
    return (eigenvectors * inv_sqrt_vals) @ eigenvectors.T


def ensure_pd(Sigma: FloatArray, *, eps: float = 1e-9) -> FloatArray:
    """Symmetrize and ensure positive definiteness via jitter if needed."""
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigenvalues = np.linalg.eigvalsh(Sigma)
    min_eval = np.min(eigenvalues)
    if min_eval <= eps:
        jitter = (eps - min_eval) + eps
        Sigma = Sigma + np.eye(Sigma.shape[0], dtype=np.float64) * jitter
    return Sigma


def trace_normalize(Sigma: FloatArray) -> FloatArray:
    """Scale matrix so that trace equals dimensionality."""
    trace = np.trace(Sigma)
    if trace <= 0.0:
        raise ValueError("Trace must be positive for normalization")
    dim = Sigma.shape[0]
    return Sigma * (dim / trace)

