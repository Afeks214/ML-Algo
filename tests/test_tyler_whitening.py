from __future__ import annotations

import numpy as np
import pytest

from ml_algo.robust_scaling import TylerConfig, fit, fit_transform, transform


def heavy_tailed_sample(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_t(df=5, size=(512, 5))  # heavy-tailed
    scale = np.array(
        [
            [1.0, 0.3, 0.0, 0.0, 0.0],
            [0.3, 2.0, 0.4, 0.0, 0.0],
            [0.0, 0.4, 1.5, 0.2, 0.0],
            [0.0, 0.0, 0.2, 0.8, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.5],
        ]
    )
    cov = scale @ scale.T
    L = np.linalg.cholesky(cov)
    mu = np.array([1.0, -0.5, 2.0, 0.0, 0.75])
    return base @ L.T + mu


def test_tyler_whitening_covariance_near_identity() -> None:
    X = heavy_tailed_sample()
    cfg = TylerConfig(rho=0.2, tol=1e-6, max_iter=400)
    Z, artifacts = fit_transform(X, config=cfg)

    assert artifacts.converged
    assert artifacts.iterations <= cfg.max_iter
    eigenvalues = np.linalg.eigvalsh(artifacts.shape)
    assert np.min(eigenvalues) > 0.0

    assert np.isfinite(Z).all()

    scatter = (Z.T @ Z) / Z.shape[0]
    scatter_norm = scatter * (Z.shape[1] / np.trace(scatter))
    deviation = np.linalg.norm(scatter_norm - np.eye(scatter.shape[0]), ord="fro")
    assert deviation < 1.5


def test_transform_matches_fit_transform() -> None:
    X = heavy_tailed_sample(seed=1)
    cfg = TylerConfig(rho=0.15, tol=1e-6, max_iter=300)
    artifacts = fit(X, config=cfg)
    Z1 = transform(X, artifacts)
    Z2, _ = fit_transform(X, config=cfg)
    assert np.allclose(Z1, Z2, atol=1e-9)


@pytest.mark.parametrize("rho", [0.05, 0.3])
def test_tyler_handles_small_sample_sizes(rho: float) -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(64, 6))
    cfg = TylerConfig(rho=rho, tol=1e-6, max_iter=300)
    artifacts = fit(X, config=cfg)
    assert artifacts.shape.shape == (6, 6)
    assert artifacts.location.shape == (6,)
