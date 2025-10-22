from __future__ import annotations

import numpy as np

from ml_algo.hyperbolic import (
    batch_lorentz_distance,
    embed,
    lorentz_distance,
    lorentz_inner,
)


def test_embed_enforces_lorentz_invariants() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(128, 6))
    U = embed(z, norm_cap=25.0)
    inner = lorentz_inner(U, U)
    assert np.allclose(inner, -1.0, atol=1e-6)

    d_self = lorentz_distance(U[[0]], U[[0]])
    assert np.allclose(d_self, 0.0, atol=1e-6)


def test_batch_lorentz_distance_matches_pairwise() -> None:
    rng = np.random.default_rng(1)
    q = embed(rng.normal(size=(4, 3)))
    candidates = embed(rng.normal(size=(4, 10, 3)).reshape(-1, 3)).reshape(4, 10, 4)
    batched = batch_lorentz_distance(q, candidates)
    manual = np.stack(
        [
            lorentz_distance(q[[i]], candidates[i])
            for i in range(candidates.shape[0])
        ],
        axis=0,
    )
    assert np.allclose(batched, manual, atol=1e-6)
