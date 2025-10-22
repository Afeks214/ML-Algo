from __future__ import annotations

import numpy as np

from ml_algo.ann_index import (
    AnnConfig,
    RecallAutoTuner,
    brute_force_knn,
    build,
    recall_probe,
    search_with_vectors,
)


def test_exact_index_returns_self_neighbors() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(50, 6))
    cfg = AnnConfig(k_cand=5)
    index = build(data, cfg)
    ids, dists, vectors = search_with_vectors(index, data[:3], k_cand=5)
    assert ids.shape == (3, 5)
    assert np.allclose(dists[:, 0], 0.0, atol=1e-6)
    assert np.allclose(vectors[:, 0, :], data[:3], atol=1e-8)

    exact_ids, _ = brute_force_knn(data, data[:3], 5)
    recall = recall_probe(index, data[:3], exact_ids, k_eval=5)
    assert np.isclose(recall, 1.0)


def test_recall_autotuner_growth_and_latency_clamp() -> None:
    cfg = AnnConfig(
        k_cand=8,
        ef_search=10,
        ef_search_max=40,
        nprobe=2,
        nprobe_max=16,
    )
    tuner = RecallAutoTuner(cfg, target_recall=0.95, max_growth_rate=2.0, latency_budget_ms=50.0)
    adjusted = tuner.step(observed_recall=0.5, latency_ms=30.0)
    assert adjusted
    assert cfg.ef_search >= 10
    assert cfg.ef_search <= cfg.ef_search_max

    # Force latency clamp to trigger parameter reduction
    cfg.nprobe = 8
    cfg.ef_search = 32
    adjusted = tuner.step(observed_recall=0.99, latency_ms=200.0)
    assert adjusted
    assert cfg.nprobe <= 8
    assert cfg.ef_search <= 32
