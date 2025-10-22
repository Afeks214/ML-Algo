from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _version_hash(array: FloatArray) -> str:
    data = np.asarray(array, dtype=np.float64)
    md5 = hashlib.md5()
    md5.update(data.tobytes())
    return md5.hexdigest()


@dataclass
class AnnConfig:
    backend: str = "exact"
    metric: str = "l2"
    k_cand: int = 64
    ef_search: int = 128
    ef_search_max: int = 512
    nprobe: int = 8
    nprobe_max: int = 64

    def clone(self) -> "AnnConfig":
        return AnnConfig(
            backend=self.backend,
            metric=self.metric,
            k_cand=self.k_cand,
            ef_search=self.ef_search,
            ef_search_max=self.ef_search_max,
            nprobe=self.nprobe,
            nprobe_max=self.nprobe_max,
        )


@dataclass
class IndexRef:
    data: FloatArray
    config: AnnConfig
    version: str = field(default="")


def build(z_train: FloatArray, config: AnnConfig) -> IndexRef:
    """Create an ANN index. Currently implements an exact L2 backend."""
    if z_train.ndim != 2:
        raise ValueError("Training vectors must be 2D array")
    if config.backend != "exact":
        raise NotImplementedError("Only backend='exact' is supported in this reference implementation")
    version = _version_hash(z_train)
    return IndexRef(data=np.asarray(z_train, dtype=np.float64), config=config.clone(), version=version)


def search(index: IndexRef, z_query: FloatArray, k_cand: int | None = None) -> Tuple[IntArray, FloatArray]:
    """Search the index and return ids and L2 distances."""
    if z_query.ndim != 2:
        raise ValueError("Query vectors must be 2D array")
    data = index.data
    dists = _pairwise_l2(data, z_query)
    k = k_cand or index.config.k_cand
    k = min(k, dists.shape[1])
    if k <= 0:
        raise ValueError("k_cand must be positive")
    kth = min(k - 1, dists.shape[1] - 1)
    idx = np.argpartition(dists, kth=kth, axis=1)[:, :k]
    idx_sorted = np.take_along_axis(idx, np.argsort(np.take_along_axis(dists, idx, axis=1), axis=1), axis=1)
    dist_sorted = np.take_along_axis(dists, idx_sorted, axis=1)
    return idx_sorted, dist_sorted


def search_with_vectors(
    index: IndexRef,
    z_query: FloatArray,
    k_cand: int | None = None,
) -> Tuple[IntArray, FloatArray, FloatArray]:
    ids, dists = search(index, z_query, k_cand=k_cand)
    vectors = get_vectors(index, ids)
    return ids, dists, vectors


def get_vectors(index: IndexRef, ids: IntArray) -> FloatArray:
    data = index.data
    return data[ids]


def recall_probe(
    index: IndexRef,
    z_query: FloatArray,
    ids_exact: IntArray,
    k_eval: int | None = None,
) -> float:
    ids, _ = search(index, z_query, k_cand=k_eval or ids_exact.shape[1])
    intersection = 0
    total = ids.size
    for row_pred, row_exact in zip(ids, ids_exact):
        intersection += len(set(row_pred).intersection(set(row_exact)))
    return intersection / max(total, 1)


def brute_force_knn(z_train: FloatArray, z_query: FloatArray, k: int) -> Tuple[IntArray, FloatArray]:
    dists = _pairwise_l2(z_train, z_query)
    idx = np.argpartition(dists, kth=min(k, dists.shape[1]-1), axis=1)[:, :k]
    idx_sorted = np.take_along_axis(idx, np.argsort(np.take_along_axis(dists, idx, axis=1), axis=1), axis=1)
    dist_sorted = np.take_along_axis(dists, idx_sorted, axis=1)
    return idx_sorted, dist_sorted


def _pairwise_l2(train: FloatArray, query: FloatArray) -> FloatArray:
    """Return matrix of pairwise L2 distances: shape (n_query, n_train)."""
    cross = query @ train.T
    query_norm = np.sum(query * query, axis=1)[:, None]
    train_norm = np.sum(train * train, axis=1)[None, :]
    dists_sq = np.maximum(query_norm - 2.0 * cross + train_norm, 0.0)
    return np.sqrt(dists_sq, out=dists_sq)


class RecallAutoTuner:
    """
    Simple recall auto-tuner that adapts ef_search / nprobe when recall drops.
    """

    def __init__(
        self,
        config: AnnConfig,
        *,
        target_recall: float = 0.95,
        max_growth_rate: float = 1.2,
        latency_budget_ms: float | None = None,
    ) -> None:
        self.config = config
        self.target_recall = target_recall
        self.max_growth_rate = max_growth_rate
        self.latency_budget_ms = latency_budget_ms
        self.param_bumps: list[tuple[str, int, int]] = []
        self._latest_latency_ms: float | None = None

    def configure(self, *, latency_budget_ms: float | None = None, max_growth_rate: float | None = None) -> None:
        if latency_budget_ms is not None:
            self.latency_budget_ms = latency_budget_ms
        if max_growth_rate is not None:
            self.max_growth_rate = max_growth_rate

    def step(self, *, observed_recall: float, latency_ms: float | None = None) -> bool:
        """
        Update search parameters based on observed recall and latency feedback.

        Returns:
            True if parameters were adjusted.
        """
        self._latest_latency_ms = latency_ms
        adjusted = False
        if observed_recall < self.target_recall:
            new_ef = min(
                math.ceil(self.config.ef_search * self.max_growth_rate),
                self.config.ef_search_max,
            )
            if new_ef > self.config.ef_search:
                self.param_bumps.append(("ef_search", self.config.ef_search, new_ef))
                self.config.ef_search = new_ef
                adjusted = True
            else:
                new_nprobe = min(
                    math.ceil(self.config.nprobe * self.max_growth_rate),
                    self.config.nprobe_max,
                )
                if new_nprobe > self.config.nprobe:
                    self.param_bumps.append(("nprobe", self.config.nprobe, new_nprobe))
                    self.config.nprobe = new_nprobe
                    adjusted = True
        if (
            latency_ms is not None
            and self.latency_budget_ms is not None
            and latency_ms > 1.2 * self.latency_budget_ms
        ):
            # latency too high, reduce parameters conservatively
            reduced = False
            if self.config.nprobe > 1:
                new_nprobe = max(1, math.floor(self.config.nprobe / self.max_growth_rate))
                if new_nprobe < self.config.nprobe:
                    self.param_bumps.append(("nprobe", self.config.nprobe, new_nprobe))
                    self.config.nprobe = new_nprobe
                    reduced = True
            if not reduced and self.config.ef_search > 1:
                new_ef = max(1, math.floor(self.config.ef_search / self.max_growth_rate))
                if new_ef < self.config.ef_search:
                    self.param_bumps.append(("ef_search", self.config.ef_search, new_ef))
                    self.config.ef_search = new_ef
                    reduced = True
            adjusted = adjusted or reduced
        return adjusted

    @property
    def latest_latency_ms(self) -> float | None:
        return self._latest_latency_ms
