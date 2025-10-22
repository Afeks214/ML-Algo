from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DEFAULT_NORM_CAP = 1e3
ARCCOSH_EPS = 1e-6
SERIES_THRESHOLD = 1e-3


def embed(z: FloatArray, *, norm_cap: float = DEFAULT_NORM_CAP) -> FloatArray:
    """
    Lift Euclidean vectors onto the Lorentzian hyperboloid.

    Ensures ||z|| <= norm_cap for numerical stability.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 2:
        raise ValueError("Input must be 2D array of shape (n, d)")
    squared_norm = np.sum(z_arr * z_arr, axis=1)
    if norm_cap is not None:
        norms = np.sqrt(squared_norm)
        scales = np.ones_like(norms)
        mask = norms > norm_cap
        scales[mask] = norm_cap / np.maximum(norms[mask], 1e-12)
        z_arr = z_arr * scales[:, None]
        squared_norm = np.sum(z_arr * z_arr, axis=1)
    u0 = np.sqrt(1.0 + squared_norm)
    return np.concatenate([u0[:, None], z_arr], axis=1)


def lorentz_inner(u: FloatArray, v: FloatArray) -> FloatArray:
    """
    Compute Lorentzian inner product between vectors u and v.

    Supports broadcasting on the leading dimensions.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if u_arr.shape[-1] != v_arr.shape[-1]:
        raise ValueError("Inputs must have matching last dimension")
    return -u_arr[..., 0] * v_arr[..., 0] + np.sum(u_arr[..., 1:] * v_arr[..., 1:], axis=-1)


def acosh_clamped(x: FloatArray, eps: float = ARCCOSH_EPS) -> FloatArray:
    """
    Numerically stable arcosh with lower clamp and series expansion near 1.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    clamped = np.maximum(x_arr, 1.0 + eps)
    delta = clamped - 1.0
    out = np.zeros_like(clamped)
    greater_mask = clamped > 1.0 + eps
    series_mask = (delta < SERIES_THRESHOLD) & greater_mask
    exact_mask = ~greater_mask
    out[series_mask] = sqrt_series(delta[series_mask])
    remaining = greater_mask & ~series_mask
    out[remaining] = np.arccosh(clamped[remaining])
    out[exact_mask] = 0.0
    return out


def sqrt_series(delta: FloatArray) -> FloatArray:
    """
    Series approximation for acosh(1 + delta) ~ sqrt(2 delta) * (1 + delta/12).
    """
    term = np.sqrt(2.0 * delta)
    return term * (1.0 + delta / 12.0)


def lorentz_distance(u: FloatArray, v: FloatArray, eps: float = ARCCOSH_EPS) -> FloatArray:
    """Geodesic distance on the Lorentzian hyperboloid."""
    s = -lorentz_inner(u, v)
    return acosh_clamped(s, eps=eps)


def batch_lorentz_distance(
    query: FloatArray,
    candidates: FloatArray,
    eps: float = ARCCOSH_EPS,
) -> FloatArray:
    """
    Compute Lorentz distances between batches of queries and candidates.

    Args:
        query: (n, d+1) array.
        candidates: (n, k, d+1) array.
    Returns:
        distances: (n, k) array.
    """
    if query.ndim != 2 or candidates.ndim != 3:
        raise ValueError("Expected query with ndim=2 and candidates with ndim=3")
    if query.shape[0] != candidates.shape[0]:
        raise ValueError("Batch size mismatch between query and candidates")
    if query.shape[1] != candidates.shape[2]:
        raise ValueError("Dimensionality mismatch")
    inner = -lorentz_inner(query[:, None, :], candidates)
    return acosh_clamped(inner, eps=eps)


@dataclass(frozen=True)
class RerankResult:
    ids: NDArray[np.int64]
    distances: FloatArray


def rerank_lorentz(
    query_embedding: FloatArray,
    neighbor_embeddings: FloatArray,
    neighbor_ids: NDArray[np.int64],
    k_final: int,
) -> RerankResult:
    """
    Lorentzian rerank selecting top-k_final neighbors.
    """
    if neighbor_embeddings.ndim != 2:
        raise ValueError("neighbor_embeddings must be 2D array")
    if neighbor_embeddings.shape[0] != neighbor_ids.shape[0]:
        raise ValueError("neighbor count mismatch")
    distances = lorentz_distance(query_embedding[None, :], neighbor_embeddings)[0]
    order = np.argsort(distances)
    top = order[:k_final]
    return RerankResult(ids=neighbor_ids[top], distances=distances[top])
