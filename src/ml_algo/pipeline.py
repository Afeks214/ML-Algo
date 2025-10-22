from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from ml_algo.ann_index import (
    AnnConfig,
    IndexRef,
    RecallAutoTuner,
    brute_force_knn,
    build as build_ann_index,
    recall_probe,
    search_with_vectors,
)
from ml_algo.calibration import CalibrationResult
from ml_algo.cv_protocol import Fold, split_cpcv
from ml_algo.data_ingest import GapPolicy, apply_gap_policy, load
from ml_algo.features import FeatureAssemblerConfig, assemble_features
from ml_algo.heikin_ashi import transform as ha_transform
from ml_algo.hyperbolic import batch_lorentz_distance, embed as hyperbolic_embed
from ml_algo.kernels import KernelEnsembleParams
from ml_algo.metrics import (
    MetricReport,
    accuracy_score,
    binary_predictions,
    expected_calibration_error,
    f1_score,
    roc_auc_score,
)
from ml_algo.model_catboost import CatBoostConfig, CatBoostModel, IsotonicCalibrator
from ml_algo.performance import LatencyRecorder, StageTimer
from ml_algo.robust_scaling import TylerArtifacts, TylerConfig, fit_transform
from ml_algo.utils.run_metadata import RunMetadata, collect_metadata, persist_metadata
from ml_algo.validation import ValidationReport, cross_validate_catboost


def ingest_and_transform(
    sources: Iterable[str | Path],
    timezone: str,
    bar_sizes: Iterable[str],
    gap_policy: GapPolicy,
) -> pd.DataFrame:
    """Phase 1–2 pipeline stub: ingest OHLCV and produce Heikin-Ashi candles."""
    ohlcv = load(sources, timezone=timezone, bar_sizes=bar_sizes)
    ohlcv = apply_gap_policy(ohlcv, gap_policy)
    ha = ha_transform(ohlcv)
    return ha


FEATURE_COLUMNS_PHASE3: Sequence[str] = (
    "ha_open",
    "ha_high",
    "ha_low",
    "ha_close",
    "volume",
)


@dataclass(frozen=True)
class Phase3Result:
    """Container for Phase 3 preprocessing outputs."""

    ha: pd.DataFrame
    features: pd.DataFrame
    whitened: pd.DataFrame
    tyler: TylerArtifacts

    @property
    def whitened_array(self) -> np.ndarray:
        return self.whitened.to_numpy()

    @property
    def observability(self) -> Dict[str, Any]:
        return {
            "tyler_iterations": self.tyler.iterations,
            "tyler_converged": self.tyler.converged,
        }

def run_phase3(
    sources: Iterable[str | Path],
    timezone: str,
    bar_sizes: Iterable[str],
    gap_policy: GapPolicy,
    tyler_config: TylerConfig | None = None,
) -> Phase3Result:
    """
    Execute Phase 3 of the pipeline: ingest -> HA -> Tyler whitening.

    Returns:
        Phase3Result holding intermediate artifacts.
    """
    ha = ingest_and_transform(
        sources=sources,
        timezone=timezone,
        bar_sizes=bar_sizes,
        gap_policy=gap_policy,
    )
    feature_df = ha.loc[:, FEATURE_COLUMNS_PHASE3].astype("float64")
    whitened_array, tyler_artifacts = fit_transform(
        feature_df.to_numpy(copy=False),
        config=tyler_config,
    )
    whitened_df = pd.DataFrame(
        whitened_array,
        columns=[f"z_{i}" for i in range(whitened_array.shape[1])],
        index=feature_df.index,
    )
    return Phase3Result(
        ha=ha,
        features=feature_df,
        whitened=whitened_df,
        tyler=tyler_artifacts,
    )


@dataclass(frozen=True)
class Phase4Result:
    """Outputs for Phase 4 ANN retrieval and hyperbolic re-ranking."""

    phase3: Phase3Result
    ann_index: IndexRef
    candidate_ids: np.ndarray
    candidate_dists: np.ndarray
    candidate_vectors: np.ndarray
    query_embeddings: np.ndarray
    candidate_embeddings: np.ndarray
    reranked_ids: np.ndarray
    reranked_dists: np.ndarray
    recall: float
    tuner: RecallAutoTuner
    timings_ms: Dict[str, float]

    @property
    def observability(self) -> Dict[str, Any]:
        return {
            "ann_recall": self.recall,
            "param_bumps": list(self.tuner.param_bumps),
            "index_version": self.ann_index.version,
            "timings_ms": self.timings_ms,
        }

def run_phase4(
    phase3: Phase3Result,
    ann_config: AnnConfig,
    *,
    k_final: int = 32,
    tuner: RecallAutoTuner | None = None,
    latency_budget_ms: float | None = None,
) -> Phase4Result:
    """
    Execute Phase 4: build ANN index, retrieve candidates, and Lorentz rerank.
    """
    if k_final <= 0:
        raise ValueError("k_final must be positive")
    z = phase3.whitened_array
    recorder = LatencyRecorder()
    with StageTimer(recorder, "ann_build"):
        index = build_ann_index(z, ann_config)
    with StageTimer(recorder, "ann_search"):
        ids, base_dists, vectors = search_with_vectors(index, z, k_cand=ann_config.k_cand)
    with StageTimer(recorder, "hyperbolic_embed_queries"):
        query_embeddings = hyperbolic_embed(z)
    flat_vectors = vectors.reshape(-1, vectors.shape[-1])
    with StageTimer(recorder, "hyperbolic_embed_candidates"):
        candidate_embeddings = hyperbolic_embed(flat_vectors).reshape(vectors.shape[0], vectors.shape[1], -1)
    with StageTimer(recorder, "lorentz_distance"):
        lorentz_dists = batch_lorentz_distance(query_embeddings, candidate_embeddings)
    order = np.argsort(lorentz_dists, axis=1)
    order_top = order[:, :k_final]
    reranked_ids = np.take_along_axis(ids, order_top, axis=1)
    reranked_dists = np.take_along_axis(lorentz_dists, order_top, axis=1)

    with StageTimer(recorder, "exact_knn"):
        exact_ids, _ = brute_force_knn(z, z, ann_config.k_cand)
    with StageTimer(recorder, "recall_probe"):
        recall = recall_probe(index, z, exact_ids, k_eval=ann_config.k_cand)

    tuner = tuner or RecallAutoTuner(index.config, target_recall=0.95)
    if latency_budget_ms is not None:
        tuner.configure(latency_budget_ms=latency_budget_ms)
    ann_latency_ms = recorder.single_measurements().get("ann_search")
    with StageTimer(recorder, "tuner_step"):
        tuner.step(observed_recall=recall, latency_ms=ann_latency_ms)

    return Phase4Result(
        phase3=phase3,
        ann_index=index,
        candidate_ids=ids,
        candidate_dists=base_dists,
        candidate_vectors=vectors,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        reranked_ids=reranked_ids,
        reranked_dists=reranked_dists,
        recall=recall,
        tuner=tuner,
        timings_ms=recorder.single_measurements(),
    )


@dataclass(frozen=True)
class Phase5Result:
    """Phase 5 feature assembly and optional CatBoost training."""

    phase4: Phase4Result
    features: pd.DataFrame
    labels: pd.Series
    kernel_params: KernelEnsembleParams
    assembler_config: FeatureAssemblerConfig
    model: CatBoostModel | None

    @property
    def observability(self) -> Dict[str, Any]:
        return {
            "feature_count": self.features.shape[1],
            "rows": self.features.shape[0],
            "model_trained": self.model is not None,
        }


def run_phase5(
    phase4: Phase4Result,
    labels: pd.Series,
    kernel_params: KernelEnsembleParams,
    *,
    assembler_config: FeatureAssemblerConfig | None = None,
    catboost_config: CatBoostConfig | None = None,
    train_model: bool = False,
) -> Phase5Result:
    if len(labels) != phase4.reranked_ids.shape[0]:
        raise ValueError("labels length must match number of queries")
    assembler_cfg = assembler_config or FeatureAssemblerConfig()
    feature_df = assemble_features(phase4, labels, kernel_params, assembler_cfg)
    model: CatBoostModel | None = None
    if train_model:
        if catboost_config is None:
            catboost_config = CatBoostConfig()
        try:
            model = CatBoostModel(catboost_config)
            model.fit(feature_df, labels.values)
        except ImportError as err:  # pragma: no cover - runtime dependency
            raise RuntimeError("CatBoost training requested but catboost is not installed") from err
    return Phase5Result(
        phase4=phase4,
        features=feature_df,
        labels=labels,
        kernel_params=kernel_params,
        assembler_config=assembler_cfg,
        model=model,
    )


@dataclass(frozen=True)
class Phase6Result:
    """Phase 6 CatBoost training, calibration, and artifact persistence."""

    phase5: Phase5Result
    model: CatBoostModel
    metrics: MetricReport
    calibration: CalibrationResult | None
    train_indices: np.ndarray
    val_indices: np.ndarray
    metadata: RunMetadata
    artifact_dir: Path | None = None
    timings_ms: Dict[str, float] | None = None

    @property
    def observability(self) -> Dict[str, Any]:
        obs = self.metrics.as_dict()
        obs["calibration_method"] = self.calibration.method if self.calibration else "none"
        obs["train_size"] = int(self.train_indices.size)
        obs["val_size"] = int(self.val_indices.size)
        if self.timings_ms is not None:
            obs["timings_ms"] = self.timings_ms
        return obs


def _train_val_split(n_rows: int, eval_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < eval_fraction < 0.5:
        raise ValueError("eval_fraction must be in (0, 0.5)")
    if n_rows < 5:
        raise ValueError("Need at least 5 samples for train/validation split")
    val_count = max(1, int(np.floor(n_rows * eval_fraction)))
    if val_count >= n_rows:
        val_count = n_rows - 1
    split_start = n_rows - val_count
    train_indices = np.arange(split_start, dtype=int)
    val_indices = np.arange(split_start, n_rows, dtype=int)
    return train_indices, val_indices


def run_phase6(
    phase5: Phase5Result,
    *,
    catboost_config: CatBoostConfig | None = None,
    eval_fraction: float | None = None,
    calibration_method: str = "isotonic",
    artifact_dir: Path | str | None = None,
) -> Phase6Result:
    """
    Execute Phase 6: CatBoost training, calibration, and artifact export.
    """
    features = phase5.features
    labels = phase5.labels
    if features.empty:
        raise ValueError("Features are empty; cannot train CatBoost model")
    if labels.empty:
        raise ValueError("Labels are empty; cannot train CatBoost model")
    if len(features) != len(labels):
        raise ValueError("Feature and label lengths must match")

    cfg = catboost_config or CatBoostConfig()
    eval_frac = eval_fraction if eval_fraction is not None else cfg.eval_fraction
    train_idx, val_idx = _train_val_split(len(features), eval_frac)

    train_X = features.iloc[train_idx]
    train_y = labels.iloc[train_idx]
    val_X = features.iloc[val_idx]
    val_y = labels.iloc[val_idx]

    recorder = LatencyRecorder()
    model = CatBoostModel(cfg)
    with StageTimer(recorder, "train_catboost"):
        model.fit(train_X, train_y, eval_set=(val_X, val_y))

    with StageTimer(recorder, "validation_inference"):
        val_probs = model.predict_proba(val_X)[:, 1]
    val_preds = binary_predictions(val_probs)
    metrics = MetricReport(
        accuracy=accuracy_score(val_y, val_preds),
        f1=f1_score(val_y, val_preds),
        roc_auc=roc_auc_score(val_y, val_probs),
        ece=expected_calibration_error(val_y, val_probs),
        threshold=0.5,
    )

    calibration: CalibrationResult | None = None
    if calibration_method == "isotonic":
        with StageTimer(recorder, "calibration_fit"):
            calibrator = IsotonicCalibrator().fit(val_probs, val_y)
        model.set_calibrator(calibrator)
        with StageTimer(recorder, "calibration_apply"):
            calibrated_probs = calibrator.transform(val_probs)
        calibration = CalibrationResult(
            method="isotonic",
            before_ece=metrics.ece,
            after_ece=expected_calibration_error(val_y, calibrated_probs),
        )
    elif calibration_method == "none":
        model.set_calibrator(None)
    else:
        raise ValueError(f"Unsupported calibration_method: {calibration_method}")

    metadata = collect_metadata()
    out_dir: Path | None = None
    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
        with StageTimer(recorder, "artifact_persist"):
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save(out_dir)
            metrics_path = out_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics.as_dict(), indent=2))
            calibration_path = out_dir / "calibration.json"
            if calibration is not None:
                calibration_path.write_text(json.dumps(calibration.as_dict(), indent=2))
            else:
                calibration_path.write_text(json.dumps({"method": "none"}, indent=2))
            splits_path = out_dir / "splits.npz"
            np.savez(splits_path, train_indices=train_idx, val_indices=val_idx)
            persist_metadata(metadata, out_dir / "metadata.json")

    return Phase6Result(
        phase5=phase5,
        model=model,
        metrics=metrics,
        calibration=calibration,
        train_indices=train_idx,
        val_indices=val_idx,
        metadata=metadata,
        artifact_dir=out_dir,
        timings_ms=recorder.single_measurements(),
    )


@dataclass(frozen=True)
class Phase7Result:
    """Phase 7 validation via CPCV/PKF and metric aggregation."""

    phase5: Phase5Result
    folds: Sequence[Fold]
    report: ValidationReport

    @property
    def summary(self) -> Dict[str, float]:
        summary = dict(self.report.aggregate())
        summary["ann_recall"] = float(self.phase5.phase4.recall)
        timings = self.phase5.phase4.timings_ms
        ann_latency = timings.get("ann_search")
        exact_latency = timings.get("exact_knn")
        if ann_latency and exact_latency and ann_latency > 0.0:
            summary["latency_speedup"] = float(exact_latency / ann_latency)
        else:
            summary["latency_speedup"] = float("nan")
        return summary


def run_phase7(
    phase5: Phase5Result,
    *,
    purge: int = 5,
    embargo: int = 5,
    k: int = 5,
    h: int = 1,
    catboost_config: CatBoostConfig | None = None,
    calibration_method: str = "isotonic",
) -> Phase7Result:
    """
    Execute Phase 7 validation using CPCV splits and aggregate metrics.
    """
    timestamps = phase5.phase4.phase3.ha.index
    folds = split_cpcv(
        list(pd.Index(timestamps)),
        purge=purge,
        embargo=embargo,
        k=k,
        h=h,
    )
    report = cross_validate_catboost(
        phase5.features,
        phase5.labels,
        folds,
        config=catboost_config,
        calibration_method=calibration_method,
    )
    return Phase7Result(
        phase5=phase5,
        folds=tuple(folds),
        report=report,
    )
