# API Contracts (SPEC §9)

## Data Pipelines
- `ml_algo.data_ingest.load(sources, timezone, bar_sizes)`
  - Inputs: iterable of CSV paths, Olson timezone, expected bar sizes (informational).
  - Output: canonical OHLCV DataFrame sorted by `ts_utc`, validated via `schema.validate_ohlcv` and gap policy ready.
  - Guarantees: raises `ValueError` on schema violations; timestamps normalized to UTC.
- `ml_algo.heikin_ashi.transform(df)`
  - Inputs: OHLCV DataFrame (UTC or tz-aware) with required fields.
  - Output: Heikin-Ashi DataFrame conforming to `validate_heikin_ashi`; no NaNs permitted per invariants.
- `ml_algo.robust_scaling.fit_transform(X, config)`
  - Inputs: numpy array of features, Tyler config.
  - Output: whitened matrix + `TylerArtifacts` capturing convergence metadata.
  - Guarantees: trace-normalized scatter, PD check > `1e-8`.
- `ml_algo.ann_index.build(z_train, config)` / `search_with_vectors`
  - Inputs: whitened embedding bank, ANN config.
  - Output: `IndexRef` with version hash; search returns candidate ids/distances/vectors.
  - Guarantees: deterministic for identical input; recall SLO enforced via `RecallAutoTuner`.
- `ml_algo.pipeline.run_phase3/4/5/6/7`
  - End-to-end execution per SPEC phases: ingest -> Tyler -> ANN -> features -> CatBoost -> validation.
  - Observability: each phase exposes `observability` dict including recall, timings, calibration, CPCV summary.

## Schema Validators (`ml_algo.schema`)
- `validate_ohlcv`, `validate_heikin_ashi`, `validate_labels`, `validate_inference_input`.
  - Raise `SchemaValidationError` on missing columns, null policy violations, or missing timezone awareness.
  - Coerce datatypes to canonical forms (floats -> float64, timestamps -> UTC).

## Metrics & Validation
- `ml_algo.metrics.MetricReport` captures Accuracy/F1/ROC-AUC/ECE at a fixed threshold.
- `ml_algo.validation.cross_validate_catboost` orchestrates CPCV splits and returns `ValidationReport` with fold-level calibration summaries.
- `ValidationReport.to_json(path)` serializes fold metrics and aggregate summaries for artifact registries.

## Performance Instrumentation
- `ml_algo.performance.LatencyRecorder` / `StageTimer` capture per-stage latency for ANN, hyperbolic rerank, CatBoost training & inference.
- `Phase4Result.timings_ms` and `Phase6Result.timings_ms` propagate measurements into downstream observability + dashboards.

## Reproducibility
- `ml_algo.utils.run_metadata.collect_metadata` captures runtime metadata (python, platform, git SHA if available) and `persist_metadata` writes JSON artifacts.
- Artifact writers (Phase 6) persist metrics, calibration payloads, CPCV splits, and metadata alongside trained models.
