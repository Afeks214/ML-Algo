# Reproducibility Runbook (SPEC §9 & Appendix B)

## Environment Capture
1. Create Conda env: `conda env create -f environment.yml`.
2. Install optional deps (GPU / catboost) per deployment target.
3. Record `git rev-parse HEAD`, `conda env export`, and `pip list --format freeze` snapshots; store under `artifacts/metadata/`.

## Metadata Logging
- `collect_metadata()` auto-populates python version, platform, hostname, git SHA (if repository available), and environment hash placeholder.
- Persist metadata via `persist_metadata(metadata, path)` for every model export (Phase 6) and validation run (Phase 7).

## Seeds & Determinism
- Configure seeds in `config/model.yaml` (`random_seed`, `global_seed`, `catboost.random_seed`).
- CatBoost training uses Ordered boosting with deterministic flags; set `GPU_DETERMINISTIC=1` when running on CUDA.
- ANN recalls rely on deterministic exact backend; track `AnnConfig` dumps alongside model artifacts.

## Artifact Registry Checklist
1. Tyler whitening artifacts (`TylerArtifacts`) per fold.
2. ANN index configs and version hashes (`IndexRef.version`).
3. CatBoost CBM + `config.json` + `calibrator.json` + `metrics.json` + `metadata.json` + `splits.npz`.
4. Validation `ValidationReport` summary (export via `.aggregate()` to JSON).

## Run Log Template
```
run_id: <uuid>
datetime_utc: <timestamp>
git_sha: <sha>
conda_env: <hash>
seeds:
  global: <int>
  catboost: <int>
ann_config:
  backend: exact
  k_cand: <int>
  ef_search: <int>
  nprobe: <int>
validation:
  folds: <k/h>
  metrics: {...}
```
