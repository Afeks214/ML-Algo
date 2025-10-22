# Colab GPU Training Runbook

This guide turns the repository into a Colab-ready training stack that follows SPEC Phase 12–14 requirements while using CatBoost on GPU.

## 1. Environment Bootstrap
1. Open a new **Google Colab** notebook with a GPU runtime (`Runtime` → `Change runtime type` → `T4`/`A100`).
2. Clone the repository and install dependencies:
   ```bash
   !git clone https://github.com/Afeks214/ML-Algo.git
   %cd ML-Algo
   !pip install --upgrade pip
   !pip install -e . catboost==1.2.5 faiss-gpu==1.7.4.post2 numpy pandas scipy ruamel.yaml
   ```
3. Set deterministic environment flags (aligns with SPEC §7/§12 reproducibility notes):
   ```python
   import os
   os.environ["PYTHONHASHSEED"] = "42"
   os.environ["GPU_DETERMINISTIC"] = "1"
   ```

## 2. Data Staging
1. Upload the dataset `@NQ - 5 min - ETH.csv` to Colab (or mount Drive) under `data/raw/`.
2. Freeze the snapshot by computing and logging its SHA256 hash:
   ```bash
   !python - <<'PY'
   import hashlib, pathlib
   path = pathlib.Path("data/raw/@NQ - 5 min - ETH.csv")
   digest = hashlib.sha256(path.read_bytes()).hexdigest()
   print(f"dataset_sha256={digest}")
   PY
   ```
3. Update `config/data.yaml` with the Colab path if it differs, and record the hash in run metadata.

## 3. Configuration (GPU Ready)
1. Edit `config/model.yaml` to switch CatBoost to GPU:
   ```yaml
   catboost:
     task_type: GPU
     devices: "0"
     boosting_type: Ordered
     allow_writing_files: false
     od_type: Iter
     depth: 8
     iterations: 1200
     learning_rate: 0.03
     l2_leaf_reg: 3.0
     subsample: 0.8
     random_seed: 42
   ```
2. Optional: tune `ann`, `kernels`, and `tyler` sections to match the large-run operating points decided in Phase 12 ablations.

## 4. End-to-End Training Script
Run the following cell to execute Phases 3–7 on GPU and persist artifacts:
```python
import json
from pathlib import Path
import pandas as pd

from ml_algo.ann_index import AnnConfig
from ml_algo.data_ingest import GapPolicy
from ml_algo.kernels import KernelEnsembleParams
from ml_algo.model_catboost import CatBoostConfig
from ml_algo.pipeline import run_phase3, run_phase4, run_phase5, run_phase6, run_phase7
from ml_algo.robust_scaling import TylerConfig

DATA = ["data/raw/@NQ - 5 min - ETH.csv"]
ARTIFACT_DIR = Path("artifacts/colab_run")

phase3 = run_phase3(
    sources=DATA,
    timezone="America/New_York",
    bar_sizes=["5min"],
    gap_policy=GapPolicy(max_gap_minutes=60),
    tyler_config=TylerConfig(rho=0.2, tol=1e-6, max_iter=400),
)
phase4 = run_phase4(
    phase3,
    ann_config=AnnConfig(
        backend="hnsw",
        k_cand=1024,
        ef_search=128,
        ef_search_max=512,
        nprobe=16,
        nprobe_max=64,
    ),
    k_final=64,
)
ha_close = phase3.ha["ha_close"].to_numpy()
labels = pd.Series((pd.Series(ha_close).shift(-1) > ha_close).astype(int).fillna(0).values, index=phase3.ha.index)
kernel_params = KernelEnsembleParams()
phase5 = run_phase5(
    phase4,
    labels=labels,
    kernel_params=kernel_params,
    train_model=False,
)
phase6 = run_phase6(
    phase5,
    catboost_config=CatBoostConfig(task_type="GPU", devices="0", iterations=1200, depth=8),
    artifact_dir=ARTIFACT_DIR,
)
phase7 = run_phase7(phase5)
phase7.report.to_json(ARTIFACT_DIR / "validation_report.json")
print("Validation summary:", phase7.summary)
```
The artifacts directory will contain the CatBoost CBM, metrics, calibration payload, CPCV splits, and run metadata (per Phase 12 DoD).

## 5. Post-run Validation & Export
1. Inspect latency/recall metrics from `phase4.timings_ms` and `phase7.summary`; confirm SLO compliance (ANN recall ≥0.95, p95 latency <25 ms).
2. Upload `artifacts/colab_run/` to versioned storage (e.g., Drive folder with run ID).
3. Document the run in the reproducibility log with dataset hash, config snapshot, git SHA, and Colab notebook link.

## 6. Troubleshooting
- **CUDA OOM**: lower `k_cand`, reduce batch size, or switch CatBoost `devices` to a subset (`"0:1"`).
- **CatBoost import errors**: ensure Colab runtime restarted after installation and that `pip show catboost` lists CUDA build.
- **ANN latency spikes**: adjust `AnnConfig.ef_search`/`nprobe` and rerun the Recall AutoTuner (Phase 4 module).
- **Non-deterministic metrics**: confirm `GPU_DETERMINISTIC=1`, fixed seeds, and Ordered boosting.

This process finalizes Phase 13–14 deliverables: GPU training readiness, artifact capture, and end-to-end rehearsal ahead of large-scale runs.
