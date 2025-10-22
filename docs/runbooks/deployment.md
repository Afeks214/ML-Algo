# Deployment & Ops Runbook (SPEC §11)

## Pre-Deployment Checklist
1. Freeze data snapshot (`data/raw`) and model configs (`config/*.yaml`).
2. Run full `pytest` suite and `ci/gates.py` (quality-gate script) to ensure thresholds met.
3. Export artifacts via `run_phase6(..., artifact_dir=...)` and `run_phase7(...).summary` for validation proof.
4. Collect metadata (git SHA, env hash) with `collect_metadata()` and store alongside artifacts.
5. Review risk/compliance gate logs; ensure `RiskState.kill_switch` is false.

## GPU Training Workflow
- Provision GPU environment (A100) with CUDA-compatible CatBoost/FAISS.
- Sync repo, restore conda env, run CPCV training using `run_phase7`.
- Persist `TylerArtifacts`, ANN indices, CatBoost model, calibration, validation report.
- Refer to `docs/runbooks/colab_gpu.md` for a Colab-specific workflow covering setup, training, and artifact export.

## Inference Service Steps
1. Warm load artifacts (Tyler matrices, ANN index, CatBoost model).
2. Initialize `LatencyRecorder` for stage monitoring; stream metrics to observability stack.
3. For each signal:
   - Ingest latest bar via `run_phase3` incremental update.
   - Retrieve neighbors (`run_phase4`) + features (`run_phase5`).
   - Evaluate CatBoost (`run_phase6`), pass through `risk_compliance.gate`.
   - Submit to ProjectX via `ProjectXClient.place_order` with idempotency key.

## Canary & Rollout
- Start canary at 5% flow with local rate limit to ProjectX (<=5 QPS).
- Monitor ANN recall, latency speedup, risk blocks; after two healthy windows escalate to 25% ? 50% ? 100%.
- Maintain rollback artifact set (previous model/index) for immediate failover.

## On-Call Playbook
- Kill switch: flip `RiskState.kill_switch=True` in control plane; confirms immediate block.
- News embargo: call `RiskState.mark_news(<timestamp>)` to enforce embargo window.
- Rate limit: adjust `ProjectXConfig.rate_limit_qps` for throttle; monitor error rate.
- Drift: schedule recalibration when `expected_calibration_error` grows > target.
