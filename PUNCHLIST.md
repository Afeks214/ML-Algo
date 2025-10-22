# Staging/Canary Punch‑List (GO) and Production Blockers (NO‑GO)

Status
- Staging / canary (1–5% flow): GO
- Production (100% flow): NO‑GO until five blockers are closed

Cross‑refs: SPEC.md §3–§11; Appendices C, F, G, H, I, M, Q, R.

## Blocker 1 — ✅ Completed — ANN Neighbor Vectors at Inference

Tasks
- ann_index: add `search_with_vectors(Z_q, K_cand) -> (ids, base_dists, Z_neighbors)` and `get_vectors(ids)`.
- pipeline: replace `Z[ids]` with vectors from index store (memmap); plumb through Lorentz re‑rank and kernels.

Acceptance tests
- Unit: `search_with_vectors` returns vectors matching train store for random `ids` (hash equality).
- Integration: E2E re‑rank correctness invariant unaffected by sliding window; no IndexError on window shifts.
- Observability: log index version, store path, and `vectors_returned=true` per query.

Owners: ML Eng (A/R), Data Eng (C), Infra (C)

## Blocker 2 — ✅ Completed — Periodic Kernel τ Semantics (Time‑difference in bars)

Tasks
- model.yaml: add `kernels.periodic.tau_policy: time_diff_bars` and validate.
- kernels: compute `τ = |t_q − t_i|` (bars); cache `sin(π τ / p)`.
- tests: golden examples; LOOCV plugin choice documented.

Acceptance tests
- Unit: `periodic(τ)` matches expected values for known τ; cache hit ratio > 90% on repeated τ.
- Config: schema validation rejects invalid `tau_policy`.

Owners: ML Eng (A/R)

## Blocker 3 — ✅ Completed — Fold Isolation (Purge/Embargo) for Tyler & ANN

Tasks
- cv_protocol/pipeline: build Σ̂ and ANN index per‑train‑fold only; isolate artifact directories per fold; forbid cross‑use.
- Add asserts preventing artifact reuse across train/val.

Acceptance tests
- Unit: `no_information_overlap(folds)`; artifact paths include `fold_id` and differ across folds (hash diff).
- Integration: training a second fold does not touch artifacts of the first.

Owners: ML Eng (A/R), Data Eng (C)

## Blocker 4 — ✅ Completed (FP16 fallbacks pending) — Numerical Guards

Tasks
- hyperbolic: pairwise/Kahan sum for `-⟨u,v⟩_L`; clamp `s≥1+1e-6`; series path near boundary.
- robust_scaling: Cholesky solves; adaptive ρ; PD+trace checks each iteration.
- metrics/inference: FP16→FP32 auto‑fallback counters per stage.

Acceptance tests
- Lorentz: `⟨u,u⟩_L = -1 ± 1e-6`; `d_L(u,u)=0`; boundary acosh tests use series path.
- Tyler: converge ≤ 500 its; `eigmin>1e-8`; `trace≈d` after whitening.
- Fallbacks: inject NaNs in FP16 path → FP32 recompute observed; `fallback_count{stage}` increments.

Owners: ML Eng (A/R)

## Blocker 5 — ANN Recall Auto‑Tuner

Tasks
- ann_index/controller: monitor rolling recall; increase `ef_search`/`nprobe` when recall < 0.95; persist chosen params.
- observability: emit `ann_recall`, `ef_search`, `nprobe`, `param_bumps`.

Acceptance tests
- Unit: synthetic planted neighbor set triggers tuner to raise `ef_search` until recall ≥ 0.95.
- Integration: p95 latency remains under budget after tuning within ±20%.

Owners: ML Eng (A/R), Infra (C)

---

## Fast Path to GO - Code/Config Checklist
- [x] ANN API + recall controller implemented and wired in pipeline.
- [x] 	au_policy=time_diff_bars pinned in model.yaml; parameter validation added.
- [x] Per-fold artifact isolation enforced for Tyler whitening and ANN indices.
- [ ] Numeric clamps + series guards + FP16/FP32 fallbacks in place; boundary unit tests added.
- [x] Kernels output effective_neighbors (eff_k) and weight-sum floors.

## Tests — Hard Gates (Appendix M)
- [ ] ANN recall ≥ 0.95 (95% CI); monotone in `ef/nprobe`.
- [ ] Lorentz invariants and acosh boundary tests pass.
- [ ] Tyler convergence ≤ 500 its; PD and whitening shape ≈ I.
- [ ] CV artifact isolation; no information set overlap.
- [ ] E2E: p95 per‑stage ≤ §8 budget; total p95 < 25 ms; accuracy ≥ 0.90× KNN‑exact.

## Observability
- Emit: `ann_recall`, `ef/nprobe`, `fallback_count{stage}`, `dL_stats`, `eff_k`, `ECE`, `PSI`.
- Degrade flags when p95 over budget (reduce `K_cand` → skip LP/Per → skip re‑rank).

## Compliance/Risk (TopstepX/ProjectX)
- Device/IP fingerprinting; hard block on VPN/VPS/remote.
- Idempotency keys; client‑side rate limit; jittered retries; deadlines.
- Kill‑switch; embargo guard chaos‑tested.

## Launch Plan (Post‑Blockers)
1) Freeze snapshot + configs + seeds; sign artifacts (model, Σ̂, index).
2) Pareto cert on frozen shard: (ef/nprobe, K_cand) → choose operating point meeting recall/latency.
3) Canary 1–5% via ProjectX sandbox → real; monitor recall, p95, ECE, coverage.
4) Ramp 5% → 25% → 50% → 100% when gates green for two windows; no compliance hits; no OOD alarms.
5) Rollback tag prepared; previous index/model warm‑standby.

## Issue Template
```
Title: [Blocker N] <short description>
Module(s): <module files>
Spec refs: <SPEC.md sections>
Tasks:
- [ ] …
Acceptance:
- [ ] …
Owners: Research (A), ML Eng (R), Data Eng (C), Infra (C), Compliance (I)
```

