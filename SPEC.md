# SOTA Specification: Hyperbolic-ANN + Kernel Ensemble + CatBoost over Heikin–Ashi with ProjectX/TopstepX Integration

## 1. Title & Executive Summary
This specification defines a production-ready, end-to-end machine learning pipeline for financial time series classification. The system transforms raw OHLCV bars into Heikin–Ashi (HA) candles, applies robust standardization using a Regularized Tyler’s M-estimator, retrieves candidate neighbors via ANN (HNSW/FAISS, L2 or IP), and re-ranks them in the Lorentz hyperbolic model using the geodesic distance \(d_L(u,v)=\operatorname{arcosh}(-\langle u, v \rangle_L)\). From these neighborhoods, it generates kernel-weighted features using an ensemble of Rational Quadratic (RQ), Periodic, and Locally-Periodic kernels (via Nadaraya–Watson, with optional MKL), and performs final classification using CatBoost (GPU). Evaluation follows purged and embargoed time-series cross-validation (CPCV/PKF). The trained model is deployed with performance SLOs and integrated to ProjectX → TopstepX under strict compliance (no VPN/VPS) and risk controls.

Key outcome goals:
- Accuracy: CatBoost achieves at least 90% of the temporal holdout accuracy of an exact KNN baseline (L2, brute-force CPU).
- Speed: ≥95× lower median/p95 prediction latency versus exact KNN (same hardware, online micro-batches), at ANN recall ≥0.95.

## 2. Scope & Non-Goals
- In-scope: Mathematical foundations; deterministic feature/label definitions; ANN retrieval + hyperbolic re-ranking; kernel ensemble feature design (NW/MKL); CatBoost training/evaluation (GPU); CPCV/PKF; performance engineering; deployment, risk/compliance; ProjectX→TopstepX integration; ops playbooks.
- Non-goals: Portfolio optimization, live PnL targets, discretionary overlays, broker/routing selection beyond ProjectX→TopstepX; optimization of execution microstructure; non-U.S. tax/compliance guidance.

## 3. System Architecture

### 3.1 Component Block Diagram (ASCII)
```
[Raw OHLCV Stream] -> [Heikin–Ashi Transform] -> [Robust Whitening (Regularized Tyler)]
         |                         |                            |
         v                         v                            v
   [Labeler (T-horizon)]   [Feature Bank (HA/Tech/Events)]  [Whitened Vectors]
                                                         |
                                                         v
                                    [ANN Index (HNSW/FAISS, L2/IP)]
                                                |
                                                v
                                     (Top-K_cand IDs, vectors)
                                                |
                                                v
                                   [Lorentz Re-Rank via d_L]
                                                |
                                                v
                 [Kernel Ensemble Features: NW (RQ + Periodic + Locally-Periodic); MKL]
                                                |
                                                v
                                      [CatBoost Classifier (GPU)]
                                                |
                                                v
                               [Validation (CPCV) & Perf (p50/p95)]
                                                |
                                                v
                   [Inference Runtime] -> [Risk/Compliance] -> [ProjectX API] -> [TopstepX]
```

### 3.2 Live Signal Sequence Diagram (ASCII)
```
Agent/Daemon -> DataIngest: fetch last N OHLCV bars
DataIngest -> HA: compute Heikin–Ashi (HA) candles
HA -> Tyler: robust scatter (Regularized Tyler), whitening map
Tyler -> ANN: search (ef_search / nprobe) -> candidate IDs + base distances
ANN -> Lorentz: compute d_L(u,v) for candidates -> re-rank -> top-K
Lorentz -> Kernels: NW weights (RQ + Periodic + Locally-Periodic), MKL blend
Kernels -> Features: neighborhood stats + kernel scores + HA/tech indicators
Features -> CatBoost: predict class and confidence
CatBoost -> Risk: risk checks (caps, news embargo, scaling plan)
Risk -> ProjectX: submit/modify/cancel order (REST/WebSocket)
ProjectX -> TopstepX: route and confirm
TopstepX -> Agent/Daemon: execution ack; update state
```

### 3.3 Runtime Control Flow & Interactions
- Data ingestion normalizes schema/timezone, fills small gaps per policy, and materializes HA candles.
- Robust whitening (Tyler) standardizes feature vectors before distance computations and embedding; outputs stabilized vectors.
- ANN performs fast candidate retrieval in L2 or IP space; Lorentz re-ranking enforces the problem’s hyperbolic geometry for the decision step.
- Kernel ensemble (NW/MKL) converts neighbor sets into continuous features (scores, densities, dispersions).
- CatBoost consumes all features; predictions flow through Risk/Compliance gating, then to ProjectX for execution to TopstepX.

### 3.4 Failure Modes (high-level)
- Data gaps/clock skew → downstream label/feature leakage; ANN recall instability → degraded accuracy; hyperbolic numeric overflow → NaNs; GPU OOM → fallback path; API throttling → order rejection; compliance violations → account suspension. Mitigations and tests in Section 11.4 (Failure & Safety).

## 4. Mathematical Foundations (formal)

### 4.1 Heikin–Ashi (HA) definitions
For time t with raw OHLCV \((O_t,H_t,L_t,C_t,V_t)\):
\[
\text{Close}_{\mathrm{HA},t}=\tfrac{O_t+H_t+L_t+C_t}{4},\quad
\text{Open}_{\mathrm{HA},t}=\tfrac{\text{Open}_{\mathrm{HA},t-1}+\text{Close}_{\mathrm{HA},t-1}}{2}
\]
\[
\text{High}_{\mathrm{HA},t}=\max\{H_t,\,\text{Open}_{\mathrm{HA},t},\,\text{Close}_{\mathrm{HA},t}\},\quad
\text{Low}_{\mathrm{HA},t}=\min\{L_t,\,\text{Open}_{\mathrm{HA},t},\,\text{Close}_{\mathrm{HA},t}\}
\]

### 4.2 Regularized Tyler’s M-estimator and whitening
Let centered samples \(x_i\in\mathbb{R}^p\) (robust centering recommended, e.g., spatial median). The Regularized Tyler shape estimator \(\hat{\Sigma}\succ 0\) solves the fixed-point iteration (one form):
\[
\Sigma_{k+1}= (1-\rho)\,\frac{p}{n}\sum_{i=1}^n \frac{x_i x_i^\top}{x_i^\top\,\Sigma_k^{-1}\,x_i} + \rho\,I,\quad \text{normalize } \operatorname{tr}(\Sigma_{k+1})=p
\]
with shrinkage \(\rho\in(0,1]\.\) Existence/uniqueness conditions hold with regularization even when \(n<p\). Properties: scale-invariance (shape), affine equivariance (up to scale), robustness to heavy tails/outliers. Whitening map: \(z_i=\hat{\Sigma}^{-\tfrac{1}{2}}(x_i-m)\) with robust location \(m\). Asymptotics: spectral limits relate to Marchenko–Pastur under elliptical models.

Invariants for whitening: \(\mathbb{E}[z]=0\), \(\operatorname{Cov}(z)\approx I\) (shape), improved conditioning for distance computations.

### 4.3 Lorentz-model hyperbolic geometry
Lorentz inner product on \(\mathbb{R}^{d+1}\):
\[\langle u,v\rangle_L = -u_0v_0 + \sum_{j=1}^d u_j v_j.\]
Hyperboloid model \(\mathcal{H}^d=\{u\in\mathbb{R}^{d+1}: \langle u,u\rangle_L=-1, u_0>0\}\). Geodesic distance:
\[d_L(u,v)=\operatorname{arcosh}\big(-\langle u,v\rangle_L\big),\quad u,v\in\mathcal{H}^d.\]
Embedding (example “lift” from \(z\in\mathbb{R}^d\)): \(u_0=\sqrt{1+\lVert z\rVert_2^2}\), \(u_{1:d}=z\), which ensures \(\langle u,u\rangle_L=-1\). Time-like constraint: \(u_0>0\). Numeric stability: clamp the argument of \(\operatorname{arcosh}(\cdot)\) to \(\ge 1+\epsilon\), and bound \(\lVert z\rVert\) (e.g., via scaling) to prevent overflow.

### 4.4 ANN retrieval theory & complexity
HNSW: multi-layer small-world graphs enable sublinear search with high recall via \(M,\,ef_{construction},\,ef_{search}\) tuning. FAISS: IVF-PQ/HNSW on GPU with \(\text{nlist},\,\text{nprobe}\) and product quantization for compression. Two-stage approach: ANN (L2/IP) retrieves \(K_{cand}\) candidates; exact re-ranking (Lorentz \(d_L\)) selects final \(K\). Trade-offs: higher \(ef_{search}/nprobe\) increases recall (and latency); PQ lowers memory at some accuracy cost.

### 4.5 Kernel ensemble (NW/MKL)
Nadaraya–Watson estimator for a query \(x\):
\[\hat{f}(x)=\frac{\sum_i w_i(x)\,y_i}{\sum_i w_i(x)},\quad w_i(x)=K_\theta(\mathrm{dist}(x,x_i)).\]
Here \(K_\theta\) acts as a weight function (NW does not require PSD). Kernels used:
- Rational Quadratic (RQ): \(K_{\mathrm{RQ}}(r)=\left(1+\frac{r^2}{2\alpha \ell^2}\right)^{-\alpha}\).
- Periodic: \(K_{\mathrm{Per}}(\tau)=\exp\!\left(-\frac{2\sin^2(\pi\tau/p)}{\ell^2}\right)\).
- Locally-Periodic: \(K_{\mathrm{LP}}=K_{\mathrm{SE}}\cdot K_{\mathrm{Per}},\; K_{\mathrm{SE}}(r)=\exp(-r^2/(2\ell^2))\).

MKL (optional): learn convex weights \(\lambda_m\ge 0,\sum_m\lambda_m=1\) over kernels \(K_m\), or use localized MKL for nonstationarity. Indefinite kernels can be handled in RKKS (Kreĭn spaces) when needed, though NW/GBDT do not require PSD.

## 5. Feature Engineering (deterministic specs)
Ordering (leakage-safe): HA → robust centering → Regularized Tyler whitening → ANN retrieval (L2/IP) → Lorentz re-rank (\(d_L\)) → kernel/NW features → CatBoost.

Feature groups and definitions:
- Neighborhood geometry (from top-K after re-rank): mean/median \(d_L\), min/max \(d_L\), std of \(d_L\), neighbor density (e.g., inverse mean \(d_L\)), angular proxies (via normalized Lorentz inner products).
- NW kernel scores: per-kernel \(\hat{f}_{\mathrm{RQ}},\hat{f}_{\mathrm{Per}},\hat{f}_{\mathrm{LP}}\), plus MKL blend \(\hat{f}_{\mathrm{MKL}}\); include weight-sum, effective-neighbor-count, and leverage diagnostics.
- ANN retrieval stats: candidate set size, HNSW levels visited, approximate distance of closest/farthest, recall proxy (intersection with brute subset if available online), ef_search/nprobe used.
- HA/technical indicators: HA body/upper/lower wicks, HA trend streak, ATR, RSI/WT/CCI/ADX; regime features (volatility buckets), event proximity (time to macro release).
- Normalization: all continuous features standardized post-whitening; clip extreme values (winsorization) per CV only on training folds.

Invariants & guards:
- Strict temporal alignment; labels computed with lookahead horizon T and embargo; no future-derived fields when constructing features at time t.
- All hyperbolic computations operate on valid \(\mathcal{H}^d\) embeddings with stability clamps.

## 6. Model Training (CatBoost on GPU)
Hyperparameters (search ranges): depth 6–10; learning rate 0.02–0.10; iterations 1k–5k; l2_leaf_reg 1–10; subsample 0.6–1.0; border_count 128–255 (if needed). Class imbalance handled via class_weights or balanced subsampling. Calibration: post-hoc isotonic or Platt scaling on validation folds. Optional domain monotonic constraints if feature semantics justify (documented per-feature).

Training protocol:
- Use Ordered boosting to mitigate target leakage on sequential data.
- Early stopping with patience; monitor Accuracy/F1 and calibration ECE; track wall-clock and GPU utilization.
- Reproducibility: fixed seeds; pinned package versions; deterministic cuDNN/CUDA flags where possible.

## 7. Validation Protocol
- Split strategy: Purged & Embargoed K-Fold or CPCV to avoid temporal leakage between train/validation; Walk-Forward for sanity checks only.
- Targets: binary/multi-class labels defined on T-horizon (e.g., direction, thresholded ATR-adjusted returns). Class definitions frozen before any tuning.
- Metrics: Accuracy, F1, ROC-AUC; plus latency SLOs (p50/p95) for serving; ANN recall ≥0.95 on validation search queries.
- Statistical hygiene: report bootstrap confidence intervals; control multiple comparisons across grids; keep a fixed baseline shard for comparability.

Acceptance criteria:
- Relative accuracy ≥ 0.90 × (Accuracy of exact KNN on the same holdout).
- Latency speedup ≥ 95× vs brute-force KNN at serving on same hardware and micro-batch size.

## 8. Performance Engineering

Index configurations:
- HNSW (CPU/GPU): \(M\in[16,48], ef_{construction}\in[100,500], ef_{search}\in[64,512]\), target recall 0.95–0.98.
- FAISS IVF-PQ (GPU): \(\text{nlist}\in[256,4096], \text{nprobe}\in[4,64]\); PQ bytes tuned to memory budget; optional pre-transform to FP16.

Two-stage retrieval:
- Stage 1 (ANN): retrieve \(K_{cand}\in[512,2048]\) using L2/IP.
- Stage 2 (Re-rank): compute \(d_L\) and select final \(K\in[16,64]\); vectorized with clamped arcosh; micro-batched.

Vectorization & memory layout:
- Contiguous arrays (row-major) for BLAS-friendly ops; FP16 distances where acceptable; pinned host memory for H2D; micro-batches sized to fit GPU SRAM; kernel precomputations cached.

Cost model & profiling plan:
- Instrument p50/p95 per stage (ingest, HA, Tyler, ANN, re-rank, kernels, CatBoost, risk, API). Run at multiple N,d,K, ef/nprobe settings; produce Pareto frontiers (recall vs latency).

Throughput/Latency Budget (serving; example targets):
| Stage | p50 (ms) | p95 (ms) | Notes |
|---|---:|---:|---|
| Ingest + HA | 0.5 | 1.5 | cached last bars |
| Tyler transform (online) | 0.6 | 2.0 | apply pre-fit whitening |
| ANN search | 1.8 | 6.0 | ef_search/nprobe tuned for recall |
| Lorentz re-rank | 0.9 | 3.0 | K_cand up to 2k |
| Kernel features | 0.6 | 2.0 | NW + stats |
| CatBoost predict | 0.4 | 1.2 | GPU/CPU depending on deployment |
| Risk gating | 0.3 | 1.0 | local checks |
| ProjectX API | 2.0 | 8.0 | network variance |
| Total | 7.1 | 24.7 | meets SLO if p95 < 25 ms |

## 9. Software Architecture & Repo Layout (VS Code workspace)

File tree and ownership (approx LOC per file):
```
repo/
├─ SPEC.md
├─ config/
│  ├─ data.yaml                # data contracts, sources, timezones
│  ├─ model.yaml               # CatBoost, ANN, kernels, MKL params
│  └─ risk.yaml                # limits, embargo times, API toggles
├─ src/
│  ├─ data_ingest.py          (~150) – schemas, tz, gap policy
│  ├─ heikin_ashi.py          (~120) – HA recurrences, sanity checks
│  ├─ robust_scaling.py       (~200) – Reg. Tyler, whitening, centering
│  ├─ hyperbolic.py           (~220) – Lorentz ops, d_L, stability clamps
│  ├─ ann_index.py            (~240) – HNSW/FAISS adapters, recall SLOs
│  ├─ kernels.py              (~240) – RQ/Periodic/Locally-Periodic, NW, MKL
│  ├─ features.py             (~200) – neighborhood stats & aggregations
│  ├─ model_catboost.py       (~220) – train/eval, calibration, export
│  ├─ cv_protocol.py          (~180) – Purged/Embargoed K-Fold, CPCV
│  ├─ metrics.py              (~160) – cls metrics, latency probes
│  ├─ risk_compliance.py      (~160) – caps, kill-switches, news embargo
│  ├─ projectx_client.py      (~220) – auth, rate limits, retries, orders
│  └─ pipeline.py             (~240) – end-to-end DAG & CLI
├─ tests/
│  ├─ test_heikin_ashi.py
│  ├─ test_tyler_whitening.py
│  ├─ test_hyperbolic_distance.py
│  ├─ test_ann_recall.py
│  ├─ test_kernels_nw.py
│  ├─ test_cv_protocol.py
│  └─ test_e2e_pipeline.py
├─ docs/
│  └─ api_contracts.md
└─ ci/
   └─ pipeline.yml
```

Layering rules:
- Math primitives (HA, Tyler, Lorentz, Kernels) are side-effect free, deterministic.
- Index adapters isolate third-party engines (FAISS/HNSW) behind stable contracts.
- Feature assembly composes math + retrieval outputs; Model layer consumes only feature tensors.
- Pipeline orchestrates; Risk/Compliance gates outputs; Client handles I/O.

Function contracts (examples):
- `robust_scaling.fit(X: R^{n×p}, center: bool) -> Sigma_hat: PD, m: R^p;` invariants: PD(\(\Sigma^\)), tr-normalized, convergence within tol.
- `robust_scaling.transform(X) -> Z` pre: fitted; post: approx identity shape.
- `hyperbolic.embed(Z: R^{n×d}) -> U: H^{d}` pre: finite Z; post: \(\langle u,u\rangle_L=-1\), \(u_0>0\).
- `hyperbolic.d_L(Uq: H^{d}, Uk: H^{d×K}) -> R^{K}` numeric clamps ensure arg(\(\operatorname{arcosh}\)) ≥ 1+ε.
- `ann_index.build(X: R^{n×d}, config) -> IndexRef` side effects: creates on-disk index.
- `ann_index.search(Q: R^{m×d}, K_cand: int) -> (ids: int^{m×K_cand}, dist: R^{m×K_cand})` SLO: recall≥0.95 on holdout.
- `kernels.nw_scores(q, NNs, params) -> {rq, per, lp, mkl, weightsum, eff_k}` deterministic for fixed params.
- `model_catboost.fit(F: R^{n×f}, y) -> Model` `predict_proba(f: R^f) -> R^C` calibration optional.
- `cv_protocol.split_purged_embargo(T, purge, embargo) -> folds` correctness: no overlap of information sets.
- `projectx_client.place(order) -> ack` with retries, rate limits, and idempotency keys.

Configuration & reproducibility:
- YAML/ENV for data/model/risk; secrets via environment (no secrets in repo); seed all RNGs; pin library versions; record git commit hashes in artifacts.

### 9.1 RACI Matrix (roles are placeholders)
| Workstream | Research | Data Eng | ML Eng | Infra | Compliance | Trading Ops |
|---|---|---|---|---|---|---|
| Data ingest & contracts | A | R | C | C | C | I |
| Heikin–Ashi & features | A | C | R | C | I | I |
| Tyler whitening | A | C | R | C | I | I |
| ANN & Lorentz re-rank | A | C | R | C | I | I |
| Kernel ensemble (NW/MKL) | A | C | R | C | I | I |
| CatBoost training | A | I | R | C | I | I |
| Validation (CPCV) | A | I | R | C | I | I |
| Inference service | C | I | R | A | I | C |
| ProjectX integration | C | I | R | A | C | A |
| Risk/Compliance gating | C | I | C | I | A | R |
| Monitoring/observability | C | I | R | A | I | C |
| Release/change control | C | I | R | A | C | C |

### 9.2 Data Contracts
| Dataset | Schema (fields: type) | Units | Sampling | Timezone | Null policy |
|---|---|---|---|---|---|
| Raw OHLCV | ts: datetime, open: float, high: float, low: float, close: float, volume: float | price: instrument currency; vol: units | 1–5m; 15–60m | Exchange tz normalized to UTC | Drop bars with missing O/H/L/C; forward-fill volume sparsely; log gaps |
| Heikin–Ashi | ts, ha_open: float, ha_high: float, ha_low: float, ha_close: float | price | inherits | UTC | deterministic from OHLCV; no nulls |
| Technical features | ts, feature_k: float | mixed | per bar | UTC | compute-only on present bars; backfill strictly avoided |
| Events | ts_event: datetime, type: enum, severity: int | n/a | event feed | UTC | filter to market hours; if missing, mark as none |
| Labels | ts, y: int/enum | n/a | per bar | UTC | defined via future window T; embargo prevents leakage |
| Inference input | latest HA/feat vector | n/a | on demand | UTC | must pass validation; else skip signal |
| Inference output | ts, y_hat: enum, p: float, meta | n/a | on demand | UTC | always populated |

## 10. Testing Strategy
Unit tests:
- HA identities and recurrence consistency; Tyler fixed-point convergence and PD; Lorentz invariants (\(\langle u,u\rangle_L=-1\), \(u_0>0\)), monotone \(d_L\) with separation; NW weight normalization.

Property-based tests:
- Affine scaling invariance post-whitening; ANN recall monotonic in ef_search/nprobe; stability of \(\operatorname{arcosh}\) clamps.

Integration tests:
- ANN → Lorentz re-rank → NW → CatBoost on a frozen shard with golden outputs; CPCV split correctness (no overlap of information sets).

End-to-end rehearsal:
- Synthetic stream with timing hooks; verify p50/p95 within budget; fail-fast on API errors; verify kill-switch behavior.

Artifacts & CI:
- Store metrics, confusion matrices, calibration plots, latency histograms; CI blocks on SLO violations; publish model cards with configs/hash.

## 11. Deployment & Ops
Training (Colab A100):
- Provision GPU runtime; install CatBoost (CUDA-enabled) and FAISS GPU if used; mount data; run CPCV; export artifacts (scalers, embeddings config, CatBoost model) with versioned registry.

Inference service:
- Health checks (liveness/readiness); latency/error dashboards; drift monitors (feature/label where available); recalibration cadence; blue/green or canary deploys.

ProjectX → TopstepX integration:
- Use ProjectX Gateway API for order routing; enforce TopstepX constraints: no VPN/VPS/remote servers; activity originates from personal device; respect eligibility policies. Implement rate limits, retries with jitter, and idempotency.

### 11.1 Failure & Safety Table
| Symptom | Likely cause | Mitigation | Test |
|---|---|---|---|
| Missing bars / clock drift | Data source lag; TZ mismatch | Enforce UTC; strict gap policy; reject stale | Inject gaps; assert embargoed labels |
| ANN recall drops | ef_search/nprobe too low; index stale | Auto-tune recall; periodic rebuild; alarms | Sweep ef/nprobe; recall unit test |
| d_L NaNs/inf | arcosh arg<1; overflow | Clamp arg≥1+ε; rescale vectors | Fuzz test near-boundary |
| Tyler non-convergence | Poor init; extreme outliers | Increase ρ; cap iterations; fallback to shrinkage | Convergence unit test |
| GPU OOM | Oversized micro-batch | Auto-resize batch; fallback CPU | Stress test with large K_cand |
| Latency SLO breach | Network/API slowness | Backpressure; degrade (skip re-rank); cache | Chaos test with network delay |
| Calibration drift | Regime change | Recalibrate; drift alarm | Rolling ECE monitoring |
| API throttling | Burst orders | Client-side rate limit; idempotent retries | Throttle simulation |
| Compliance violation | VPN/VPS use; geo policy | Device/IP checks; hard blocks | Simulate VPN detection |
| Timezone mismatch | Wrong TZ mapping | Canonical UTC; contract tests | Contract unit test |

## 12. Roadmap & Ablations
Experiments matrix:
- HNSW vs FAISS IVF-PQ; ef_search/nprobe sweeps; K_cand/K trade-offs; RQ vs Periodic vs Locally-Periodic; MKL on/off; Lorentz re-rank on/off; Tyler on/off; CatBoost hypergrid.

GO/NO-GO criteria:
- Relative accuracy ≥ 0.90 of KNN-exact; serving speedup ≥95×; ANN recall ≥0.95; no critical failure in Failure & Safety; compliance checks pass.

Definition of Done (DoD):
- Mathematics: HA, Tyler, Lorentz, NW/MKL documented and validated with unit/property tests.
- Performance: p95 latency budget met; recall SLO met; profiling report with Pareto curves delivered.
- Testing: CPCV reports with CIs; integration/e2e tests pass; golden files updated.
- Compliance: ProjectX/TopstepX integration tested; device/IP verification; kill-switch and risk caps verified.
- Reproducibility: seeds, version pins, artifacts registry, and runbooks finalized.

## 13. References (endnotes)
1. Nickel, M., Kiela, D. (2018). Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry. PMLR v80. https://proceedings.mlr.press/v80/nickel18a/nickel18a.pdf
2. Law, M. T., et al. (2019). Lorentzian Distance Learning for Hyperbolic Representations. PMLR v97. https://proceedings.mlr.press/v97/law19a/law19a.pdf
3. Sun, Y., Babu, P., Palomar, D. P. (2014). Regularized Tyler’s Scatter Estimator: Existence, Uniqueness, and Algorithms. https://arxiv.org/abs/1407.3079
4. Zhang, L., Cheng, X., Singer, A. (2014/2016). Marchenko–Pastur Law for Tyler’s and Maronna’s M-estimators. https://web.math.princeton.edu/~amits/publications/1401.3424v1.pdf
5. Malkov, Y., Yashunin, D. (2016). Efficient and Robust Approximate Nearest Neighbor Search using HNSW. https://arxiv.org/abs/1603.09320
6. Johnson, J., Douze, M., Jégou, H. (2017). Billion-scale similarity search with GPUs (FAISS). https://arxiv.org/abs/1702.08734
7. Wilson, A. G., Adams, R. P. (2013). Gaussian Process Kernels for Pattern Discovery and Extrapolation. https://arxiv.org/abs/1302.4245
8. Rasmussen, C. E., Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. https://gaussianprocess.org/gpml/
9. Nadaraya–Watson Kernel Regression (tutorial). https://scispace.com/pdf/the-nadaraya-watson-kernel-regression-function-estimator-feusjgollb.pdf
10. Dorogush, A. V., Ershov, V., Gulin, A. (2018). CatBoost: gradient boosting with categorical features support. https://arxiv.org/abs/1810.11363
11. López de Prado, M. (2018). Advances in Financial Machine Learning (purged/embargoed CV). Wiley.
12. Heikin–Ashi chart (definitions). https://en.wikipedia.org/wiki/Heikin-Ashi_chart
13. ProjectX Gateway API Documentation. https://gateway.docs.projectx.com/
14. TopstepX API Access and VPN policy. https://help.topstep.com/en/articles/11187768-topstepx-api-access and https://help.topstep.com/en/articles/8680268-can-i-use-a-vpn
15. Ong, C. S., et al. (2004). Learning with Non-Positive Kernels (Kreĭn/RKKS). https://www.ong-home.my/papers/ong04krein-icml.pdf

---

# Appendix A — Function Interaction Matrix & State Machines

Cross-reference: aligns with §3 (System Architecture), §9 (Software Architecture), and function contracts therein.

## A.1 Function Interaction Matrix
| Module/Function | Direct Callers | Direct Callees | Inputs (types) | Outputs (types) | Failures/Errors | Notes |
|---|---|---|---|---|---|---|
| `pipeline.run` | CLI, scheduler | all stage drivers | `config`, `run_id` | artifacts, reports | stage timeout, dependency failure | Orchestrates end-to-end DAG; idempotent via `run_id` |
| `data_ingest.load` | `pipeline.run` | validators | source URIs, tz | OHLCV frame | schema mismatch, TZ errors | Enforces Data Contracts (§9.2) |
| `heikin_ashi.transform` | `pipeline.run`, `pipeline.infer` | — | OHLCV | HA bars | NaNs, length mismatch | Recurrences and invariants (§4.1) |
| `robust_scaling.fit` | `pipeline.run` | `tyler_mm_iter` | X_train | `Σ̂`, `m` | non-convergence | Regularized Tyler (§4.2) |
| `robust_scaling.transform` | `pipeline.run`, `pipeline.infer` | — | X | Z | PD loss | Whitening; affine equivariance |
| `tyler_mm_iter` | `robust_scaling.fit` | — | `Σ_k`, X, ρ | `Σ_{k+1}` | loss of PD, divergence | Clamp, trace-normalize |
| `ann_index.build` | `pipeline.run` | backend APIs | Z_train, config | IndexRef | OOM, build timeout | HNSW/FAISS (§4.4) |
| `ann_index.search` | `pipeline.infer` | backend APIs | Z_q, K_cand | ids, base_dists | recall drop | Tune ef_search/nprobe |
| `hyperbolic.embed` | `pipeline.run`, `pipeline.infer` | — | Z | U∈ℋ^d | overflow | Clamp norms; enforce `u₀>0` (§4.3) |
| `hyperbolic.d_L` | `pipeline.infer` | — | U_q, U_ids | d_L vector | arcosh domain | arg clamp ≥1+ε |
| `lorentz_rerank` | `pipeline.infer` | `hyperbolic.d_L` | ids, U, K | topK_ids, dL | NaN, ties | Stable sort; drop NaNs |
| `kernels.nw_scores` | `pipeline.infer` | kernels | q, NNs, θ | {rq, per, lp, mkl} | under/overflow | Normalize weights; floor sum>ε |
| `features.assemble` | `pipeline.run`, `pipeline.infer` | — | geometry, kernels, HA | F | schema mismatch | Strict schema/types |
| `model_catboost.fit` | `pipeline.run` | CatBoost API | F_train, y | model | early stop | Ordered boosting (§6) |
| `model_catboost.predict` | `pipeline.infer` | CatBoost API | F_q | ŷ, p | — | Calibrated output |
| `cv_protocol.split` | `pipeline.run` | — | T, purge, embargo | folds | overlap | Leakage guard (§7) |
| `risk_compliance.gate` | `pipeline.infer` | — | ŷ, p, state | action | rule breach | Kill-switch, caps (§11) |
| `projectx_client.place` | `pipeline.infer` | HTTP/WebSocket | action | ack | throttle, 4xx/5xx | Retries, idempotency |

## A.2 State Machines

### A.2.1 Inference Runtime
States: `IDLE → FETCH → PREPARE(HA/Tyler) → RETRIEVE(ANN) → RERANK(Lorentz) → SCORE(NW/MKL) → PREDICT(CatBoost) → GATE(Risk) → EXEC(ProjectX) → IDLE`.

- Transition FETCH→PREPARE: pre: new OHLCV ≥ 1 bar; post: HA computed; idempotent: yes (same input → same HA).
- PREPARE→RETRIEVE: pre: whitening ready; post: query vector Z_q valid; timeout: soft (retry once).
- RETRIEVE→RERANK: pre: ids length = K_cand; post: d_L computed; failure: low recall → adjust ef_search/nprobe.
- RERANK→SCORE: pre: top-K selected; post: NW features present; idempotent: yes for fixed params.
- SCORE→PREDICT: pre: feature schema matches; post: (ŷ,p) available; timeout: short.
- PREDICT→GATE: pre: (ŷ,p) within [0,1]; post: action ∈ {pass, throttle, block}; idempotent: stateful (depends on limits).
- GATE→EXEC: pre: action=pass; post: ack or retry; idempotency key required; timeout: bounded with backoff.

### A.2.2 Risk/Compliance Gate
States: `CHECK_LIMITS → (OK→PASS) | (WARN→THROTTLE) | (VIOLATION→BLOCK/KILL)`.

- Preconditions: account state fresh (≤ Δt), exposure snapshot consistent; Postconditions: action emitted, audit log written. Idempotency: repeated checks yield same action for same state. Timeout: enforce deadline and default to THROTTLE.

### A.2.3 ProjectX Order Client
States: `READY → SUBMIT → (ACK→READY) | (RETRY→SUBMIT) | (FAIL→ALERT)`.

- Preconditions: risk PASS; connectivity healthy; Postconditions: order ack with idempotency. Retries: exponential backoff with jitter; Max attempts M; Timeouts per API SLO; Idempotency: per (client_id, nonce).

## A.3 Pseudocode-style Function Contracts

- Tyler iteration (Regularized, §4.2):
  - `tyler_mm_iter(Sigma_k, X, rho) -> Sigma_next`
  - Pre: `Sigma_k ≻ 0`, `rho∈(0,1]`; Post: `Sigma_next ≻ 0`, `tr(Sigma_next)=p`; Converges if tol ≤ 1e-6 within max_iter; Fallback: increase `rho`.

- Two-stage ANN + Lorentz re-rank (§4.4, §4.3):
  - `retrieve_and_rerank(Z_q, IndexRef, K_cand, K) -> (topK_ids, dL_topK)`
  - `search_with_vectors(IndexRef, Z_q, K_cand) -> (ids, base_dists, Z_neighbors)` (ANN API addition for inference)
  - Pre: IndexRef built; Post: `|topK_ids|=K`, `dL_topK` finite; Invariants: recall≥target; Timeouts: ANN search bounded; Idempotent for same inputs.

- NW/MKL scoring (§4.5):
  - `nw_mkl_scores(q, NNs, params) -> {rq, per, lp, mkl, weightsum, eff_k}`
  - Pre: distances finite; Post: weight sum>ε; MKL weights convex; Numerical: clamp exponents.

- CPCV splitting (§7):
  - `cpcv_split(T, K, purge, embargo) -> folds`
  - Pre: `T` strictly ordered; Post: no overlap of information sets; Invariants: embargo bars excluded from training around validation windows.

---

# Appendix B — Configuration Schemas & Reproducibility

Cross-reference: complements §9 (Configuration & reproducibility) and §10 (Testing Strategy).

## B.1 Canonical YAML Schemas

### B.1.1 `config/data.yaml`
- `sources: list<uri>` (required)
- `timezone: string` default `UTC`
- `bar_sizes: list<enum>` in {`1m`,`5m`,`15m`,`60m`} (non-empty)
- `gap_policy: enum` in {`drop`,`interpolate_limited`} default `drop`
- `events_feed: uri|null`
- `label:`
  - `horizon_T: int` min 1
  - `type: enum` in {`direction`,`ternary`,`thresholded`}
  - `embargo_bars: int` ≥ 0
- `splits:`
  - `scheme: enum` in {`CPCV`,`PurgedKFold`,`WalkForward`}
  - `purge_bars: int` ≥ 0
  - `folds: int` ≥ 2

Validation rules: all times normalized to UTC; `horizon_T + embargo_bars` must be < smallest split window.

### B.1.2 `config/model.yaml`
- `whitening.tyler:` `rho: float∈[0.01,1.0]`, `max_iter: int≤1000`, `tol: float≤1e-6`
- `ann:` `backend: enum{HNSW,FAISS_IVF_PQ}`, `K_cand: int∈[256,4096]`, `K: int∈[8,128]`
  - `hnsw: {M:int∈[16,48], ef_construction:int∈[100,500], ef_search:int∈[64,512]}`
  - `faiss: {nlist:int∈[256,4096], nprobe:int∈[4,64], pq_bytes:int∈[8,32]}`
- `hyperbolic:` `clamp_eps: float=1e-6`, `norm_cap: float>0`
- `kernels:` `rq:{alpha>0, ell>0}`, `periodic:{period>0, ell>0}`, `locally_periodic:{ell_se>0, period>0, ell_per>0}`, `mkl_weights:list<float>` sum≈1.0
- `catboost:` `depth:int∈[6,10]`, `lr:float∈(0,1)`, `iterations:int∈[1000,5000]`, `l2_leaf_reg:float∈[1,10]`, `subsample:float∈(0,1]`, `class_weights:list<float>|null`

Additions for Blocker 2 (Periodic Kernel tau semantics):
- Add `kernels.periodic.tau_policy: enum{time_diff_bars}` — compute tau = |t_q - t_i| in bars; schema must reject invalid values; default must be explicit in config.

### B.1.3 `config/risk.yaml`
- `caps:` `{daily_loss: float>0, max_pos: int≥0, news_embargo_min: int≥0}`
- `projectx:` `{endpoint: uri, key_env: string, rate_limit_qps: float>0}`
- `compliance:` `{no_vpn: bool=true, device_check: bool=true}`

## B.2 Reproducibility Policies
- Random seeds: `{global, cv, catboost}` tracked in run metadata.
- Versions: pin Python/conda/pip packages; record CUDA/cuDNN versions and driver; store `requirements-lock.txt`.
- Artifacts: hash model (`SHA256(model.cbm)`), `Σ̂.bin`, `index.bin`, `config/*.yaml`, and data snapshot IDs; include git commit SHA and dirty flag.
- Run metadata: `{start_ts, end_ts, hostname, GPU model & count, OS, git SHA, data snapshot IDs, config hashes}` persisted with reports.

---

# Appendix C — Numerical Stability & Precision Policy

Cross-reference: details for §4 (Mathematical Foundations) and §8 (Performance Engineering).

## C.1 Numerical Stability Table
| Routine | Domain/Range | Epsilon/Clamp | Precision | Tolerances | Fallback Strategy |
|---|---|---|---|---|---|
| Tyler MM (§4.2) | `Σ_k ≻ 0` | `tol=1e-6`, `max_iter≤500` | FP32 | Δtr≤1e-6, ‖ΔΣ‖_F≤1e-5 | Increase ρ; cap influence; shrink towards I |
| Whitening | finite X | clip | FP32 | ‖Z‖ ≤ z_max | revert to z-score scaling |
| Lorentz embed (§4.3) | `u₀=√(1+‖z‖²)` | cap ‖z‖≤z_max | FP32 | u₀ finite, >1 | rescale z by s<1 |
| Lorentz inner prod | finite U | clip sum | FP32 | within float range | Kahan summation |
| arcosh | x≥1 | arg←max(x,1+1e-6) | FP32 | near-1 stable | series approx near 1+ε |
| NW kernels (§4.5) | r≥0 | exp clamp to [-40,40] | FP16/32 | sum w_i≥1e-12 | renorm; switch to FP32 |
| MKL blending | λ_m≥0, ∑λ=1 | project to simplex | FP32 | ‖λ‖₁≈1 | reproject via softmax |
| ANN dists | finite | [0,d_max] | FP16 for dists | NaN-free | recompute slab in FP32 |
| CatBoost predict | valid feature range | clip features | FP32 | calibrated ECE≤5% | recalibrate |

## C.2 Parametric Cost Model (Throughput/Latency)
Let N=corpus size, d=feature dim, K_cand and K final, `ef=ef_search`, `np=nprobe`, `nl=nlist`.

- HNSW search: \(t_{ann} ≈ a_0 + a_1\,ef + a_2\,\log N\). Empirically, `a_1` dominates; coefficients profiled per host.
- FAISS IVF: \(t_{ann} ≈ b_0 + b_1\,np\,(N/nl) + b_2\,K_{cand}\). Memory lower with PQ at slight accuracy cost.
- Re-rank (Lorentz): \(t_{rerank} ≈ c_0 + c_1\,K_{cand}\,d\) (vectorized arcosh, bounded by memory bandwidth).
- NW/Kernels: \(t_{kern} ≈ k_0 + k_1\,K\,M\) with M=#kernels (here 3–4) plus overhead for MKL.
- CatBoost predict: \(t_{cb} ≈ m_0 + m_1\,\text{trees}\) (often negligible vs retrieval at micro-batch sizes).

Total per-query latency:
\[ t_{total} ≈ t_{ingest}+t_{ha}+t_{whiten}+t_{ann}+t_{rerank}+t_{kern}+t_{cb}+t_{risk}+t_{api}. \]
Use §8 budgets as constraints and fit \(\{a_i,b_i,c_i,k_i,m_i\}\) via profiling to forecast p50/p95 as N,K parameters evolve.

---

# Appendix D — Testing & Quality Gates

Cross-reference: extends §10 (Testing Strategy), §7 (Validation Protocol), and §8 (Performance Engineering).

## D.1 Expanded Tests and Assertions
- HA unit tests: exact recurrence; no NaNs; golden file comparison (tolerance 1e-12).
- Tyler: convergence within 500 iters; PD check (`eigmin>1e-8`); invariance to scaling; whitening yields \(\operatorname{trace}(\hat{\Sigma})≈p\).
- Lorentz: \(\langle u,u\rangle_L=-1\), `u₀>0`; \(d_L(u,u)=0\); monotonicity with ‖z‖.
- arcosh clamp: `arg≥1+1e-6`; series approx error ≤ 1e-6 near boundary.
- ANN recall: monotone non-decreasing in `ef`/`nprobe`; recall≥0.95 on validation queries.
- NW/MKL: weight sums≥1e-12; MKL weights convex; outputs bounded.
- CPCV: no overlap of information sets; embargo strictly enforced.

## D.2 CI Quality Gates (Hard Fail)
- ANN recall ≥ 0.95 (validation queries, 95% CI reported).
- p95 latency per stage ≤ §8 budget (environment-tagged).
- Accuracy ≥ 0.90 × KNN-exact (same holdout and label definition).
- Coverage: math modules ≥ 90% branch; pipeline ≥ 80%.
- Failure-injection suite passes (Appendix E.2).

## D.3 Leakage Ledger (Fields & Mitigations)
| Potential leakage field | Source | Risk | Mitigation |
|---|---|---|---|
| Future returns within horizon T | labeler | direct target leakage | Purged/embargoed splits; labels computed after split materialization |
| Post-event timestamps | events | look-ahead | Truncate to t; use only known-at-t flags |
| Rolling indicators using future bars | features | peeking | Causal windows ending at t-1 |
| ANN index built on full data | ann | transductive leakage | Build index per train fold only |
| Whitening fit on full data | scaling | leakage | Fit on train only; transform val/test |
| Hyperparam tuning on test | model | selection bias | Test set quarantined; only final check |

---

# Appendix E — Observability, Safety, and Governance

Cross-reference: complements §11 (Deployment & Ops) and §11.1 (Failure & Safety).

## E.1 Observability Specification
Additions for Blockers alignment:
- Structured logs: add `index_version`, `store_path`, `vectors_returned` (Blocker 1).
- Counters: add `fallback_count{stage}`, `param_bumps` (Blockers 4, 5).
- Gauges: add `ef_search`, `nprobe`, `eff_k` alongside `ann_recall` (Blockers 1, 5).
- Summaries: add `dL_stats` (distribution summaries) to existing `distance_values`.
- Structured logging fields: `{ts, stage, req_id, symbol, fold_id, K_cand, ef, nprobe, latency_ms, recall_est, decision, risk_state, order_id, http_status}`.
- Tracing: propagate `trace_id`, `span_id` across stages; annotate retries and backoffs.
- Metrics:
  - Counters: `orders_placed`, `orders_rejected`, `killswitch_trips`, `api_retries`, `ann_timeouts`.
  - Gauges: `ann_recall`, `queue_depth`, `gpu_mem_used_mb`.
  - Histograms: `latency_ms{stage=…}` with p50/p95; `distance_values{type=L2,dL}` sanity bands.
- Drift monitors: feature means/variances vs training; population stability index (PSI); calibration ECE; alert thresholds: PSI>0.2 or ECE>0.1 for 3 consecutive windows.

## E.2 Failure & Safety (Detailed)
| Symptom | Likely cause | Mitigation | Test |
|---|---|---|---|
| Missing bars / clock drift | Data lag; TZ mismatch | Enforce UTC; gap policy; reject stale | Inject gaps; assert embargo correctness |
| ANN recall drops | ef/nprobe too low; stale index | Auto-tune; rebuild; alarms | Param sweep; teardown/rebuild test |
| d_L NaNs/inf | arcosh arg<1; overflow | Clamp; rescale; FP32 fallback | Fuzz at boundary; overflow injection |
| Tyler non-convergence | Heavy tails; poor init | Increase ρ; max_iter cap; shrinkage | Convergence watchdog |
| GPU OOM | Batch too large | Auto-resize; CPU fallback | Stress with increasing batch |
| Latency breach | Network/API slowness | Backpressure; degrade (skip re-rank); cache | Chaos: add latency to API |
| Calibration drift | Regime change | Recalibrate; alert | Rolling ECE monitor |
| API throttling | Burst traffic | Client rate limit; idempotent retries | Throttle simulation |
| Compliance violation | VPN/VPS; geo block | Device/IP checks; hard block | VPN detection test |
| Order duplication | Retry without idempotency | Use idempotency keys | Duplicate injection test |

Backpressure & degrade modes:
- When p95 latency>budget: reduce `K_cand`, skip non-essential kernels, or temporarily disable re-rank while preserving risk checks; log degrade mode with `degrade=on` tag.

## E.3 Model Card Template
- Model: name, version, git SHA, training dates, CatBoost params.
- Data: sources, symbols, bar sizes, timezone, snapshot IDs, label definition.
- Metrics: Accuracy/F1/AUC with 95% CIs; p50/p95 latencies; ANN recall.
- Risks & limits: intended use, exclusions, compliance notes (no VPN/VPS), kill-switch triggers.
- Owners & RACI: contacts per §9.1; approvals (ML Eng, Compliance, Trading Ops).
- Artifacts: checksums for model, Σ̂, index, configs; environment details.

## E.4 Change Management Process
- All changes via PR with approvals from ML Eng and Compliance; tag releases (semver) and create release notes including metrics deltas.
- Maintain audit logs of deployments (who/when/what SHA); support rollback to previous tag; store signed model cards.

## E.5 Go/No-Go Sign-offs

---

# Appendix F — Decision Policy, Calibration, and Uncertainty

Cross-reference: consumes outputs of §6 (CatBoost), §7 (Validation), and integrates with §11 (Risk/Compliance) and Appendix E.

## F.1 Cost-Sensitive Decision Rule
Let \(\hat p=\Pr(y=1\mid x)\) be the calibrated score (§F.2). With false-positive cost \(C_{FP}>0\), false-negative cost \(C_{FN}>0\), and class prior \(\pi\), select threshold \(\tau\) to minimize expected cost:
\[\tau^*=\arg\min_\tau\; \mathbb{E}\big[ C_{FP}\,\mathbf{1}\{\hat p\ge \tau, y=0\} + C_{FN}\,\mathbf{1}\{\hat p<\tau, y=1\}\big].\]
Operationally choose \(\tau\) along ROC/PR curves (per fold) where expected utility is maximized; log \(\tau\) and \(C_{FP},C_{FN}\) in the Model Card (Appendix E.3). Different venues/sessions may warrant per-regime \(\tau\) (§I.1).

## F.2 Calibration Diagnostics and Cadence
- Metrics: Expected Calibration Error (ECE), Brier score; report per fold (§7) and in production rolling windows (§E.1).
- Recalibration: perform isotonic or Platt scaling on validation folds; retrain calibrator on schedule (e.g., weekly) or when ECE>0.1 for 3 consecutive windows (§E.1).
- Threshold tuning: maintain a validation-time ROC/PR catalog; at deployment, pick \(\tau\) matching current class balance and costs; re-evaluate with each recalibration.

## F.3 Conformal Prediction for Confidence Sets
Adopt Mondrian (class-conditional) conformal prediction to emit a confidence set \(\Gamma(x)\subseteq \{0,1\}\) targeting marginal coverage \(1-\alpha\) per class. Nonconformity scores can be based on calibrated \(1-\hat p\) or margin \(|\hat p-0.5|\).

- Coverage target: \(1-\alpha=0.9\) by default; monitor empirical coverage in production; alert if below by >5% for 3 windows.
- Consumption by Risk Gate (§A.2.2, §E):
  - PASS: if \(\max\hat p \ge \tau\), \(|\Gamma(x)|=1\) and in-domain.
  - THROTTLE: if \(|\Gamma(x)|>1\) or nonconformity above warn threshold.
  - BLOCK: if OOD detected (via drift monitors §E.1) or persistent coverage shortfall.
- Logging: record \(\Gamma(x)\), \(\alpha\), empirical coverage, and set size distributions.

---

# Appendix G — Kernel Bandwidth & MKL Procedures

Cross-reference: extends §4.5 (Kernel ensemble & NW/MKL) and §8 (Performance).

## G.1 Variable-Bandwidth Nadaraya–Watson
Let \(h(x)\) be a local bandwidth. Two policies:
- Balloon estimator: kernel centered at \(x\) with \(h(x)\) based on local density (e.g., \(k\)-NN distance: \(h(x)=c\cdot r_k(x)\)).
- Sample-point estimator: per-sample bandwidth \(h_i\) (e.g., \(h_i=c\cdot r_k(x_i)\)); weights use \(h_i\) instead of \(h(x)\).

Selection:
- Local LOOCV: choose \(h\) to minimize LOOCV loss in a neighbor shell; ensure floor \(h_{min}>0\), cap \(h_{max}\).
- Plug-in: use Silverman-like rules with robust scale estimates after whitening (§4.2).

Numerical guards: floor weight-sum at \(\varepsilon_w=10^{-12}\); clamp exponents to [-40,40]; renormalize.

## G.2 MKL Optimization (Convex Weights)
Given kernels \(\{K_m\}_{m=1}^M\) with per-kernel NW scores \(s_m(x)\), learn \(\lambda\in\Delta^M\) minimizing validation loss \(L(\lambda)\):
1) Initialize \(\lambda_m=1/M\).
2) Compute gradient/subgradient \(g=\nabla_\lambda L\) (e.g., via finite differences or differentiable surrogate).
3) Take step \(\lambda'\leftarrow \lambda - \eta g\).
4) Project to simplex: \(\lambda\leftarrow\Pi_\Delta(\lambda')\) (e.g., sorting-based projection).
5) Stop when \(|\Delta L|<\epsilon\) or `max_iter` reached.

Regularization: add \(\ell_2\) or entropy term to avoid collapse; optionally localized MKL (weights as a function of regime features) with smoothness penalty.

---

# Appendix H — Curvature, Embedding Dimension, and Stability in the Lorentz Model

Cross-reference: deepens §4.3 (Lorentz-model) and Appendix C (stability).

## H.1 Curvature Scaling
Default curvature \(K=-1\). Introduce a temperature \(T>0\) to scale embeddings \(z\leftarrow z/\sqrt{T}\): larger \(T\) flattens distances (stability ↑, separability ↓); smaller \(T\) sharpens distances (stability ↓, separability ↑). Tune \(T\) by CPCV (§7) balancing accuracy and numeric headroom.

## H.2 Embedding Dimension d
Select \(d\) via CPCV based on validation loss/metrics and serving latency (§8). Cost grows as \(\mathcal{O}(d)\) in re-ranking and memory; consider PCA on whitened features before embedding to control \(d\).

## H.3 Contracts & Stability Clamps
- `embed(z: R^d) -> u ∈ H^d`:
  - Pre: finite z; Post: \(u_0=\sqrt{1+\lVert z\rVert^2}\), \(u_{1:d}=z\), \(\langle u,u\rangle_L=-1\), \(u_0>0\); Bounds: \(\lVert z\rVert\le z_{max}\).
- `d_L(u,v) -> R_+`:
  - Pre: \(u,v\in\mathcal{H}^d\); Post: \(d_L\ge 0\), symmetric; Numeric: compute \(s=-\langle u,v\rangle_L\), clamp \(s\leftarrow\max(s,1+\epsilon)\), then \(\operatorname{arcosh}(s)\).
- Error propagation: use Kahan summation in inner product; bound relative error near \(s\approx 1+\epsilon\) via series \(\operatorname{arcosh}(1+\delta)\approx \sqrt{2\delta}\) for \(\delta\ll 1\).

---

# Appendix I — Regime-Aware Indexing, Caching, and Degrade Modes

Cross-reference: extends §8 (Performance Engineering), Appendix E (Observability & Safety), and §3 (Architecture).

## I.1 Dual-Index and Two-Tier Retrieval
- Regime router: select index by session (RTH/ETH) and vol-bucket; features from §5 provide regime signals.
- Two-Tier: Tier-1 FAISS IVF-PQ (GPU) for broad recall → Tier-2 HNSW refine → Lorentz re-rank (§4.4, §3.2).
- Delta-builds: nightly full rebuild; intra-day delta appends every \(\Delta t\) with merge; maintain warm-standby index for hot-swap.

## I.2 Caching & Invalidation
- Cache top-K ids and NW features for recent queries within stable regimes; key by (symbol, regime, grid-hash).
- Invalidate on regime change, index version bump, or parameter grid update; TTL configurable in `model.yaml` (§B.1.2).

## I.3 Backpressure & Degrade Modes
- SLAs (see §8 table): if p95 exceeds budget for \(M\) consecutive windows:
  1) Reduce \(K_{cand}\) (floor at policy minimum).
  2) Skip Lorentz re-rank temporarily; use L2/IP ranking only.
  3) Drop least-informative kernels (retain RQ).
- Emit observability signals: `degrade=on`, stage-level latency spikes, and ANN recall estimates (§E.1). Restore when budget recovered.

---

# Appendix J — Simulator, Sandbox, and Governance

Cross-reference: complements §11 (Deployment & Ops), Appendix E (Governance), and ProjectX/TopstepX references (§13).

## J.1 ProjectX Sandbox Simulator
- Record/replay HTTP/WebSocket interactions; preserve idempotency keys; simulate throttling, partial fills, cancels; configurable latency/error distributions.
- Canary plan: start with 1–5% traffic for \(N\) signals; monitor §E metrics; automatic rollback on gate violations (§D.2).
- Rollback protocols: revert to prior release tag; drain in-flight orders; verify health checks before reopening.

## J.2 Security & Compliance Policies
- Secrets: via ENV/KMS only; no secrets in repo; rotate keys; least-privilege service accounts.
- Artifacts: sign and hash (SHA256 + signature) model, Σ̂, index, and configs; verify on load; maintain SBOM.
- Device/IP checks at startup; enforce “no VPN/VPS/remote” per Topstep policy; log environment fingerprint.
- Audit logs: who/when/what SHA/config hashes; immutable storage with retention policy.

## J.3 Change Control
- Release tags (semver); approvers: ML Eng + Compliance + Trading Ops; update Model Card (Appendix E.3) with metric deltas.
- DR playbooks: loss of venue connectivity, data source failure, API schema change.

---

# Appendix K — Symbol Glossary and Data Lineage

Cross-reference: supports math in §4 and features in §5; complements Data Contracts (§9.2).

## K.1 Symbol Glossary
- \(O,H,L,C,V\): raw open, high, low, close, volume.
- HA: Heikin–Ashi candles (\(\mathrm{HA}\_\mathrm{open},\mathrm{HA}\_\mathrm{high},\mathrm{HA}\_\mathrm{low},\mathrm{HA}\_\mathrm{close}\)).
- \(x\): raw feature vector; \(m\): location; \(\hat{\Sigma}\): Tyler shape; \(z\): whitened vector.
- \(u\in\mathcal{H}^d\): Lorentz embedding; \(\langle\cdot,\cdot\rangle_L\): Lorentz inner product; \(d_L\): geodesic distance.
- \(K_{cand},K\): candidate and final neighbor counts; \(ef\): ef_search; \(np\): nprobe.
- \(\hat f\): NW estimate; \(\lambda\): MKL weights; \(\alpha\): conformal significance.
- \(\tau\): decision threshold; \(C_{FP},C_{FN}\): costs.

## K.2 Data Lineage Table
| Artifact | Derived from | Transform | Versioning |
|---|---|---|---|
| HA bars | OHLCV | §4.1 recurrences | `data.yaml` hash + code SHA |
| Whitened Z | HA features | §4.2 whitening | `model.yaml` (tyler.*), code SHA |
| ANN Index | Z_train | §4.4 index build | index version, params hash |
| Hyperbolic U | Z | §4.3 embed | hyperbolic params hash |
| Kernel scores | NNs, d_L | §4.5 NW/MKL | kernels/MKL params hash |
| Model | F_train,y | §6 training | CatBoost params hash |

---

# Appendix C Supplement — Stage-wise Precision & Fallbacks

Cross-reference: expands Appendix C; ties to §8 budgets and Appendix I degrade modes.

## C.S.1 Precision Choices per Stage
| Stage | Default Precision | Notes | Fallback |
|---|---|---|---|
| Ingest/HA | FP64→FP32 | deterministic transforms | FP64 for audits |
| Tyler fit | FP32 | stable with tol=1e-6 | increase ρ; FP64 if unstable |
| Tyler transform | FP32 | matrix-vector ops | FP16 if validated; else FP32 |
| ANN (dist calc) | FP16 | bandwidth-bound; accumulate FP32 | Recompute slab in FP32 |
| Lorentz inner/acos | FP32 | clamp arg; Kahan sum | FP64 near boundary |
| NW/MKL | FP16 weights, FP32 sum | clamp exponents | FP32 throughout |
| CatBoost predict | FP32 | tree eval | CPU FP32 fallback |

## C.S.2 Fallback Logic
- Detect NaNs/Infs per stage; trigger recompute in higher precision; emit metric `fallback_count{stage=…}`; if rate > threshold, disable FP16 for that stage until next deploy.

---

# Appendix L — Parameter Cookbook & Profiling

Cross-reference: complements §8 (Performance) and §12 (Roadmap & Ablations).

## L.1 Recommended Ranges (Recap + Context)
- HNSW: \(M\in[16,48], ef_{search}\in[64,512]\) → target recall 0.95–0.98.
- FAISS IVF: \(nlist\in[256,4096], nprobe\in[4,64]\); PQ bytes 8–32 per vector.
- Re-rank: \(K_{cand}\in[512,2048], K\in[16,64]\).
- Kernels: RQ \(\alpha\in[0.1,10], \ell\in[0.1,10]\); Periodic \(p\in[5,1000]\) (bars), \(\ell\in[0.1,10]\); Locally-Periodic use product of SE and Periodic.
- CatBoost: depth 6–10; lr 0.02–0.10; iterations 1–5k; l2_leaf_reg 1–10.

## L.2 Pareto-front Profiling Instructions
1) Fix dataset shard and label; freeze whitening and embedding configs.
2) Sweep ANN parameters (ef/nprobe, nlist) to map recall vs latency; record p50/p95; fit the parametric model (§C.2).
3) For promising ANN settings, sweep \(K_{cand},K\) to quantify accuracy lift from re-rank vs cost.
4) For kernels, grid RQ/Periodic/LP and MKL weights to measure marginal gains; prune dominated configs.
5) Train CatBoost on top-3 retrieval/kernel profiles; select with CPCV metrics and serving latency constraints.
6) Deliver a Pareto set with metric table and chosen operating point; archive in Model Card.

---

# Appendix M — CI Quality Gates (Hard, Consolidated)

Cross-reference: consolidates §D.2 and links to §7 (metrics), §8 (budgets), Appendix E (failure tests).

Minimum criteria to pass CI:
- ANN recall ≥ 0.95 on validation queries (with 95% CI reported).
- Per-stage p95 latency ≤ §8 budget (environment-tagged; ANN, re-rank, kernels, predict, risk, API).
- Accuracy ≥ 0.90 × KNN-exact on the same temporal holdout and label definition.
- Coverage: math modules ≥ 90% branch; pipeline ≥ 80%.
- Failure-injection suite: all scenarios in Appendix E.2 must pass (including network chaos, throttling, GPU OOM, VPN detection, duplicate order protection).
 - Lorentz invariants hold; acosh boundary (near 1+ε) uses series path; tests pass.
 - Tyler convergence ≤ 500 iterations; PD (`eigmin>1e-8`) and whitening `trace≈d` checks pass.
 - CV artifact isolation: per-fold artifacts include `fold_id`; no information set overlap.
 - Monotonic ANN recall in `ef_search/nprobe` (non-decreasing across sweeps).
 - End-to-end: total p95 < 25 ms; per-stage p95 within budgets; matches staging SLOs.

---

# Appendix N — Tooling, Versions, and Environment Profiles

Cross-reference: Appendix B (Reproducibility), §11 (Deployment & Ops), Appendix E (Security/Compliance).

## N.1 Environment Profiles
- Local Windows (CPU-first):
  - ANN: `hnswlib` (CPU) or `faiss-cpu`.
  - CatBoost: CPU or GPU if CUDA present; validate with small batches.
  - Note: `faiss-gpu` is best on Linux; on Windows prefer CPU or WSL2.
- WSL2 Ubuntu / Linux Workstation (GPU-capable):
  - ANN: `faiss-gpu` (CUDA), optional IVF-PQ; `hnswlib` fallback.
  - CatBoost: GPU (CUDA) enabled; verify with `model.get_params()` device flags.
- Colab A100 (Training profile):
  - CatBoost GPU training; optional FAISS-GPU indexing; export artifacts; copy to registry.

## N.2 Version Pins (Two Tracks)
- Conservative (maximum compatibility):
  - Python 3.10/3.11; numpy 1.26.x; pandas 2.2.x; scipy 1.11.x; scikit-learn 1.4.x
  - catboost 1.2.5; hnswlib 0.8.0; faiss-cpu 1.7.4; (faiss-gpu 1.7.4 on Linux)
  - cupy 12.x (if used), CUDA 11.8/12.1 matching driver
- Edge (performance/modern):
  - Python 3.11; numpy 2.0.x (verify downstream), pandas 2.2.x; scipy 1.12+
  - catboost ≥1.2.5 (verify numpy 2 compatibility); faiss-cpu/gpu 1.8.0; hnswlib 0.8.0

Decision: default to Conservative for CI; allow Edge in research branches with guardrails.

## N.3 CUDA/CU* Compatibility (GPU Profiles)
- CUDA 11.8/12.1 supported broadly; align FAISS-GPU and CatBoost wheels accordingly.
- Record driver/toolkit versions in run metadata (Appendix B.2).

## N.4 Packaging & Lockfiles
- Maintain `requirements-lock.txt` or conda `environment.lock.yml` with exact hashes for CI runners per OS profile.
- For Colab, export `pip freeze` snapshot per training run and attach to artifacts.

---

# Appendix O — Test Fixtures & Synthetic Data Generators

Cross-reference: §10 (Testing Strategy), Appendix D (Expanded Tests), Appendix M (Gates).

## O.1 Synthetic OHLCV Generator
- Deterministic with `seed`: generate AR(1) price process with jumps; derive OHLCV per bar size; ensure realistic spreads and volumes. Use for HA golden tests and end-to-end rehearsals.

## O.2 Label Generator
- Given HA or raw closes, define horizon T and threshold (e.g., ATR-adjusted); produce binary/ternary labels strictly causally; apply purge/embargo per split.

## O.3 Feature Matrix Fixtures
- Small (n=1k, d=32) for unit tests; Medium (n=100k, d=64) for ANN recall profiling; each with saved seeds and golden statistics (means/vars) post-whitening.

## O.4 ANN Recall Benchmark Set
- Construct with planted neighbors: duplicate each query with slight noise to guarantee a top-1 true neighbor; evaluate recall curves vs ef/nprobe; assert ≥0.95 at selected settings.

## O.5 Golden Files
- HA transforms on a fixed OHLCV sample; Tyler-whitening eigenvalues; Lorentz distances for selected pairs; store as CSV/JSON under `tests/golden/` with checksums.

---

# Appendix P — CI/CD Matrix & Runners

Cross-reference: Appendix M (Gates), §11 (Ops), Appendix E (Governance).

## P.1 CI Matrix
- OS: `ubuntu-latest` (primary), `windows-latest` (compat), optional `macos-latest` (dev only).
- Python: 3.10, 3.11.
- ANN backends: `hnswlib` (all), `faiss-cpu` (Linux/Windows), `faiss-gpu` on self-hosted GPU runner.

## P.2 Stages
- Lint/typecheck (spec and stubs), Unit/Property, Integration (frozen shard), E2E rehearsal (synthetic), Artifact signing & Model Card generation, Compliance checks.

## P.3 GPU Runner (Optional)
- Self-hosted Linux with CUDA 12.1; jobs limited to training benchmarks and FAISS-GPU indexing; tagged and isolated with secrets policy.

---

# Appendix Q — Production Specification Review (Consolidated)

Cross-reference: aligns with §3–§11 and Appendices A–P. This appendix captures mandatory fixes, precision notes, missing components to formalize, modular code suggestions, ops/compliance directives, and Go/No-Go criteria.

## Q.A Mandatory (Blocking) Fixes

| Issue | Description | Required Fix/Action |
|---|---|---|
| ANN ID usage during inference | In `pipeline.run_infer`, `ids` returned by ANN refer to the training memory map, not the sliding window `Z`; `Z[ids]` is incorrect. | API change: ANN must either return neighbor vectors (`search_with_vectors(...)->(ids, base_dists, Z_neighbors)`) or expose `get_vectors(ids)` over `Z_train_memmap/U_train_memmap`. See §Q.C.3. |
| Heikin–Ashi invariants | `assert_invariants` compares HA bars to themselves, not raw O/H/L. | Enforce: `ha_high ≥ max(H_raw, ha_open, ha_close)` and `ha_low ≤ min(L_raw, ha_open, ha_close)` (§4.1). |
| Lorentz distance stability | `d_L` needs stable vectorized implementation. | Compute `s = -⟨u,v⟩_L` in batches; clamp `s ← max(s, 1+ε)` before `acosh`; near boundary use `acosh(1+δ)≈√(2δ)` (§4.3, Appendix C, Appendix H). |
| Tyler whitening guardrails | `tyler_mm_iter` must use stable solves and convergence management. | Replace explicit inverse with Cholesky-based solves; cap influence of outliers; increase ρ adaptively if not converged; final fallback to identity shrinkage (§4.2, Appendix C). |
| Periodic kernel τ semantics | Ambiguity in τ meaning for `periodic(τ)` during neighborhood search. | Specify τ as either time difference in bars `|t_q−t_i|` or a phase indicator (e.g., day-of-week). Provide parameter estimation guidance (§Q.C.1). |
| Purge/Embargo enforcement | Whitening and ANN index must be trained per-fold only. | Enforce artifact isolation per split; no sharing train↔val; add asserts (§7, Appendix D, Appendix I). |
| ANN recall SLO auto-tuner | Recall SLO (≥0.95) needs runtime enforcement. | Add adaptive tuner to increase `ef_search`/`nprobe` when rolling recall drops; log adjustments (Appendix M, §8). |

## Q.B Mathematical Deepening and Precision

- Tyler robust scaling (§4.2): scale-invariant shape; regularization ρ ensures existence/uniqueness (especially n<p); invariants: PD (`eigmin>ε`) and trace normalization per iteration.
- Lorentz geometry (§4.3, Appendix H): curvature temperature `T` rescales `z←z/√T` (effective |K|/T). Use Kahan/pairwise summation for `-⟨u,v⟩_L` to reduce cancellation.
- Nadaraya–Watson (§4.5, Appendix G): compare balloon vs sample-point bandwidths; enforce `sum_w ≥ ε` to avoid division by zero; formal NW: `f̂(x)=Σ K_h(x,x_i)y_i / Σ K_h(x,x_i)`.
- Conformal prediction (§F.3): use Mondrian/class-conditional conformal with label-conditional coverage; cite formal MCP sources; set coverage target and monitor.
- CatBoost (§6): enforce ordered/has_time training settings to reduce leakage in time-series.

## Q.C Essential Missing Components to Formalize

### Q.C.1 Periodic Kernel τ Semantics (spec)
Define τ under two policies:
- Time-difference policy: `τ = |t_q − t_i|` (bars). Parameters: `p` (period in bars) from dominant cycle analysis (e.g., Lomb–Scargle or autocorrelation peaks), `ℓ` tuned via LOOCV (§G.1). Computationally, cache `sin(π τ/p)` for common τ.
- Phase policy: τ is a categorical/phase indicator (e.g., day-of-week, session phase); `period=p` maps to cycle length; ensure τ is known-at-t (§9.2). Provide examples and ensure determinism.

### Q.C.2 External Data Contract (known-at-t)
| Field | Source | Known-at-t? | Late data policy | Notes |
|---|---|---|---|---|
| Macro event release | Econ calendar | Yes (scheduled) | Truncate to scheduled fields; ignore revisions | Use only scheduled time/importance |
| Sentiment tick | Vendor feed | No (latency) | Drop if arrival>Δt window | Do not leak near real-time feeds |
| Holiday/session flags | Exchange | Yes | Deterministic | Part of regime router (§I.1) |

### Q.C.3 ANN API Contract (formal)
- `build(Z_train, config) -> IndexRef` (persist version, params, hashes)
- `search(IndexRef, Z_q, K_cand) -> (ids, base_dists)`
- `get_vectors(IndexRef, ids) -> Z_neighbors` OR `search_with_vectors(IndexRef, Z_q, K_cand) -> (ids, base_dists, Z_neighbors)`
- `save(IndexRef, path) -> None`; `load(path) -> IndexRef`
- `stats(IndexRef) -> {ef, nprobe, recall_rolling, index_version}`

### Q.C.4 Feature Schema Table (hardened)
| Feature | dtype | Range | Normalization | Clip |
|---|---|---|---|---|
| dL_mean | float32 | [0, ∞) | z-score post-whitening | p99 winsorize |
| dL_med | float32 | [0, ∞) | z-score | p99 |
| dL_std | float32 | [0, ∞) | z-score | p99 |
| density | float32 | (0, ∞) | none | max 1e6 |
| score_rq | float32 | [−1,1] | none | [−1,1] |
| score_per | float32 | [−1,1] | none | [−1,1] |
| score_lp | float32 | [−1,1] | none | [−1,1] |
| eff_k | float32 | [1, K] | none | none |
| ha_body | float32 | (−∞,∞) | robust z | p99 |
| regime_vol_bucket | int8 | {0..4} | n/a | n/a |

### Q.C.5 Data QA & Lineage Hooks
- QA checks: duplicate bars, out-of-order timestamps, zero/negative volumes, jumps > X·σ; enforce rejects with reason codes.
- Lineage: stamp artifacts with `{data_hash, config_hash, git_sha, index_version}`; assert lineage in tests (Appendix K.2).

### Q.C.6 OOD/ODD Coverage & RiskGate Behavior
- OOD detection: PSI>0.2, variance shift beyond tolerance, conformal coverage shortfall; define windowed thresholds.
- RiskGate (§E, §A.2.2): PASS (singleton conformal set and p≥τ), THROTTLE (ambiguity or soft violations), BLOCK (hard OOD or policy breach). Log decisions and reasons.

### Q.C.7 Config Typing & Validation
- Adopt Pydantic/Pydantic-Settings for `data.yaml`, `model.yaml`, `risk.yaml` (§B.1); handle unit conversions (ms↔bars) and cross-field constraints; fail fast on invalid configs.

## Q.D Modular Code Review & Suggestions

### Q.D.1 robust_scaling.py
- Tyler iteration via Cholesky solves (no explicit inverse):
```python
L = np.linalg.cholesky(Sigma_k)
Y = scipy.linalg.solve_triangular(L, Z.T, lower=True)  # stability
W = np.sum(Y*Y, axis=0)
```
- Adaptive ρ: increase on non-convergence; fallback to identity shrinkage.

### Q.D.2 hyperbolic.py
- Batch Lorentz distance: `d_L_batch(Uq: (B,d+1), Uk: (B,K,d+1)) -> (B,K)` with pairwise/Kahan summation; clamp `s` before `acosh`.

### Q.D.3 ann_index.py
- Search API:
```python
def search_with_vectors(index, Z_q, K_cand, return_vectors=True):
    ids, d0 = index.search(Z_q, K_cand)
    Z_nn = index.train_store.take(ids) if return_vectors else None
    return ids, d0, Z_nn
```
- FAISS: IVF-PQ with `nlist≈√N`, tune `nprobe` to meet latency SLO (§8).

### Q.D.4 kernels.py
- Report effective neighbors: `eff_k = (Σw)^2 / Σ(w^2)` for diagnostics; clamp exponent range to [−40,40].

### Q.D.5 features.py
- Add “angle proxies” from normalized Lorentz inner products `(-⟨u_i,u_q⟩_L)` as features.

### Q.D.6 cv_protocol.py
- Add asserts to prevent any train leakage around validation windows; enforce purge/embargo strictly (§7).

### Q.D.7 projectx_client.py
- Standardize: idempotency keys, rate limiting, retries with jitter, deadline propagation; structured error taxonomy.

## Q.E Operations, Performance, and Compliance

- Deterministic ANN tuning: maintain a signed Pareto profile of (recall, latency) over (N,d,K) settings; store with artifacts.
- Numerical precision: use FP16 distances with FP32 accumulation; auto-fallback to full FP32 on numeric exceptions (Appendix C Supplement).
- Conformal RiskGate: PASS for singleton confidence set with `p≥τ`; THROTTLE for ambiguity; BLOCK on OOD or coverage failure (§F.3, §Q.C.6).
- Compliance: enforce no-VPN/VPS/remote policy; device/IP verification; kill-switch capability; audit logs (Appendix E.2/E.4).

## Q.H Go/No-Go Checklist (Based on Proposed Changes)

| Area | Status (Current) | Required Action Before Go |
|---|---|---|
| Activation Gate | Misaligned (`0.2` hardcode) | Refactor to annealed `Threshold(epoch)` tuned by ROC/PR (§F.1). |
| Reward Scaling | Inflated (×100) | Replace with volatility-aware z-normalization on returns. |
| Core Math | Tyler/Lorentz/NW | Implement Cholesky solves, Lorentz vectorization, acosh clamping (§Q.A). |
| ANN API | Missing `vectors(ids)` | Implement `search_with_vectors` or `get_vectors` (§Q.C.3). |
| Cross-Validation | Purge/Embargoed | Add asserts; artifact isolation per fold (§Q.A). |
| Data Contracts | Implicit | Finalize Feature Schema table (§Q.C.4) and Data QA asserts (§Q.C.5). |
| Compliance | Theoretical | Implement device/IP checks and kill-switch (§E, §J.2). |
| CI Gate | Theoretical | Certify recall≥0.95, p95 latency≤budget, coverage targets (Appendix M). |

---

# Appendix R — Final Pre-Flight Review: Confirmation and Action Plan

Cross-reference: confirms and operationalizes §3–§11 and Appendices C, G, H, I, M, Q.

## R.1 Verified Components (Ready to Pin)

| Component | Status | Source/Validation |
|---|---|---|
| Heikin–Ashi | Formulae confirmed | Matches standard definitions (§4.1; Wikipedia) |
| Lorentz/Hyperbolic | Distance and hyperboloid model confirmed | Nickel & Kiela (PMLR 2018); §4.3 |
| Regularized Tyler | Existence/uniqueness (ρ), MM iteration, scale-invariance | §4.2; Sun et al. |
| ANN Backends | HNSW and FAISS IVF-PQ are correct high-performance strategies | §4.4; arXiv refs |
| CatBoost | `boosting_type=Ordered` for time-series leakage mitigation; GPU supported | §6; CatBoost docs |
| Compliance | Explicit NO-VPN/VPS/Remote policy | §11; Topstep docs |

## R.2 Ready-to-Pin Golden Defaults

### R.2.1 Data & Labeling
- Timezone: `UTC`; Bar sizes: {1m, 5m, 15m, 60m}.
- Labeler: horizon T for triple-barrier; purge ≥ T; embargo ≥ ceil(T/2).
- QA: strict drop on gaps; allow limited forward-fill of sparse volume only if labels unaffected.

### R.2.2 Robust Whitening (Regularized Tyler)
- ρ = 0.1 (adaptive ↑ to 0.2 if no convergence ≤ 300 iters); tol = 1e-6; max_iter = 500.
- Center: spatial median. Stability: Cholesky solves (no explicit inverse).

### R.2.3 Hyperbolic Geometry (Lorentz)
- Embedding dimension d = 32 (consider 64 later). Guards: ‖z‖ ≤ 10.0; ε_acosh = 1e-6; Kahan/Pairwise sums.

### R.2.4 Approximate Nearest Neighbors (ANN)
- CPU (HNSW): M=32, ef_construction=300, ef_search=128.
- GPU (FAISS IVF-PQ): nlist ≈ √N (256–4096), nprobe=16, PQ=16 bytes.
- Two-stage: K_cand=1024 → Lorentz re-rank → K=32; Recall SLO ≥ 0.95.

### R.2.5 Kernels (NW/MKL)
- τ semantics (Periodic): Time-difference in bars, τ = |Δbars|.
- Parameters: RQ α=1.0, ℓ_RQ ≈ 0.8·median(d_L); Periodic period = 390 (bars/day), ℓ_Per ∈ [0.5, 3].
- MKL init weights [0.5, 0.25, 0.25] (RQ/Per/LP). Guards: exponent clamp [-40,40]; floor Σw ≥ 1e-12.

### R.2.6 CatBoost (GPU)
- depth=8, learning_rate=0.06, iterations=2000, l2_leaf_reg=3, subsample=0.8, loss=Logloss, boosting=Ordered.

### R.2.7 Validation
- CPCV/PurgedKFold with artifacts (Σ̂, index, model) isolated per-fold.
- SLOs: Accuracy/F1/AUC; ANN recall ≥ 0.95.

## R.3 Blocking Issues to Fix Now

1) ANN vector retrieval at inference: update ANN API to return Z_neighbors or provide `get_vectors(ids)`; implement `search_with_vectors` (see §Q.C.3).
2) Periodic τ semantics (final pin): τ = |Δbars| (time-difference policy) and document in §G.1/§Q.C.1.
3) Cross-validation fold isolation: ensure Σ̂ and ANN index are built per-train fold only; forbid sharing across train↔val.
4) Numerical guards: acosh clamp (1+ε), Kahan/Pairwise Lorentz sums, Tyler via Cholesky solves, FP16→FP32 fallback counters/logging.
5) Recall auto-tuner: adapt ef_search/nprobe upward if rolling recall < 0.95; log adjustments.

## R.4 GO / NO-GO

| Verdict | Condition |
|---|---|
| GO | Implement the 5 blockers and pin τ policy to |Δbars| for Periodic kernel. |
| NO-GO | If any blockers on ANN safety, numerical stability, or CV isolation remain. |
| Extra Safety | Run a short Pareto (ef/nprobe × K_cand) on a frozen shard to certify Recall ≥ 0.95 at p95 latency < 8 ms for ANN. |

## R.5 Immediate Next Step
- Prepare skeletal code structure and API contracts for modules covering the blockers (ANN API, Tyler stability, Lorentz vectorization, CV isolation, auto-tuner), using the golden defaults in §R.2 as initial config values.

---

# Appendix S — Staging/Canary Punch‑List

Cross-reference: See `PUNCHLIST.md` for the actionable, module-level tasks, acceptance tests, and owners mapped to the five blockers. This appendix serves as the link from specification to execution (staging/canary GO, production NO‑GO until blockers closed).


---

Status
- Staging/canary (1–5% flow): GO
- Production (100% flow): NO-GO until five blockers are closed

Cross-refs: SPEC sections 3, 9–12; Appendices C, F, G, H, I, L, M, Q, R.

## Blocker 1 - ANN Neighbor Vectors at Inference
Tasks
- ann_index: add `search_with_vectors(Z_q, K_cand) -> (ids, base_dists, Z_neighbors)` and `get_vectors(ids)`.
- pipeline: replace `Z[ids]` with vectors from index store (e.g., memmap); plumb through Lorentz re-rank and kernels.

Acceptance tests
- Unit: `search_with_vectors` returns vectors matching train store for random `ids` (hash equality).
- Integration: E2E re-rank correctness invariant unaffected by sliding window; no IndexError on window shifts.
- Observability: log index version, store path, and `vectors_returned=true` per query.

Owners: ML Eng (A/R), Data Eng (C), Infra (C)

## Blocker 2 - Periodic Kernel Tau Semantics (Time-difference in bars)
Tasks
- model.yaml: add `kernels.periodic.tau_policy: time_diff_bars` and validate.
- kernels: compute `tau = |t_q - t_i|` (bars); cache periodic term `sin(2π tau / period)` or equivalent.
- tests: golden examples; LOOCV plugin choice documented.

Acceptance tests
- Unit: `periodic(tau)` matches expected values for known taus; cache hit ratio > 90% on repeated taus.
- Config: schema validation rejects invalid `tau_policy`.

Owners: ML Eng (A/R)

## Blocker 3 - Fold Isolation (Purge/Embargo) for Tyler & ANN
Tasks
- cv_protocol/pipeline: build scaler and ANN index per-train-fold only; isolate artifact directories per fold; forbid cross-use.
- Add asserts preventing artifact reuse across train/val.

Acceptance tests
- Unit: `no_information_overlap(folds)`; artifact paths include `fold_id` and differ across folds (hash diff).
- Integration: training a second fold does not touch artifacts of the first.

Owners: ML Eng (A/R), Data Eng (C)

## Blocker 4 - Numerical Guards
Tasks
- hyperbolic: pairwise/Kahan sum for `-⟨u,v⟩_L`; clamp `s≥1+1e-6`; series path near boundary.
- robust_scaling: Cholesky solves; adaptive rho; PD+trace checks each iteration.
- metrics/inference: FP16→FP32 auto-fallback counters per stage.

Acceptance tests
- Lorentz: `⟨u,u⟩_L = -1 ± 1e-6`; `d_L(u,u)=0`; boundary acosh tests use series path.
- Tyler: converge ≤ 500 its; `eigmin>1e-8`; `trace≈d` after whitening.
- Fallbacks: inject NaNs in FP16 path + verify FP32 recompute; `fallback_count{stage}` increments.

Owners: ML Eng (A/R)

## Blocker 5 - ANN Recall Auto-Tuner
Tasks
- ann_index/controller: monitor rolling recall; increase `ef_search`/`nprobe` when recall < 0.95; persist chosen params.
- observability: emit `ann_recall`, `ef_search`, `nprobe`, `param_bumps`.

Acceptance tests
- Unit: synthetic planted neighbor set triggers tuner to raise `ef_search` until recall ≥ 0.95.
- Integration: p95 latency remains under budget after tuning within ≤ +20%.

Owners: ML Eng (A/R), Infra (C)

---

## Fast Path to GO - Code/Config Checklist
- [ ] ANN API + recall controller implemented and wired in pipeline.
- [ ] `tau_policy=time_diff_bars` pinned in model.yaml; parameter validation added.
- [ ] Per-fold artifact isolation enforced for scaler and index.
- [ ] Numeric clamps + Kahan/series + FP16→FP32 fallbacks in place; boundary unit tests added.
- [ ] Kernels output `effective_neighbors (eff_k)` and weight-sum floors.

## Tests - Hard Gates (Appendix M)
- [ ] ANN recall ≥ 0.95 (95% CI); monotone in `ef/nprobe`.
- [ ] Lorentz invariants and acosh boundary tests pass.
- [ ] Tyler convergence ≤ 500 its; PD and whitening shape ≈ I.
- [ ] CV artifact isolation; no information set overlap.
- [ ] E2E: p95 per-stage within budget; total p95 < 25 ms; accuracy ≥ 0.90 of KNN-exact.

## Observability
- Emit: `ann_recall`, `ef/nprobe`, `fallback_count{stage}`, `dL_stats`, `eff_k`, `ECE`, `PSI`.
- Degrade flags when p95 over budget (reduce `K_cand` + skip LP/Per + skip re-rank).

## Compliance/Risk (TopstepX/ProjectX)
- Device/IP fingerprinting; hard block on VPN/VPS/remote.
- Idempotency keys; client-side rate limit; jittered retries; deadlines.
- Kill-switch; embargo guard chaos-tested.

## Launch Plan (Post-Blockers)
1) Freeze snapshot + configs + seeds; sign artifacts (model, scaler, index).
2) Pareto cert on frozen shard: (ef/nprobe, K_cand) + choose operating point meeting recall/latency.
3) Canary 1–5% via ProjectX sandbox + real; monitor recall, p95, ECE, coverage.
4) Ramp 5% + 25% + 50% + 100% when gates green for two windows; no compliance hits; no OOD alarms.
5) Rollback tag prepared; previous index/model warm-standby.

## Issue Template
```
Title: [Blocker N] <short description>
Module(s): <module files>
Spec refs: <SPEC.md sections>
Tasks:
- [ ] ...
Acceptance:
- [ ] ...
Owners: Research (A), ML Eng (R), Data Eng (C), Infra (C), Compliance (I)
```

# Module Blueprints & Call Diagrams

Cross-reference: §3 (Architecture), §9 (Repo Layout), Appendix A (Function Matrix), Appendix C (Stability), Appendix M (Quality Gates).

## data_ingest.py — Blueprint

ASCII diagram
```
[sources URIs] -> [load()] -> [validate_schema()] -> [to_utc()] -> [apply_gap_policy()] -> [OHLCV DataFrame]
```

Public API
- load(sources, timezone, bar_sizes) -> OHLCV
- validate_schema(df) -> df (raises on mismatch)
- to_utc(df, tz) -> df
- apply_gap_policy(df, policy) -> df

Usage
- Called by `pipeline.run_*` first; enforces Data Contracts (§9.2) and stamps lineage (Appendix K).

## heikin_ashi.py — Blueprint

ASCII diagram
```
[OHLCV] -> [transform()] -> [HA bars] -> [assert_invariants()]
```

Public API
- transform(ohlcv) -> ha_df
- assert_invariants(ha_df) -> None (raises on violation)

## robust_scaling.py — Blueprint

ASCII diagram
```
[X] -> [fit() -> Σ̂,m] -> [transform()] -> [Z]
               ^
            [tyler_mm_iter()]
```

Public API
- fit(X, rho, tol, max_iter, center=True) -> (Sigma_hat, m)
- transform(X) -> Z
- tyler_mm_iter(Sigma_k, X, rho) -> Sigma_next

## hyperbolic.py — Blueprint

ASCII diagram
```
[Z] -> [embed()] -> [U∈ℋ^d] -> [lorentz_inner()] -> [d_L()]
```

Public API
- embed(Z) -> U in hyperboloid
- lorentz_inner(u, v) -> float
- d_L(u, v, eps=1e-6) -> float

## ann_index.py — Blueprint

ASCII diagram
```
[Z_train] --build--> [IndexRef]
[Z_q, K_cand] --search--> [ids, base_dists]
[Z_q, K_cand] --search_with_vectors--> [ids, base_dists, Z_neighbors]
```

Public API
- build(Z_train, config) -> IndexRef
 - search(IndexRef, Z_q, K_cand) -> (ids, base_dists)
 - search_with_vectors(IndexRef, Z_q, K_cand) -> (ids, base_dists, Z_neighbors)
 - get_vectors(IndexRef, ids) -> Z_neighbors
 - recall_probe(IndexRef, Z_q, ids_exact) -> recall

Notes
- Vector retrieval comes from the index-backed store (e.g., memmap). Emit logs with `index_version`, `store_path`, and `vectors_returned=true` per query.

## kernels.py — Blueprint

ASCII diagram
```
[distances] -> [rq/periodic/locally_periodic] -> [weights] -> [nw_scores()] -> [mkl_blend()]
```

Public API
- rq(r, alpha, ell) -> w
- periodic(tau, period, ell) -> w  # tau semantics from config.tau_policy
- locally_periodic(r, tau, ell_se, period, ell_per) -> w
- nw_scores(y, weights) -> {scores, weightsum, eff_k}
- mkl_blend(scores, lambdas) -> blended_score

Notes
- `tau_policy=time_diff_bars`: compute tau = |t_q - t_i| in bars. Cache `sin(2π tau / period)` or equivalent periodic term; target cache hit ratio > 90% on repeated taus.
- Apply exponent clamps ([-40, 40]); enforce weight-sum floor (>=1e-12). Emit `eff_k` (effective neighbors after floor/thresholding).

## features.py — Blueprint

ASCII diagram
```
[top-K ids, d_L] -> [stats: mean/median/std/density] + [kernel scores + weightsum + eff_k] -> [feature vector]
```

Public API
- assemble(neighbors, dL, kernel_scores, ha_feats) -> feature_row
- schema() -> dict

## model_catboost.py — Blueprint

ASCII diagram
```
[F_train,y] -> [fit()] -> [model] -> [predict_proba()] -> [calibrate()] -> [p̂ calibrated]
```

Public API
- fit(F_train, y, params) -> Model
- predict_proba(model, F_q) -> p
- calibrate(model, F_val, y_val, method) -> CalibratedModel

## cv_protocol.py — Blueprint

ASCII diagram
```
[timeline] -> [purge/embargo] -> [folds] -> [CPCV/PKF]
```

Public API
- split_purged_embargo(T, purge, embargo, k) -> folds
- cpcv_split(T, purge, embargo, k) -> folds

## metrics.py — Blueprint

ASCII diagram
```
[y,ŷ] -> [accuracy/f1/auc] ; [stage timers] -> [latency histograms] ; [ANN] -> [recall_estimate]
```

Public API
- accuracy(y, yhat) -> float; f1(y,yhat) -> float; auc(y, p) -> float
- latency_probe(stage, t_start, t_end) -> record
- recall_estimate(ids_approx, ids_exact) -> float
 - set_gauge(name, value, tags=None) -> None  # e.g., ann_recall, eff_k
 - inc_counter(name, delta=1, tags=None) -> None  # e.g., fallback_count{stage}, param_bumps

## risk_compliance.py — Blueprint

ASCII diagram
```
[ŷ,p,Γ(x)] -> [limits_check()+embargo()] -> [gate()] -> {pass|throttle|block} + audit
```

Public API
- gate(yhat, p, conformal_set, state, caps) -> action
- limits_check(state, caps) -> ok/warn/violation
- news_embargo(now, embargo_cfg) -> ok/block

## projectx_client.py — Blueprint

ASCII diagram
```
[action] -> [sign+rate_limit] -> [submit] -> (ACK|RETRY|FAIL) with idempotency
```

Public API
- place(action, idempotency_key, timeout) -> ack
- cancel(order_id) -> ack
- modify(order_id, params) -> ack

## pipeline.py — Blueprint

ASCII diagram
```
[run_train] : ingest -> HA -> Tyler.fit -> index.build -> features -> CatBoost.fit -> reports
[run_infer] : ingest -> HA -> Tyler.transform -> index.search_with_vectors -> Lorentz re-rank -> kernels -> features -> CatBoost.predict -> gate -> ProjectX
```

Public API
- run_train(config) -> artifacts
- run_infer(config, stream) -> decisions

Notes
- During inference, avoid `Z[ids]` lookups; rely on vectors returned by the index. Log `index_version`, `store_path`, and `vectors_returned=true`.
- warmup(config) -> cache
- shutdown() -> None

---

# Reference Implementations (Non-executable)

Note: These are reference snippets in Python-style pseudocode for clarity. They must be implemented in code with the contracts and guards specified in §4, Appendix C, and Appendix A.

## data_ingest.py
```python
def load(sources: list[str], timezone: str, bar_sizes: list[str]) -> DataFrame:
    """Load OHLCV from sources; validate schema; normalize to UTC.
    Pre: sources non-empty; bar_sizes ⊆ {"1m","5m","15m","60m"}
    Post: df columns = [ts, open, high, low, close, volume], tz=UTC, no gross gaps per policy.
    """
    df = concat_read(sources)
    validate_schema(df)
    df = to_utc(df, timezone)
    df = apply_gap_policy(df, policy="drop")
    return df

def validate_schema(df: DataFrame) -> None:
    required = {"ts","open","high","low","close","volume"}
    assert required.issubset(df.columns)
    assert df["ts"].is_monotonic_increasing

def to_utc(df: DataFrame, tz: str) -> DataFrame:
    df["ts"] = to_timezone(df["ts"], tz, target="UTC")
    return df

def apply_gap_policy(df: DataFrame, policy: str) -> DataFrame:
    if policy == "drop":
        return drop_large_gaps(df)
    elif policy == "interpolate_limited":
        return interpolate_small_gaps(df, limit=2)
    else:
        raise ValueError(policy)
```

## heikin_ashi.py
```python
def transform(ohlcv: DataFrame) -> DataFrame:
    """Compute HA per §4.1; first HA open/close initialized from first bar.
    Invariants: ha_high ≥ max(H, ha_open, ha_close); ha_low ≤ min(L, ha_open, ha_close).
    """
    ha = empty_like(ohlcv)
    ha["ha_close"] = (ohlcv.open + ohlcv.high + ohlcv.low + ohlcv.close)/4.0
    ha_open = [ (ohlcv.open.iloc[0] + ohlcv.close.iloc[0]) / 2.0 ]
    for t in range(1, len(ohlcv)):
        ha_open.append((ha_open[-1] + ha["ha_close"].iloc[t-1]) / 2.0)
    ha["ha_open"] = ha_open
    ha["ha_high"] = maximum(ohlcv.high, maximum(ha.ha_open, ha.ha_close))
    ha["ha_low"]  = minimum(ohlcv.low,  minimum(ha.ha_open, ha.ha_close))
    return concat_ts(ohlcv.ts, ha)

def assert_invariants(ha: DataFrame) -> None:
    assert (ha.ha_high >= ha[["ha_open","ha_close","ha_high"]].max(axis=1)).all()
    assert (ha.ha_low  <= ha[["ha_open","ha_close","ha_low"]].min(axis=1)).all()
```

## robust_scaling.py
```python
def fit(X: NDArray, rho: float=0.1, tol: float=1e-6, max_iter: int=500, center: bool=True) -> tuple[NDArray, NDArray]:
    """Regularized Tyler per §4.2; returns (Σ̂, m)."""
    m = spatial_median(X) if center else zeros(p)
    Z = X - m
    Sigma = eye(p)
    for _ in range(max_iter):
        Sigma_next = tyler_mm_iter(Sigma, Z, rho)
        if fro_norm(Sigma_next - Sigma) <= tol:
            Sigma = Sigma_next; break
        Sigma = Sigma_next
    ensure_pd(Sigma)
    Sigma *= p / trace(Sigma)
    return Sigma, m

def transform(X: NDArray, Sigma: NDArray, m: NDArray) -> NDArray:
    """Whitening: z = Σ̂^{-1/2}(x-m)."""
    L = chol_inv_sqrt(Sigma)
    return (X - m) @ L.T

def tyler_mm_iter(Sigma_k: NDArray, Z: NDArray, rho: float) -> NDArray:
    W = (Z @ inv(Sigma_k) * Z).sum(axis=1)
    A = (p / len(Z)) * (Z.T / clip(W, 1e-12, inf)) @ Z
    Sigma_next = (1 - rho) * A + rho * eye(p)
    Sigma_next *= p / trace(Sigma_next)
    ensure_pd(Sigma_next)
    return Sigma_next
```

## hyperbolic.py
```python
def embed(Z: NDArray) -> NDArray:
    """Lift to hyperboloid: u0=√(1+‖z‖²), u1:d=z; enforce u0>0; clamp ‖z‖."""
    z = Z
    z = clip_norm(z, z_max)
    u0 = sqrt(1.0 + (z*z).sum(axis=1))
    U = concatenate([u0[:,None], z], axis=1)
    return U

def lorentz_inner(u: NDArray, v: NDArray) -> float:
    return -u[0]*v[0] + (u[1:]*v[1:]).sum()

def d_L(u: NDArray, v: NDArray, eps: float=1e-6) -> float:
    s = -lorentz_inner(u, v)
    s = max(s, 1.0 + eps)
    return arcosh(s)
```

## ann_index.py
```python
def build(Z_train: NDArray, config: dict) -> IndexRef:
    """Create HNSW/FAISS index per §8; store params, version hash."""
    backend = config["ann"]["backend"]
    return create_backend_index(backend, Z_train, config)

def search(index: IndexRef, Z_q: NDArray, K_cand: int) -> tuple[NDArray, NDArray]:
    ids, dists = index.search(Z_q, K_cand)
    return ids, dists

def recall_probe(index: IndexRef, Z_q: NDArray, ids_exact: NDArray) -> float:
    ids, _ = search(index, Z_q, len(ids_exact[0]))
    return intersection_over_topk(ids, ids_exact)
```

## kernels.py
```python
def rq(r: NDArray, alpha: float, ell: float) -> NDArray:
    return (1.0 + (r*r) / (2.0*alpha*ell*ell)) ** (-alpha)

def periodic(tau: NDArray, period: float, ell: float) -> NDArray:
    x = sin(pi * tau / period)
    return exp( -2.0*(x*x) / (ell*ell) )

def locally_periodic(r: NDArray, tau: NDArray, ell_se: float, period: float, ell_per: float) -> NDArray:
    return exp( -(r*r) / (2.0*ell_se*ell_se) ) * periodic(tau, period, ell_per)

def nw_scores(y: NDArray, weights: NDArray) -> float:
    w = clip(weights, 0.0, inf)
    s = w.sum()
    s = max(s, 1e-12)
    return (w @ y) / s

def mkl_blend(scores: dict[str,float], lambdas: NDArray) -> float:
    lam = project_simplex(lambdas)
    return sum(lam[i]*scores[k] for i,k in enumerate(sorted(scores)))
```

## features.py
```python
def assemble(neighbors: NDArray, dL: NDArray, kernel_scores: dict, ha_feats: dict) -> dict:
    feats = {}
    feats.update(ha_feats)
    feats["dL_mean"] = dL.mean(); feats["dL_med"] = median(dL); feats["dL_std"] = dL.std()
    feats["density"] = 1.0 / max(dL.mean(), 1e-6)
    feats.update({f"score_{k}": v for k,v in kernel_scores.items()})
    return feats

def schema() -> dict:
    return {"dL_mean": float, "dL_med": float, "dL_std": float, "density": float}
```

## model_catboost.py
```python
def fit(F_train: NDArray, y: NDArray, params: dict) -> Model:
    model = CatBoostClassifier(**params)
    model.fit(F_train, y, verbose=False)
    return model

def predict_proba(model: Model, F_q: NDArray) -> NDArray:
    return model.predict_proba(F_q)[:,1]

def calibrate(p: NDArray, y_val: NDArray, method: str="isotonic") -> Calibrator:
    return fit_calibrator(p, y_val, method)
```

## cv_protocol.py
```python
def split_purged_embargo(T: NDArray, purge: int, embargo: int, k: int) -> list[Fold]:
    # Interval arithmetic; ensure no information overlap
    return build_purged_folds(T, purge, embargo, k)

def cpcv_split(T: NDArray, purge: int, embargo: int, k: int) -> list[Fold]:
    return build_cpcv_folds(T, purge, embargo, k)
```

## metrics.py
```python
def accuracy(y: NDArray, yhat: NDArray) -> float: return (y==yhat).mean()
def f1(y: NDArray, yhat: NDArray) -> float: return f1_score(y, yhat)
def auc(y: NDArray, p: NDArray) -> float: return roc_auc_score(y, p)
def latency_probe(stage: str, t0: float, t1: float) -> None: record_hist(stage, (t1-t0)*1000)
def recall_estimate(ids_approx: NDArray, ids_exact: NDArray) -> float: return intersection_over_topk(ids_approx, ids_exact)
```

## risk_compliance.py
```python
def gate(yhat: int, p: float, conformal_set: set[int], state: dict, caps: dict) -> str:
    limits = limits_check(state, caps)
    if limits == "violation": return "block"
    if (len(conformal_set)>1) or (p < state.get("tau", 0.5)): return "throttle"
    embargo_ok = news_embargo(now(), state.get("embargo", {}))
    return "pass" if embargo_ok else "block"
```

## projectx_client.py
```python
def place(action: dict, idem_key: str, timeout: float) -> dict:
    with rate_limiter():
        for attempt in backoff(max_attempts=M):
            resp = http_post("/order", action, headers={"Idempotency-Key": idem_key}, timeout=timeout)
            if resp.status in {200,201}: return resp.json
            if resp.status in {429,503}: continue  # retry
            raise ApiError(resp.status)
    raise TimeoutError()
```

## pipeline.py
```python
def run_infer(config: dict, stream: Iterable[Bar]) -> Iterable[Decision]:
    idx = load_index(config)
    for window in windows(stream, size=config["window_size"]):
        ohlcv = to_df(window)
        ha = heikin_ashi.transform(ohlcv)
        Z = robust_scaling.transform(extract_feats(ha), Sigma, m)
        ids, _ = ann_index.search(idx, Z[-1:], config["K_cand"])  # last point
        Uq, Uk = hyperbolic.embed(Z[-1:]), hyperbolic.embed(Z[ids])
        dL = compute_dL(Uq, Uk)
        kscores = kernels_and_mkl(Uq, Uk, dL)
        feats = features.assemble(ids, dL, kscores, ha_feats(ha))
        p = model.predict_proba(vectorize(feats))
        action = risk_compliance.gate(int(p>=tau), p, conformal_set(p), state, caps)
        if action=="pass": projectx_client.place(build_order(p), idem_key(), timeout)
        yield Decision(p, action)
```

---

# Test Specifications & Examples

Cross-reference: §10 (Testing Strategy), Appendix D (Expanded tests), Appendix M (Quality Gates).

## tests/test_heikin_ashi.py
```python
def test_ha_recurrence_and_bounds(ohlcv):
    ha = transform(ohlcv)
    assert_invariants(ha)
    assert not ha.isna().any().any()
```

## tests/test_tyler_whitening.py
```python
def test_tyler_converges_pd(X):
    Sigma, m = fit(X, rho=0.1)
    assert eigmin(Sigma) > 1e-8
    Z = transform(X, Sigma, m)
    assert abs(trace(cov(Z)) - Z.shape[1]) < 1e-1
```

## tests/test_hyperbolic_distance.py
```python
def test_embed_and_distance_stability(Z):
    U = embed(Z)
    for u in U: assert abs(lorentz_inner(u,u)+1.0) < 1e-6 and u[0] > 0
    d0 = d_L(U[0], U[0])
    assert abs(d0) < 1e-8
```

```python
def test_acosh_boundary_series_path():
    # Near-boundary: arg ~ 1 + eps
    s = 1.0 + 1e-7
    d_series = acosh_series_approx(s)
    d_direct = acosh_clamped(s)  # implementation clamps and chooses series path
    assert abs(d_series - d_direct) < 1e-6

def test_kahan_sum_vs_naive(U):
    # Construct adversarial vector pairs to stress summation
    naive = lorentz_inner_naive(U[0], U[1])
    kahan = lorentz_inner_kahan(U[0], U[1])
    # Kahan should be closer to high-precision reference
    ref = lorentz_inner_fp64(U[0], U[1])
    assert abs(kahan - ref) < abs(naive - ref)
```

## tests/test_ann_recall.py
```python
def test_recall_monotone(index, Z_q, ids_exact):
    r1 = recall_probe(index.with_params(ef=64), Z_q, ids_exact)
    r2 = recall_probe(index.with_params(ef=256), Z_q, ids_exact)
    assert r2 >= r1 and r2 >= 0.95
```

```python
def test_recall_autotuner_increases_params_when_low_recall(tuner, index, Z_q, ids_exact, latency_budget_ms):
    # Seed with intentionally low settings
    tuner.attach(index.with_params(ef=32, nprobe=4))
    r0 = recall_probe(index, Z_q, ids_exact)
    tuner.step(observed_recall=r0)
    # After step, params should be increased if recall < 0.95
    params = tuner.current_params()
    assert (r0 < 0.95) implies (params.ef > 32 or params.nprobe > 4)

def test_recall_autotuner_latency_guardrail(tuner, index, Z_q, ids_exact, latency_budget_ms):
    tuner.attach(index)
    tuner.configure(latency_budget_ms=latency_budget_ms, max_growth_rate=1.2)
    for _ in range(3):
        r = recall_probe(index, Z_q, ids_exact)
        tuner.step(observed_recall=r)
    # Ensure p95 remains within budget after tuning (≤ +20%)
    assert p95_latency('ANN') <= 1.2 * latency_budget_ms
```

## tests/test_kernels_nw.py
```python
def test_nw_weight_floor(y, r):
    w = rq(r, alpha=1.0, ell=1.0)
    s = nw_scores(y, w)
    assert isfinite(s)
```

```python
def test_periodic_tau_policy_time_diff_bars(t_q, t_i_values, period=390, ell=1.0):
    # tau computed as absolute bar difference
    taus = [abs(t_q - t_i) for t_i in t_i_values]
    w = [periodic(tau, period=period, ell=ell) for tau in taus]
    # golden values for selected taus
    assert_allclose(w[:3], [0.0, 0.42, 0.77], atol=1e-2)  # example placeholders

def test_periodic_cache_hit_ratio(repeated_taus):
    # Implementation should cache periodic terms; hit ratio > 0.9
    hits, total = periodic_cache_stats(repeated_taus)
    assert hits / max(total, 1) > 0.9

def test_nw_outputs_eff_k_and_weightsum(y, weights):
    out = nw_scores(y, weights)
    assert 'weightsum' in out and out['weightsum'] >= 1e-12
    assert 'eff_k' in out and out['eff_k'] >= 1
```

## tests/test_cv_protocol.py
```python
def test_purged_embargo_no_overlap(T):
    folds = split_purged_embargo(T, purge=3, embargo=2, k=5)
    assert no_information_overlap(folds)
```

```python
def test_artifact_paths_include_fold_id(tmp_path, T, config):
    folds = split_purged_embargo(T, purge=3, embargo=2, k=3)
    for fold in folds:
        art = train_fold(fold, out_dir=tmp_path)
        assert f"fold_{fold.id}" in art.scaler_path
        assert f"fold_{fold.id}" in art.index_path

def test_fold_artifacts_isolated(tmp_path, T, config):
    f1, f2, *_ = split_purged_embargo(T, purge=3, embargo=2, k=3)
    a1 = train_fold(f1, out_dir=tmp_path)
    mtime_before = stat(a1.index_path).st_mtime
    a2 = train_fold(f2, out_dir=tmp_path)
    # Training fold 2 must not modify fold 1 artifacts
    assert stat(a1.index_path).st_mtime == mtime_before
```

## tests/test_e2e_pipeline.py
```python
def test_e2e_latency_and_quality(stream, config):
    t0 = now(); results = list(run_infer(config, stream)); t1 = now()
    assert p95_stage_latency(results, "ANN") <= budget("ANN")
    assert ci_recall(results) >= 0.95
```

```python
def test_sliding_window_invariance(stream_sliding, config):
    # Ensure no IndexError on window shifts and re-rank correctness invariant holds
    results = []
    for window in stream_sliding:
        results.append(run_infer(config, window))
    assert not any(r.error for r in results)
    assert invariant_rerank_correctness(results)

def test_ann_logging_fields(stream, config):
    logs = collect_logs(run_infer(config, stream))
    for rec in logs:
        if rec.stage == 'ANN':
            assert 'index_version' in rec and 'store_path' in rec
            assert rec.get('vectors_returned') is True
```

---

# Robustness Playbook (Engineering)

Cross-reference: Appendix C (Stability), Appendix E (Safety), Appendix I (Degrade), Appendix M (Gates).

- Input validation: strict schema checks for all DataFrames (§9.2); type coercion; NaN guards; timezone normalization.
- Numeric clamps: per Appendix C; FP16→FP32 fallback counter; acosh near 1+ε series; Kahan sums for Lorentz inner products.
- ANN resilience: auto-tune `ef/nprobe` when recall dips; rebuild indices nightly; detect staleness via version hash.
- API reliability: retries with exponential backoff + jitter; idempotency keys; deadline propagation; circuit breakers on repeated 5xx.
- Observability: structured logs + trace IDs; per-stage histograms; drift monitors and conformal coverage alarms; degrade-mode flags.
- Security/compliance: ENV/KMS secrets; artifact signatures; device/IP checks (no VPN/VPS); immutable audit logs.

---

# CI & Quality Gates Integration

Cross-reference: Appendix M (consolidated gates) and §7, §8.

- Map tests to gates:
  - ANN recall suite → Recall≥0.95.
  - Latency probes in E2E → p95 per-stage ≤ §8 table.
  - Model eval vs KNN-exact baseline → Accuracy≥0.90×.
  - Coverage report from unit/property tests → math≥90%, pipeline≥80%.
  - Failure-injection suite → all pass (Appendix E.2 scenarios).

- CI stages:
  - Lint/typecheck (non-functional spec).
  - Unit/property tests (math, kernels, hyperbolic, whitening).
  - Integration (ANN→Lorentz→NW→CatBoost on frozen shard).
  - E2E rehearsal with synthetic stream and latency budgets.
  - Artifact checksums and Model Card generation.
- Training start checklist:
  - Data snapshot frozen; Data Contracts validated (§9.2).
  - Label definition locked; Leakage Ledger reviewed (§D.3).
  - Configs pinned (`data.yaml`, `model.yaml`, `risk.yaml`); seeds set.
  - Compute budget allocated; CI pipeline green on unit tests.
- Deployment checklist:
  - CPCV and ablations completed; acceptance criteria met (§7, §12).
  - CI Quality Gates green (§D.2); p95 latency within §8 budget on staging.
  - Model Card completed; approvals recorded; compliance checks (device/IP, no VPN) verified.
  - Kill-switch tested; canary plan defined; rollback path prepared.
