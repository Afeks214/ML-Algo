# Architecture Runbook (SPEC §9)

## Layering Rules
- Math primitives (`heikin_ashi`, `robust_scaling`, `hyperbolic`, `kernels`) remain side-effect free; do not import pipeline or client layers.
- Retrieval layer (`ann_index`) depends only on numpy + internal utils; returns immutable `IndexRef` structures.
- Feature assembly (`features`) consumes Phase 4 outputs and kernel params only; any pipeline imports are TYPE_CHECKING guards.
- Pipeline orchestrator (`pipeline`) is the sole module allowed to compose phases and manage artifacts.

## Module Responsibilities
| Module | Responsibility | Owner (RACI) |
| --- | --- | --- |
| `data_ingest` | Schema alignment, gap policy enforcement | Data Eng (R), ML Eng (A) |
| `heikin_ashi` | HA recurrence + invariants | Research (A), ML Eng (R) |
| `robust_scaling` | Tyler whitening fit/transform | ML Eng (R) |
| `ann_index` | ANN API, recall tuner | ML Eng (R), Infra (C) |
| `hyperbolic` | Lorentz embeds + distances | Research (A) |
| `kernels` | Kernel ensemble + NW stats | Research (A) |
| `features` | Feature assembly, diagnostics | ML Eng (R) |
| `model_catboost` | CatBoost harness + calibration | ML Eng (R) |
| `validation` | CPCV evaluation, reporting | ML Eng (R) |
| `performance` | Latency instrumentation | Infra (R) |

## Runbooks
- Refer to `docs/api_contracts.md` for interface contracts.
- See `docs/runbooks/reproducibility.md` for metadata & artifact steps.
- Deployment and risk ops covered in Phase 11 runbook (to be appended).

## Code Reviews Checklist
1. Verify new modules respect layering (no upward imports).
2. Confirm schema validators updated for any new data sources.
3. Ensure observability fields (timings, recall, calibration) remain populated.
4. Require tests covering new behaviours (unit/property/integration).
