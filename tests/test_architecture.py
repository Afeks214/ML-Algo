from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "ml_algo"
FORBIDDEN_IMPORT = "ml_algo.pipeline"

MODULES = [
    "heikin_ashi.py",
    "robust_scaling.py",
    "hyperbolic.py",
    "kernels.py",
    "ann_index.py",
]


def test_math_layers_do_not_import_pipeline() -> None:
    for module in MODULES:
        text = (ROOT / module).read_text(encoding="utf-8")
        assert FORBIDDEN_IMPORT not in text, f"{module} must not depend on pipeline"
