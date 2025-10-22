from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class QualityThresholds:
    """Acceptance criteria derived from SPEC §10/§12."""

    min_accuracy: float = 0.90
    min_recall: float = 0.95
    min_latency_speedup: float = 0.95


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    messages: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {"passed": self.passed, "messages": list(self.messages)}


def evaluate_quality_gates(
    metrics: Dict[str, float],
    *,
    thresholds: QualityThresholds | None = None,
) -> QualityGateResult:
    """
    Compare aggregated metrics against thresholds.

    Expected metric keys:
        - accuracy_mean
        - ann_recall
        - latency_speedup
    """
    thresholds = thresholds or QualityThresholds()
    messages: List[str] = []

    accuracy = metrics.get("accuracy_mean")
    if accuracy is None or accuracy < thresholds.min_accuracy:
        messages.append(f"Accuracy {accuracy!r} below minimum {thresholds.min_accuracy}")

    recall = metrics.get("ann_recall")
    if recall is None or recall < thresholds.min_recall:
        messages.append(f"ANN recall {recall!r} below minimum {thresholds.min_recall}")

    latency_speedup = metrics.get("latency_speedup")
    if latency_speedup is None or latency_speedup < thresholds.min_latency_speedup:
        messages.append(
            f"Latency speedup {latency_speedup!r} below minimum {thresholds.min_latency_speedup}"
        )

    passed = not messages
    return QualityGateResult(passed=passed, messages=messages)
