from __future__ import annotations

from ml_algo.quality_gates import QualityThresholds, evaluate_quality_gates


def test_quality_gates_pass() -> None:
    metrics = {"accuracy_mean": 0.93, "ann_recall": 0.96, "latency_speedup": 0.98}
    result = evaluate_quality_gates(metrics)
    assert result.passed
    assert not result.messages


def test_quality_gates_failures() -> None:
    metrics = {"accuracy_mean": 0.85, "ann_recall": 0.90, "latency_speedup": 0.80}
    thresholds = QualityThresholds(min_accuracy=0.90, min_recall=0.95, min_latency_speedup=0.95)
    result = evaluate_quality_gates(metrics, thresholds=thresholds)
    assert not result.passed
    assert len(result.messages) == 3
