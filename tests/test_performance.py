from __future__ import annotations

import time

from ml_algo.performance import LatencyRecorder, StageTimer


def test_latency_recorder_records_stage() -> None:
    recorder = LatencyRecorder()
    with StageTimer(recorder, "stage_a"):
        time.sleep(0.001)
    summary = recorder.summary()
    assert "stage_a" in summary
    assert summary["stage_a"].count == 1
    single = recorder.single_measurements()
    assert "stage_a" in single
