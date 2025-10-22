from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class StageStats:
    stage: str
    count: int
    total_ms: float
    mean_ms: float
    max_ms: float


class LatencyRecorder:
    """Utility to capture per-stage latency measurements (Appendix M requirements)."""

    def __init__(self) -> None:
        self._records: dict[str, List[float]] = defaultdict(list)

    def record(self, stage: str, duration_ms: float) -> None:
        self._records[stage].append(float(duration_ms))

    def extend(self, stage: str, durations_ms: Iterable[float]) -> None:
        self._records[stage].extend(float(d) for d in durations_ms)

    def summary(self) -> Dict[str, StageStats]:
        stats: Dict[str, StageStats] = {}
        for stage, durations in self._records.items():
            arr = np.asarray(durations, dtype=np.float64)
            stats[stage] = StageStats(
                stage=stage,
                count=arr.size,
                total_ms=float(np.sum(arr)),
                mean_ms=float(np.mean(arr)),
                max_ms=float(np.max(arr)),
            )
        return stats

    def single_measurements(self) -> Dict[str, float]:
        """Return a simplified {stage: duration_ms} view when stages recorded once."""
        return {stage: float(durations[-1]) for stage, durations in self._records.items() if durations}


class StageTimer:
    """Context manager to time a specific pipeline stage."""

    def __init__(self, recorder: LatencyRecorder, stage: str) -> None:
        self.recorder = recorder
        self.stage = stage
        self._start: float | None = None

    def __enter__(self) -> "StageTimer":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end = perf_counter()
        if self._start is not None:
            duration_ms = (end - self._start) * 1000.0
            self.recorder.record(self.stage, duration_ms)


@contextmanager
def measure_stage(recorder: LatencyRecorder, stage: str):
    timer = StageTimer(recorder, stage)
    try:
        timer.__enter__()
        yield
    finally:
        timer.__exit__(None, None, None)


class FallbackCounters:
    """Simple counter bag to track FP16->FP32/FP64 fallbacks per stage."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def increment(self, stage: str) -> None:
        self._counters[stage] = self._counters.get(stage, 0) + 1

    def as_dict(self) -> dict[str, int]:
        return dict(self._counters)

    def get(self, stage: str) -> int:
        return int(self._counters.get(stage, 0))

