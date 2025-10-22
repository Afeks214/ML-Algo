from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    before_ece: float
    after_ece: float | None

    def as_dict(self) -> Dict[str, float | str | None]:
        return {
            "method": self.method,
            "before_ece": self.before_ece,
            "after_ece": self.after_ece,
        }

