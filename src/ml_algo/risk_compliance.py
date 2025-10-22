from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RiskCaps:
    """Risk guard configuration (SPEC §11.1)."""

    daily_loss: float
    max_position: int
    news_embargo_min: int


@dataclass(frozen=True)
class CompliancePolicy:
    """Compliance configuration toggles."""

    require_device_check: bool = True
    disallow_vpn: bool = True


@dataclass
class RiskState:
    """Mutable state tracked during runtime (positions, PnL, kill switch)."""

    cumulative_pnl: float = 0.0
    open_positions: int = 0
    kill_switch: bool = False
    last_news_timestamp: datetime | None = None
    vpn_detected: bool = False
    device_verified: bool = True

    def mark_news(self, ts: datetime) -> None:
        self.last_news_timestamp = ts.replace(tzinfo=timezone.utc)


def _embargo_active(state: RiskState, caps: RiskCaps, now: datetime) -> bool:
    if state.last_news_timestamp is None:
        return False
    delta = now - state.last_news_timestamp
    embargo = timedelta(minutes=caps.news_embargo_min)
    return delta < embargo


def _is_within_caps(prob: float, caps: RiskCaps, state: RiskState) -> bool:
    projected_position = state.open_positions + (1 if prob >= 0.5 else 0)
    if abs(projected_position) > caps.max_position:
        return False
    if state.cumulative_pnl <= -abs(caps.daily_loss):
        return False
    return True


def evaluate_compliance(policy: CompliancePolicy, state: RiskState) -> list[str]:
    violations: list[str] = []
    if policy.disallow_vpn and state.vpn_detected:
        violations.append("vpn_detected")
    if policy.require_device_check and not state.device_verified:
        violations.append("device_not_verified")
    return violations


def gate(
    label: int,
    probability: float,
    state: RiskState,
    caps: RiskCaps,
    policy: CompliancePolicy,
    *,
    now: datetime | None = None,
) -> tuple[str, list[str]]:
    """
    Risk/compliance gate returning action ("pass" | "block") and reasons.
    """
    now = now or datetime.now(tz=timezone.utc)
    if state.kill_switch:
        return "block", ["kill_switch_active"]

    reasons: list[str] = []
    if not _is_within_caps(probability, caps, state):
        reasons.append("risk_limits_exceeded")
    if _embargo_active(state, caps, now):
        reasons.append("news_embargo_active")
    reasons.extend(evaluate_compliance(policy, state))

    if reasons:
        return "block", reasons

    # Update state conservatively for bookkeeping
    if label != 0:
        state.open_positions += 1 if probability >= 0.5 else -1
    return "pass", reasons


def reset_daily(state: RiskState) -> None:
    state.cumulative_pnl = 0.0
    state.open_positions = 0
    state.kill_switch = False
