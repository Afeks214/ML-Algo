from __future__ import annotations

from datetime import datetime, timezone

from ml_algo.risk_compliance import (
    CompliancePolicy,
    RiskCaps,
    RiskState,
    gate,
)


def test_gate_blocks_on_kill_switch() -> None:
    state = RiskState(kill_switch=True)
    caps = RiskCaps(daily_loss=1500.0, max_position=1, news_embargo_min=60)
    policy = CompliancePolicy()
    action, reasons = gate(1, 0.8, state, caps, policy)
    assert action == "block"
    assert "kill_switch_active" in reasons


def test_gate_pass_when_within_limits() -> None:
    state = RiskState(cumulative_pnl=100.0, open_positions=0)
    caps = RiskCaps(daily_loss=1500.0, max_position=2, news_embargo_min=60)
    policy = CompliancePolicy()
    now = datetime.now(tz=timezone.utc)
    action, reasons = gate(1, 0.6, state, caps, policy, now=now)
    assert action == "pass"
    assert not reasons


def test_gate_blocks_on_embargo_and_compliance() -> None:
    state = RiskState(
        cumulative_pnl=0.0,
        open_positions=0,
        last_news_timestamp=datetime.now(tz=timezone.utc),
        vpn_detected=True,
        device_verified=False,
    )
    caps = RiskCaps(daily_loss=1500.0, max_position=2, news_embargo_min=120)
    policy = CompliancePolicy()
    action, reasons = gate(1, 0.8, state, caps, policy, now=datetime.now(tz=timezone.utc))
    assert action == "block"
    assert "news_embargo_active" in reasons
    assert "vpn_detected" in reasons
    assert "device_not_verified" in reasons
