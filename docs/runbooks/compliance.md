# Compliance & Risk Runbook

## Pre-Trade Controls
- Device verification log reviewed; set `RiskState.device_verified=True` only after hardware attestation.
- VPN detection pipeline updates `RiskState.vpn_detected`; block trade if true.
- News feed ingestion should call `RiskState.mark_news(ts)` whenever embargo-triggering event occurs.

## Runtime Monitoring
- Track risk gate outputs (`gate(...)[0/1]`) with structured logs; alert on repeated `risk_limits_exceeded` or `vpn_detected` reasons.
- Daily reset: invoke `reset_daily(state)` at session start; confirm cumulative PnL zeroed.
- Loss limits: once `cumulative_pnl <= -daily_loss`, kill switch automatically blocks.

## Incident Response
1. Immediate block: toggle kill switch and confirm downstream logs show `kill_switch_active`.
2. API anomalies (ProjectX): reduce `ProjectXConfig.rate_limit_qps`, use idempotency keys to avoid duplicates.
3. Compliance breach: persist request/response, escalate to compliance team, disable trading until resolved.

## Audit Trail
- Store risk/compliance decisions with timestamp, order metadata, and reasons.
- Maintain monthly attestation that no VPN/remote access breaches occurred.
