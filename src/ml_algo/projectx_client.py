from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ProjectXConfig:
    endpoint: str
    api_key: str
    rate_limit_qps: float = 5.0
    timeout_seconds: float = 2.0


@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    quantity: int
    price: float


class RateLimitExceeded(Exception):
    pass


class ProjectXClient:
    """
    Thin client implementing idempotency and local rate limiting for ProjectX.
    """

    def __init__(self, config: ProjectXConfig) -> None:
        self.config = config
        self._last_call_ts: float | None = None
        self._idempotency_cache: Dict[str, dict] = {}

    def _guard_rate_limit(self) -> None:
        if self.config.rate_limit_qps <= 0:
            return
        now = time.time()
        interval = 1.0 / self.config.rate_limit_qps
        if self._last_call_ts is not None and now - self._last_call_ts < interval:
            raise RateLimitExceeded("Rate limit exceeded")
        self._last_call_ts = now

    def place_order(self, order: Order, *, idem_key: str, timeout: Optional[float] = None) -> dict:
        """
        Simulate order submission; returns acknowledgement payload.

        The function respects local rate limiting and idempotency caching.
        """
        if idem_key in self._idempotency_cache:
            return self._idempotency_cache[idem_key]

        self._guard_rate_limit()

        ack = {
            "status": "accepted",
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": order.price,
            "timeout": timeout if timeout is not None else self.config.timeout_seconds,
        }
        self._idempotency_cache[idem_key] = ack
        return ack

    def reset_rate_limit(self) -> None:
        self._last_call_ts = None

