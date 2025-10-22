from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import uuid


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


class ProjectXHttpClient(ProjectXClient):
    """
    Real HTTP client for ProjectX. Uses environment for secrets if not provided.

    This client is optional and requires the `requests` package.
    """

    def __init__(self, config: ProjectXConfig | None = None) -> None:
        if config is None:
            endpoint = os.environ.get("PROJECTX_ENDPOINT", "")
            api_key = os.environ.get("PROJECTX_API_KEY", "")
            config = ProjectXConfig(endpoint=endpoint, api_key=api_key)
        super().__init__(config)

    def place_order(self, order: Order, *, idem_key: str | None = None, timeout: Optional[float] = None) -> dict:
        # Idempotency key
        idem = idem_key or str(uuid.uuid4())
        if idem in self._idempotency_cache:
            return self._idempotency_cache[idem]
        self._guard_rate_limit()
        try:
            import requests  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("requests is required for ProjectXHttpClient") from e
        url = self.config.endpoint.rstrip("/") + "/orders"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Idempotency-Key": idem,
            "Content-Type": "application/json",
        }
        payload = {
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": order.price,
        }
        to = timeout if timeout is not None else self.config.timeout_seconds
        resp = requests.post(url, json=payload, headers=headers, timeout=to)
        resp.raise_for_status()
        ack = resp.json()
        self._idempotency_cache[idem] = ack
        return ack


def make_projectx_client(use_http: bool | None = None) -> ProjectXClient:
    """Factory that returns an HTTP client if explicitly requested via env or flag."""
    if use_http is None:
        env_flag = os.environ.get("PROJECTX_USE_HTTP", "0").lower() in {"1", "true", "yes"}
    else:
        env_flag = use_http
    if env_flag:
        return ProjectXHttpClient()
    # Default to local stub client with env-provided config if present
    endpoint = os.environ.get("PROJECTX_ENDPOINT", "https://projectx.invalid")
    api_key = os.environ.get("PROJECTX_API_KEY", "test")
    return ProjectXClient(ProjectXConfig(endpoint=endpoint, api_key=api_key))
