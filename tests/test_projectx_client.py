from __future__ import annotations

import pytest

from ml_algo.projectx_client import Order, ProjectXClient, ProjectXConfig, RateLimitExceeded


def test_projectx_client_idempotency_and_rate_limit(monkeypatch) -> None:
    config = ProjectXConfig(endpoint="https://api.example", api_key="test", rate_limit_qps=100.0)
    client = ProjectXClient(config)
    order = Order(symbol="NQ", side="buy", quantity=1, price=100.0)
    ack1 = client.place_order(order, idem_key="abc")
    ack2 = client.place_order(order, idem_key="abc")
    assert ack1 == ack2

    client = ProjectXClient(ProjectXConfig(endpoint="https://api.example", api_key="test", rate_limit_qps=1.0))
    client.place_order(order, idem_key="id1")
    with pytest.raises(RateLimitExceeded):
        client.place_order(order, idem_key="id2")
