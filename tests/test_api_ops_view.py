from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from api.ops_view import get_portfolio_status
from ops.journal import Journal


def _seed(path: str) -> None:
    j = Journal(path)
    j.record_cash_adjustment(kind="seed", amount=Decimal("250"))
    j.record_order(
        client_order_id="c1", symbol="AAPL", side="BUY",
        notional_dollars=Decimal("20"), stop_loss_price=None,
    )
    j.record_fill(
        order_id="o1", client_order_id="c1", symbol="AAPL", side="BUY",
        quantity=Decimal("0.1"), price=Decimal("200"),
        filled_at=datetime.now(timezone.utc),
        stop_loss_price=Decimal("184"),
    )
    j.close()


@pytest.mark.unit
def test_get_portfolio_status_none_when_unset(monkeypatch):
    monkeypatch.delenv("OPS_JOURNAL_PATH", raising=False)
    assert get_portfolio_status() is None


@pytest.mark.unit
def test_get_portfolio_status_returns_json_safe_snapshot(tmp_path, monkeypatch):
    path = str(tmp_path / "j.sqlite")
    _seed(path)
    monkeypatch.setenv("OPS_JOURNAL_PATH", path)

    status = get_portfolio_status()

    assert status is not None
    assert status["service"]["journal_path"] == path
    assert status["cash"]["cash"] == 230.0  # 250 seed - 20 BUY, as a float not Decimal
    positions = status["positions"]["items"]
    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"
    assert isinstance(positions[0]["quantity"], float)


@pytest.mark.unit
def test_get_portfolio_status_is_json_serializable(tmp_path, monkeypatch):
    import json

    path = str(tmp_path / "j.sqlite")
    _seed(path)
    monkeypatch.setenv("OPS_JOURNAL_PATH", path)

    status = get_portfolio_status()

    # Would raise TypeError if any Decimal/datetime survived _json_safe.
    json.dumps(status)
