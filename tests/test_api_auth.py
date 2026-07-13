from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.auth import require_auth


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.post("/protected", dependencies=[Depends(require_auth)])
    async def protected():
        return {"ok": True}

    return app


@pytest.mark.unit
def test_open_when_token_unset(monkeypatch):
    monkeypatch.delenv("ENGINE_API_TOKEN", raising=False)
    client = TestClient(_make_app())
    resp = client.post("/protected")
    assert resp.status_code == 200


@pytest.mark.unit
def test_rejects_missing_header_when_token_set(monkeypatch):
    monkeypatch.setenv("ENGINE_API_TOKEN", "s3cr3t")
    client = TestClient(_make_app())
    resp = client.post("/protected")
    assert resp.status_code == 401


@pytest.mark.unit
def test_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("ENGINE_API_TOKEN", "s3cr3t")
    client = TestClient(_make_app())
    resp = client.post("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


@pytest.mark.unit
def test_accepts_correct_token(monkeypatch):
    monkeypatch.setenv("ENGINE_API_TOKEN", "s3cr3t")
    client = TestClient(_make_app())
    resp = client.post("/protected", headers={"Authorization": "Bearer s3cr3t"})
    assert resp.status_code == 200


@pytest.mark.unit
def test_rejects_non_bearer_scheme(monkeypatch):
    monkeypatch.setenv("ENGINE_API_TOKEN", "s3cr3t")
    client = TestClient(_make_app())
    resp = client.post("/protected", headers={"Authorization": "Basic s3cr3t"})
    assert resp.status_code == 401
