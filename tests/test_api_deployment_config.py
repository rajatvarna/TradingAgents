from __future__ import annotations

import logging

import pytest

from api.deployment_config import get_engine_token, parse_cors_origins


@pytest.mark.unit
def test_parse_cors_origins_defaults_when_empty():
    assert parse_cors_origins("") == ["http://localhost:3000"]


@pytest.mark.unit
def test_parse_cors_origins_defaults_when_blank():
    assert parse_cors_origins("   ") == ["http://localhost:3000"]


@pytest.mark.unit
def test_parse_cors_origins_uses_custom_default():
    assert parse_cors_origins("", default="https://example.com") == ["https://example.com"]


@pytest.mark.unit
def test_parse_cors_origins_splits_and_strips():
    result = parse_cors_origins("https://a.example.com, https://b.example.com ,https://c.example.com")
    assert result == ["https://a.example.com", "https://b.example.com", "https://c.example.com"]


@pytest.mark.unit
def test_parse_cors_origins_rejects_missing_scheme():
    with pytest.raises(ValueError, match="not a valid origin"):
        parse_cors_origins("example.com")


@pytest.mark.unit
def test_parse_cors_origins_rejects_missing_host():
    with pytest.raises(ValueError, match="not a valid origin"):
        parse_cors_origins("https://")


@pytest.mark.unit
def test_parse_cors_origins_rejects_path_suffix():
    with pytest.raises(ValueError, match="must be an origin only"):
        parse_cors_origins("https://example.com/some/path")


@pytest.mark.unit
def test_parse_cors_origins_allows_bare_trailing_slash():
    assert parse_cors_origins("https://example.com/") == ["https://example.com/"]


@pytest.mark.unit
def test_get_engine_token_returns_none_when_unset():
    assert get_engine_token("") is None
    assert get_engine_token("   ") is None


@pytest.mark.unit
def test_get_engine_token_strips_whitespace():
    assert get_engine_token("  a-reasonably-long-token-value  ") == "a-reasonably-long-token-value"


@pytest.mark.unit
def test_get_engine_token_warns_on_short_token(caplog):
    with caplog.at_level(logging.WARNING):
        token = get_engine_token("short")
    assert token == "short"
    assert any("shorter than" in record.message for record in caplog.records)


@pytest.mark.unit
def test_get_engine_token_no_warning_on_long_token(caplog):
    with caplog.at_level(logging.WARNING):
        token = get_engine_token("a" * 32)
    assert token == "a" * 32
    assert not any("shorter than" in record.message for record in caplog.records)
