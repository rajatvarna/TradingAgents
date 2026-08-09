"""Unit tests for MCP streamable HTTP client import compatibility."""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_mcp_client_imports_streamable_transport() -> None:
    from ops.broker import mcp_client

    assert callable(mcp_client.streamablehttp_client)
