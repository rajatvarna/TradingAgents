"""Bearer-token auth for mutating and secret-touching engine API endpoints.

Auth is opt-in: set ``ENGINE_API_TOKEN`` to require a matching
``Authorization: Bearer <token>`` header on endpoints that depend on
:func:`require_auth`. Leaving it unset preserves today's open-access
behavior, so existing deployments and the built-in HTML consoles (``/ui``,
``/batching``, ``/settings``) keep working until an operator opts in ahead
of exposing this API publicly (see docs/DEPLOYMENT_INTEGRATION_PLAN.md
Phase 1/2).
"""
from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request

from api.deployment_config import get_engine_token


def require_auth(request: Request) -> None:
    """FastAPI dependency: enforce ``ENGINE_API_TOKEN`` if one is configured."""
    token = get_engine_token(os.getenv("ENGINE_API_TOKEN", ""))
    if not token:
        return

    header = request.headers.get("authorization", "")
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer" or not credential:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if not hmac.compare_digest(credential, token):
        raise HTTPException(status_code=401, detail="Invalid token")
