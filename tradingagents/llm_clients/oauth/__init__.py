"""OAuth "Sign in with ChatGPT" (flusso Codex) per TradingAgents.

Espone il login PKCE, lo store dei token con refresh, l'auth httpx e le
costanti del backend Codex. Vedi docs/superpowers/specs per il protocollo
verificato dal sorgente openai/codex.
"""
from .auth import CodexOAuth
from .flow import OAuthLoginError, exchange_code, login
from .models import (
    ModelAvailabilityCache,
    available_models,
    discover_available_models,
)
from .payload import apply_codex_payload_constraints
from .pkce import (
    CODEX_BASE_URL,
    CODEX_DEFAULT_HEADERS,
    OAUTH_AUTHORIZE_URL,
    OAUTH_CLIENT_ID,
    OAUTH_REDIRECT_FALLBACK_PORT,
    OAUTH_REDIRECT_PORT,
    OAUTH_SCOPE,
    OAUTH_TOKEN_URL,
    build_authorize_url,
    generate_pkce_pair,
    generate_state,
    redirect_uri_for_port,
)
from .store import (
    OAuthError,
    OAuthNotLoggedIn,
    OAuthRefreshFailed,
    OAuthTokenStore,
    StoredTokens,
    default_store_path,
)


def ensure_token(store: "OAuthTokenStore | None" = None) -> "StoredTokens":
    """Carica i token; se scaduti fa refresh; se assenti solleva OAuthNotLoggedIn."""
    store = store or OAuthTokenStore()
    tokens = store.load()
    if tokens.is_expired():
        tokens = store.refresh()
    return tokens


__all__ = [
    "OAUTH_CLIENT_ID",
    "OAUTH_AUTHORIZE_URL",
    "OAUTH_TOKEN_URL",
    "OAUTH_REDIRECT_PORT",
    "OAUTH_REDIRECT_FALLBACK_PORT",
    "OAUTH_SCOPE",
    "CODEX_BASE_URL",
    "CODEX_DEFAULT_HEADERS",
    "generate_pkce_pair",
    "generate_state",
    "build_authorize_url",
    "redirect_uri_for_port",
    "OAuthTokenStore",
    "StoredTokens",
    "OAuthError",
    "OAuthNotLoggedIn",
    "OAuthRefreshFailed",
    "default_store_path",
    "CodexOAuth",
    "apply_codex_payload_constraints",
    "login",
    "exchange_code",
    "OAuthLoginError",
    "ensure_token",
    "available_models",
    "discover_available_models",
    "ModelAvailabilityCache",
]
