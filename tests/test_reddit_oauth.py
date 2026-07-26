"""Reddit OAuth2 client-credentials support (upstream #1134).

Reddit's WAF only blocks *unauthenticated* JSON search requests (#862); an
app authenticated via the client-credentials grant gets the richer response
back plus a materially higher rate limit than the RSS fallback. These tests
cover: no-credentials falls straight through to RSS (no token request at
all), token fetch/caching/expiry, 401 invalidates the cached token, 429
backs off and retries once, and other failures fall back to RSS.
"""

from __future__ import annotations

import http.client
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import reddit


@pytest.fixture(autouse=True)
def _reset_oauth_cache():
    """Each test starts with a clean module-level token cache."""
    reddit._invalidate_oauth_token()
    yield
    reddit._invalidate_oauth_token()


def _resp(read_fn):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return read_fn()
    return _Resp()


def _json_resp(payload):
    return _resp(lambda: json.dumps(payload).encode())


def _token_resp(token="tok-123", expires_in=3600):
    return _json_resp({"access_token": token, "expires_in": expires_in})


def _search_resp(posts):
    children = [{"data": p} for p in posts]
    return _json_resp({"data": {"children": children}})


@pytest.mark.unit
class TestNoCredentialsSkipsOAuth:
    def test_missing_credentials_returns_none_without_network_call(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        with patch.object(reddit, "urlopen") as op:
            token = reddit._get_oauth_token(5.0)
        assert token is None
        op.assert_not_called()

    def test_fetch_subreddit_goes_to_rss_without_credentials(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        with patch.object(reddit, "_fetch_subreddit_rss", return_value=["sentinel"]) as rss, \
             patch.object(reddit, "urlopen", side_effect=AssertionError("must not touch network")):
            out = reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out == ["sentinel"]

    def test_partial_credentials_returns_none(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "id-only")
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        assert reddit._reddit_oauth_credentials() is None


@pytest.mark.unit
class TestTokenFetchAndCaching:
    def test_token_request_uses_basic_auth_and_client_credentials_grant(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        with patch.object(reddit, "urlopen", return_value=_token_resp()) as op:
            token = reddit._get_oauth_token(5.0)
        assert token == "tok-123"
        req = op.call_args.args[0]
        assert req.get_header("Authorization").startswith("Basic ")
        assert req.data == b"grant_type=client_credentials"

    def test_second_call_within_expiry_uses_cache(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        with patch.object(reddit, "urlopen", return_value=_token_resp()) as op:
            first = reddit._get_oauth_token(5.0)
            second = reddit._get_oauth_token(5.0)
        assert first == second == "tok-123"
        op.assert_called_once()

    def test_expired_token_is_refetched(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        with patch.object(reddit, "urlopen", return_value=_token_resp(expires_in=60)):
            reddit._get_oauth_token(5.0)
        # Force the cached token to look expired.
        reddit._oauth_expires_at = 0.0
        with patch.object(reddit, "urlopen", return_value=_token_resp(token="tok-456")) as op:
            token = reddit._get_oauth_token(5.0)
        assert token == "tok-456"
        op.assert_called_once()

    def test_token_request_failure_returns_none(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        with patch.object(reddit, "urlopen", side_effect=OSError("network down")):
            token = reddit._get_oauth_token(5.0)
        assert token is None


@pytest.mark.unit
class TestOAuthSearch:
    def test_successful_search_returns_richer_data(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        posts = [{"title": "NVDA pops", "score": 500, "num_comments": 20}]
        with patch.object(reddit, "urlopen", side_effect=[_token_resp(), _search_resp(posts)]) as op:
            out = reddit._fetch_subreddit_oauth("NVDA", "stocks", 5, 5.0)
        assert out == posts
        # Second call carries the bearer token.
        search_req = op.call_args_list[1].args[0]
        assert search_req.get_header("Authorization") == "Bearer tok-123"

    def test_401_invalidates_token_and_falls_back_to_rss(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        err = HTTPError("url", 401, "Unauthorized", {}, None)
        with patch.object(reddit, "urlopen", side_effect=[_token_resp(), err]), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=["rss-fallback"]) as rss:
            out = reddit._fetch_subreddit_oauth("NVDA", "stocks", 5, 5.0)
        assert out == ["rss-fallback"]
        rss.assert_called_once()
        assert reddit._oauth_token is None  # cache cleared for next call

    def test_429_backs_off_and_retries_once(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        err = HTTPError("url", 429, "Too Many Requests", {"Retry-After": "3"}, None)
        posts = [{"title": "retry worked", "score": 1, "num_comments": 0}]
        with patch.object(reddit, "urlopen", side_effect=[_token_resp(), err, _search_resp(posts)]), \
             patch.object(reddit.time, "sleep") as slept:
            out = reddit._fetch_subreddit_oauth("NVDA", "stocks", 5, 5.0)
        assert out == posts
        slept.assert_called_once_with(3.0)

    def test_other_failure_falls_back_to_rss(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        with patch.object(reddit, "urlopen",
                          side_effect=[_token_resp(), http.client.IncompleteRead(b"")]), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=["rss-fallback"]) as rss:
            out = reddit._fetch_subreddit_oauth("NVDA", "stocks", 5, 5.0)
        assert out == ["rss-fallback"]
        rss.assert_called_once()

    def test_fetch_subreddit_uses_oauth_when_credentials_configured(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        with patch.object(reddit, "_fetch_subreddit_oauth", return_value=["oauth-result"]) as oauth:
            out = reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0)
        oauth.assert_called_once()
        assert out == ["oauth-result"]


@pytest.mark.unit
class TestThreadSafety:
    def test_concurrent_get_oauth_token_calls_fetch_once(self, monkeypatch):
        """A lock around the cache read/refresh means concurrent callers
        during a cold cache converge on a single token fetch, not a race
        where each thread independently re-authenticates."""
        import threading

        monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
        results = []

        def fetch():
            results.append(reddit._get_oauth_token(5.0))

        with patch.object(reddit, "urlopen", return_value=_token_resp()) as op:
            threads = [threading.Thread(target=fetch) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert results == ["tok-123"] * 8
        assert op.call_count == 1
