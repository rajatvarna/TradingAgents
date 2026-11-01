from __future__ import annotations

import tradingagents.dataflows.reddit as reddit


def test_fetch_reddit_posts_returns_message_when_all_sources_empty(monkeypatch):
    def fake_fetch_subreddit(ticker, sub, limit, timeout, **kwargs):
        return []

    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", fake_fetch_subreddit)
    monkeypatch.setattr(reddit.time, "sleep", lambda *_args, **_kwargs: None)

    result = reddit.fetch_reddit_posts(
        "ULVR.L",
        subreddits=("stocks", "investing"),
        limit_per_sub=5,
        timeout=1.0,
        inter_request_delay=0.0,
    )

    assert result.strip()
    assert "No Reddit discussion posts were available matching ULVR.L" in result
    assert "rate-limited" in result
    assert "temporarily unavailable" in result


def test_fetch_reddit_posts_marks_empty_subreddit_blocks(monkeypatch):
    calls = []

    def fake_fetch_subreddit(ticker, sub, limit, timeout, **kwargs):
        calls.append(sub)
        return [
            {
                "title": "ULVR discussion",
                "score": None,
                "num_comments": None,
                "created_utc": None,
                "selftext": "Some discussion body",
                "source": "rss",
                "subreddit": "investing",
            }
        ]

    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", fake_fetch_subreddit)
    monkeypatch.setattr(reddit.time, "sleep", lambda *_args, **_kwargs: None)

    result = reddit.fetch_reddit_posts(
        "ULVR.L",
        subreddits=("stocks", "investing"),
        limit_per_sub=5,
        timeout=1.0,
        inter_request_delay=0.0,
    )

    assert "r/stocks: <no posts found mentioning ULVR.L" in result
    assert "r/investing — 1 posts" in result
    assert calls == ["stocks+investing"]  # one combined request
