from __future__ import annotations

import pytest
from tradingagents.dataflows.yfinance_news import _extract_article_data

@pytest.mark.unit
def test_extract_article_data_no_truncation():
    article = {
        "title": "Test Title",
        "summary": "This is a relatively short summary that should not be truncated.",
        "publisher": "Test Publisher",
        "link": "http://example.com"
    }
    # With max_summary_chars=0, it should not be truncated
    data = _extract_article_data(article, max_summary_chars=0)
    assert data["summary"] == "This is a relatively short summary that should not be truncated."

@pytest.mark.unit
def test_extract_article_data_basic_truncation():
    article = {
        "title": "Test Title",
        "summary": "This is a longer summary that will definitely exceed twenty characters.",
        "publisher": "Test Publisher",
        "link": "http://example.com"
    }
    # With max_summary_chars=20, it should truncate at word boundary before 20 chars and add "..."
    data = _extract_article_data(article, max_summary_chars=20)
    # "This is a longer summary" is 24 chars. It should cut at "This is a" or "This is a longer..."
    assert len(data["summary"]) <= 23  # 20 + 3 for "..."
    assert data["summary"].endswith("...")
    assert "This is a" in data["summary"]

@pytest.mark.unit
def test_extract_article_data_paragraph_break():
    article = {
        "title": "Test Title",
        "summary": "First paragraph.\n\nSecond paragraph that is very long and has lots of text.",
        "publisher": "Test Publisher",
        "link": "http://example.com"
    }
    # First paragraph is "First paragraph." (16 chars)
    # If max_summary_chars=30, since first paragraph <= 30, it should use the first paragraph exactly
    data = _extract_article_data(article, max_summary_chars=30)
    assert data["summary"] == "First paragraph."
