"""Guard the news analyst prompt against tool-signature drift (#1116).

The prompt used to advertise ``get_news(query, ...)`` while the tool takes a
``ticker``, tricking the LLM into hallucinating free-text query calls.

The prompt text moved out of ``news_analyst.py`` into
``tradingagents/prompts/analysts/news.v1.txt`` (T1.4b registry migration),
so the guard now reads the template file rather than the module source.
"""
import pytest

from tradingagents.agents.utils.news_data_tools import get_news
from tradingagents.audit.prompt_registry import default_registry


@pytest.mark.unit
def test_get_news_takes_ticker_not_query():
    arg_names = set(get_news.args.keys())
    assert "ticker" in arg_names
    assert "query" not in arg_names


@pytest.mark.unit
def test_news_prompt_matches_get_news_signature():
    template, _ = default_registry().load("analysts/news", "v1")
    assert "get_news(ticker, start_date, end_date)" in template
    assert "get_news(query" not in template
