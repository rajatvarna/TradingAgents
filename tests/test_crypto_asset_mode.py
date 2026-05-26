import pytest
from unittest.mock import MagicMock, patch
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.analysts.market_analyst import create_market_analyst

def test_build_instrument_context_stock():
    context = build_instrument_context("AAPL", asset_type="stock")
    assert "instrument" in context
    assert "asset" not in context
    assert "Treat it as a crypto asset" not in context

def test_build_instrument_context_crypto():
    context = build_instrument_context("BTC-USD", asset_type="crypto")
    assert "asset" in context
    assert "Treat it as a crypto asset rather than a company" in context
    assert "do not assume company fundamentals are available" in context

def test_market_analyst_crypto_mode():
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    node = create_market_analyst(mock_llm)
    
    state = {
        "company_of_interest": "BTC-USD",
        "trade_date": "2026-05-26",
        "asset_type": "crypto",
        "messages": [("human", "BTC-USD")]
    }
    
    with patch("tradingagents.agents.analysts.market_analyst.invoke_with_retry") as mock_invoke, \
         patch("tradingagents.agents.analysts.market_analyst.load_prompt", return_value="Market analyst prompt."):
        mock_result = MagicMock()
        mock_result.tool_calls = []
        mock_result.content = "Crypto market report"
        mock_invoke.return_value = mock_result
        
        res = node(state)
        
    assert res["market_report"] == "Crypto market report"
    mock_invoke.assert_called_once()
