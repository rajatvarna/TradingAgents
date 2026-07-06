from unittest.mock import MagicMock
import pytest
from tradingagents.agents import (
    create_technical_analyst,
    create_quant_analyst,
    create_alternative_data_analyst,
)

@pytest.mark.unit
def test_technical_analyst():
    llm = MagicMock()
    
    mock_msg = MagicMock()
    mock_msg.tool_calls = []
    mock_msg.content = "Technical Report"
    # Mock both .invoke() and direct __call__() on the bound LLM
    llm.bind_tools.return_value.invoke.return_value = mock_msg
    llm.bind_tools.return_value.return_value = mock_msg
    
    analyst = create_technical_analyst(llm)
    state = {
        "trade_date": "2026-07-06",
        "company_of_interest": "NVDA",
        "messages": []
    }
    res = analyst(state)
    assert res["technical_report"] == "Technical Report"

@pytest.mark.unit
def test_quant_analyst():
    llm = MagicMock()
    
    mock_msg = MagicMock()
    mock_msg.tool_calls = []
    mock_msg.content = "Quant Report"
    llm.bind_tools.return_value.invoke.return_value = mock_msg
    llm.bind_tools.return_value.return_value = mock_msg
    
    analyst = create_quant_analyst(llm)
    state = {
        "trade_date": "2026-07-06",
        "company_of_interest": "NVDA",
        "messages": []
    }
    res = analyst(state)
    assert res["quant_report"] == "Quant Report"

@pytest.mark.unit
def test_alternative_data_analyst():
    llm = MagicMock()
    
    mock_msg = MagicMock()
    mock_msg.tool_calls = []
    mock_msg.content = "Alternative Report"
    llm.bind_tools.return_value.invoke.return_value = mock_msg
    llm.bind_tools.return_value.return_value = mock_msg
    
    analyst = create_alternative_data_analyst(llm)
    state = {
        "trade_date": "2026-07-06",
        "company_of_interest": "NVDA",
        "messages": []
    }
    res = analyst(state)
    assert res["alternative_report"] == "Alternative Report"
