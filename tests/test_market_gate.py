"""Tests for the pre-trade market state verification gate."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

from tradingagents.agents.risk_mgmt.market_gate import (
    _ticker_to_mic,
    check_market_state,
    create_market_gate,
)


class TestTickerToMic:
    """Test ticker suffix to MIC resolution."""

    def test_plain_us_ticker(self):
        assert _ticker_to_mic("AAPL") == "XNYS"

    def test_london_suffix(self):
        assert _ticker_to_mic("VOD.L") == "XLON"

    def test_tokyo_suffix(self):
        assert _ticker_to_mic("7203.T") == "XJPX"

    def test_hong_kong_suffix(self):
        assert _ticker_to_mic("0700.HK") == "XHKG"

    def test_case_insensitive(self):
        assert _ticker_to_mic("vod.l") == "XLON"

    def test_unknown_suffix_defaults_to_xnys(self):
        assert _ticker_to_mic("UNKNOWN.ZZ") == "XNYS"


class TestCheckMarketState:
    """Test the oracle HTTP call and fail-closed behavior."""

    @patch("tradingagents.agents.risk_mgmt.market_gate.urlopen")
    def test_open_market(self, mock_urlopen):
        response = BytesIO(json.dumps({"status": "OPEN", "mic": "XNYS"}).encode())
        mock_urlopen.return_value.__enter__ = lambda s: response
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        result = check_market_state("AAPL")
        assert result["status"] == "OPEN"
        assert result["blocked"] is False
        assert result["reason"] == ""

    @patch("tradingagents.agents.risk_mgmt.market_gate.urlopen")
    def test_closed_market(self, mock_urlopen):
        response = BytesIO(json.dumps({"status": "CLOSED", "mic": "XNYS"}).encode())
        mock_urlopen.return_value.__enter__ = lambda s: response
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        result = check_market_state("AAPL")
        assert result["status"] == "CLOSED"
        assert result["blocked"] is True
        assert "BLOCK TRADE" in result["reason"]

    @patch("tradingagents.agents.risk_mgmt.market_gate.urlopen")
    def test_halted_market(self, mock_urlopen):
        response = BytesIO(json.dumps({"status": "HALTED", "mic": "XNYS"}).encode())
        mock_urlopen.return_value.__enter__ = lambda s: response
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        result = check_market_state("AAPL")
        assert result["status"] == "HALTED"
        assert result["blocked"] is True

    @patch("tradingagents.agents.risk_mgmt.market_gate.urlopen")
    def test_network_failure_defaults_to_unknown(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("connection refused")

        result = check_market_state("AAPL")
        assert result["status"] == "UNKNOWN"
        assert result["blocked"] is True
        assert "BLOCK TRADE" in result["reason"]


class TestMarketGateNode:
    """Test the LangGraph node integration."""

    @patch("tradingagents.agents.risk_mgmt.market_gate.check_market_state")
    def test_open_market_adds_safe_advisory(self, mock_check):
        mock_check.return_value = {
            "status": "OPEN", "mic": "XNYS", "blocked": False, "reason": ""
        }

        node = create_market_gate()
        state = {
            "company_of_interest": "AAPL",
            "risk_debate_state": {
                "history": "Prior debate...",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "latest_speaker": "Neutral",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 3,
            },
        }

        result = node(state)
        history = result["risk_debate_state"]["history"]
        assert "[MARKET GATE]" in history
        assert "OPEN" in history
        assert "safe to proceed" in history

    @patch("tradingagents.agents.risk_mgmt.market_gate.check_market_state")
    def test_closed_market_adds_block_advisory(self, mock_check):
        mock_check.return_value = {
            "status": "CLOSED",
            "mic": "XNYS",
            "blocked": True,
            "reason": "BLOCK TRADE — market XNYS is CLOSED",
        }

        node = create_market_gate()
        state = {
            "company_of_interest": "AAPL",
            "risk_debate_state": {
                "history": "Prior debate...",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "latest_speaker": "Neutral",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 3,
            },
        }

        result = node(state)
        history = result["risk_debate_state"]["history"]
        assert "[MARKET GATE]" in history
        assert "BLOCK TRADE" in history
        assert "Do NOT approve execution" in history
