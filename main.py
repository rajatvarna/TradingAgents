from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
import os

# Load environment variables from .env files
load_dotenv()
load_dotenv(".env.enterprise", override=False)

def validate_config(config):
    if not config.get("deep_think_llm"):
        raise ValueError("deep_think_llm is not configured")

    if not config.get("quick_think_llm"):
        raise ValueError("quick_think_llm is not configured")

def validate_input(ticker, date):
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Invalid ticker symbol")

    if not date or not isinstance(date, str):
        raise ValueError("Invalid date format")

def main():
    try:
        # Create config
        config = DEFAULT_CONFIG.copy()
        config["deep_think_llm"] = "gpt-5.4-mini"
        config["quick_think_llm"] = "gpt-5.4-mini"
        config["max_debate_rounds"] = 1

        config["data_vendors"] = {
            "core_stock_apis": "yfinance",           # Options: alpha_vantage, yfinance
            "technical_indicators": "yfinance",      # Options: alpha_vantage, yfinance
            "fundamental_data": "yfinance",          # Options: alpha_vantage, yfinance
            "news_data": "yfinance",                 # Options: alpha_vantage, yfinance, searxng
        }

        validate_config(config)

        ticker = "NVDA"
        date = datetime.today().strftime("%Y-%m-%d")
        validate_input(ticker, date)

        print("[INFO] Initializing TradingAgentsGraph...")
        ta = TradingAgentsGraph(debug=True, config=config)

        print(f"[INFO] Running analysis for {ticker} on {date}...")
        _, decision = ta.propagate(ticker, date)

        print("[RESULT]")
        print(decision)

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
