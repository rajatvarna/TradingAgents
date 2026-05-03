# 🦞 OpenClaw Trading Analyzer Skill

This skill brings **AI-powered stock trading analysis** directly into your OpenClaw assistant's chat apps (Telegram, WhatsApp, Discord, Slack, etc.).

## What It Does

Uses the **TradingAgents** multi-agent system to deliver:
- 📊 **Technical Analysis** - MACD, RSI, trend indicators
- 📈 **Fundamental Insights** - P/E ratios, earnings, balance sheet
- 📰 **News & Events** - Recent catalysts, macro events
- 💬 **Sentiment Analysis** - Insider trading activity, social signals
- 🎯 **Investment Thesis** - Bull vs. Bear debate
- 🔴 **Risk Assessment** - Conservative vs. Aggressive perspectives
- 💡 **Clear Recommendation** - Buy/Hold/Sell with rationale

All powered by your choice of LLM provider (OpenAI, Anthropic, Google, DeepSeek, local models).

## Installation

### Option 1: Install via OpenClaw CLI (Easiest)

```bash
openclaw skill add trading-analyzer
```

### Option 2: Manual Installation

1. Clone/download the trading-analyzer skill
2. Copy to your OpenClaw skills directory:

```bash
cp skills/openclaw/trading_analyzer.py ~/.openclaw/skills/
```

3. Restart OpenClaw

## Setup

### Prerequisites

Install the TradingAgents package:

```bash
pip install tradingagents
```

Or add to your OpenClaw environment:

```bash
openclaw pip install tradingagents
```

### Configuration

Set environment variables for your LLM:

**For OpenAI:**
```bash
export OPENCLAW_LLM_PROVIDER="openai"
export OPENCLAW_LLM_MODEL="gpt-4"
export OPENAI_API_KEY="sk-..."
```

**For Anthropic (Claude):**
```bash
export OPENCLAW_LLM_PROVIDER="anthropic"
export OPENCLAW_LLM_MODEL="claude-3-opus"
export ANTHROPIC_API_KEY="sk-ant-..."
```

**For Google (Gemini):**
```bash
export OPENCLAW_LLM_PROVIDER="google"
export OPENCLAW_LLM_MODEL="gemini-2.0-pro"
export GOOGLE_API_KEY="..."
```

**For DeepSeek via OpenAI-compatible API:**
```bash
export OPENCLAW_LLM_PROVIDER="deepseek"
export OPENCLAW_LLM_MODEL="deepseek-reasoner"
export DEEPSEEK_API_KEY="sk-..."
```

**Using OpenClaw's own LLM proxy:**
```bash
export OPENCLAW_LLM_PROVIDER="openclaw"
export OPENCLAW_LLM_MODEL="gpt-4"  # or your configured model
export OPENCLAW_BASE_URL="http://localhost:8000/v1"  # your OpenClaw endpoint
```

## Usage

### Chat Examples

**Via Telegram/WhatsApp/Discord/etc:**

```
User: "Analyze NVDA"
Bot: 📊 Stock Analysis: NVDA (2024-05-10)
     
     Market Analysis
     [Technical indicators and price action summary]
     
     Fundamentals
     [P/E, revenue growth, margins...]
     
     Recommendation: 🟢 Buy
     
     NVIDIA's strong earnings guidance combined with AI
     catalyst positioning suggest upside to $900...
```

```
User: "What about TSLA?"
Bot: 📊 Stock Analysis: TSLA (2024-05-10)
     [Full analysis...]
```

```
User: "Market analysis for SPY as of 2024-01-15"
Bot: 📊 Stock Analysis: SPY (2024-01-15)
     [Historical analysis for backtesting...]
```

### Supported Commands

The skill recognizes queries like:
- "Analyze [TICKER]"
- "[TICKER] analysis"
- "Trading setup for [TICKER]"
- "[TICKER] outlook"
- "[TICKER] as of YYYY-MM-DD" (specific date)
- "What about [TICKER]?"
- "Do market analysis for [TICKER]"

### Ticker Symbols

Any publicly traded ticker works:
- Stocks: NVDA, TSLA, AAPL, MSFT, AMD, etc.
- ETFs: SPY, QQQ, IWM, etc.
- Cryptocurrencies: BTC (if the data backend supports it)

## Data Sources

The skill uses configurable data vendors:

| Data Type | Default | Alternative |
|-----------|---------|-------------|
| Stock Prices | yfinance | Alpha Vantage |
| Technical Indicators | yfinance stockstats | Alpha Vantage |
| Fundamentals | yfinance | Alpha Vantage |
| News | yfinance | Alpha Vantage |

No additional API keys needed for yfinance. For Alpha Vantage, set:
```bash
export ALPHA_VANTAGE_API_KEY="..."
```

## Architecture

```
OpenClaw (Telegram/WhatsApp/Discord)
    ↓
trading_analyzer.py skill
    ↓
TradingAgentsGraph
    ├─ Market Analyst (technical indicators)
    ├─ Fundamentals Analyst (financial metrics)
    ├─ News Analyst (events & catalysts)
    ├─ Sentiment Analyst (insider activity)
    ├─ Debate Phase (Bull vs. Bear)
    ├─ Research Manager (synthesis)
    ├─ Trader Agent (proposal)
    ├─ Risk Management (debate)
    └─ Portfolio Manager (final decision)
    ↓
LLM Provider (OpenAI/Anthropic/Google/DeepSeek/OpenClaw/Local)
    ↓
Analysis Report
```

## Advanced Configuration

### Using Multiple LLM Providers

Use different providers for different tasks:

```bash
# Fast analysis with OpenAI mini model
export OPENCLAW_LLM_PROVIDER="openai"
export OPENCLAW_LLM_MODEL="gpt-4-turbo"  # Fast, cheap
```

### Local LLM via OpenClaw

Run locally without external API calls:

```bash
# Use OpenClaw's local model proxy
export OPENCLAW_LLM_PROVIDER="openclaw"
export OPENCLAW_LLM_MODEL="ollama/mistral"
export OPENCLAW_BASE_URL="http://localhost:8000/v1"
```

### Thinking Models (Advanced Reasoning)

For deeper analysis with reasoning:

```bash
# Use DeepSeek's reasoning model
export OPENCLAW_LLM_PROVIDER="deepseek"
export OPENCLAW_LLM_MODEL="deepseek-reasoner"

# Or OpenAI's reasoning model
export OPENCLAW_LLM_PROVIDER="openai"
export OPENCLAW_LLM_MODEL="gpt-5-with-reasoning"
```

## Troubleshooting

### "Trading Agents library not found"

Install the package:
```bash
openclaw pip install tradingagents
```

### "No API key provided"

Set the appropriate API key for your LLM provider:
```bash
export OPENAI_API_KEY="sk-..."  # for OpenAI
export ANTHROPIC_API_KEY="sk-ant-..."  # for Anthropic
```

### "Model not recognized"

Ensure your configured model is available with your LLM provider:
```bash
export OPENCLAW_LLM_MODEL="gpt-4"  # valid OpenAI model
```

### Slow Analysis

Use a faster model:
```bash
# Instead of:
export OPENCLAW_LLM_MODEL="gpt-4"

# Try:
export OPENCLAW_LLM_MODEL="gpt-4-turbo"
```

## Performance Tips

1. **Reduce thinking rounds** (in TradingAgents config):
   ```python
   config["max_debate_rounds"] = 1  # default is 1
   ```

2. **Use faster models** for real-time chat
3. **Batch analyses** - ask for multiple stocks in one message
4. **Cache results** - similar queries run faster on second request

## Integration with TradingAgents

This skill is a thin wrapper around [TradingAgents](https://github.com/dkoul/TradingAgents), an open-source research framework for multi-agent trading analysis.

For more control, you can use TradingAgents directly:

```python
from tradingagents.graph import TradingAgentsGraph

ta = TradingAgentsGraph(config={
    "llm_provider": "openclaw",
    "deep_think_llm": "gpt-4",
})

state, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)
```

## Contributing

Found a bug or have a feature request? Open an issue on the [TradingAgents GitHub](https://github.com/dkoul/TradingAgents/issues).

## License

This skill follows the same license as TradingAgents. See [LICENSE](../../LICENSE).

---

**Made with 🦞 by the TradingAgents community**
