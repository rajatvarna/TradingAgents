# 🦞 TradingAgents + OpenClaw Quick Reference

## Installation (1 minute)

```bash
# 1. Install TradingAgents package
openclaw pip install tradingagents

# 2. Install the skill
openclaw skill add https://github.com/TauricResearch/TradingAgents/tree/main/skills/openclaw

# 3. Restart OpenClaw
openclaw restart
```

## Configuration (2 minutes)

**Choose one:**

### OpenAI (GPT-4)
```bash
export OPENCLAW_LLM_PROVIDER="openai"
export OPENCLAW_LLM_MODEL="gpt-4"
export OPENAI_API_KEY="sk-..."
```

### Anthropic (Claude)
```bash
export OPENCLAW_LLM_PROVIDER="anthropic"
export OPENCLAW_LLM_MODEL="claude-3-opus"
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Google (Gemini)
```bash
export OPENCLAW_LLM_PROVIDER="google"
export OPENCLAW_LLM_MODEL="gemini-2.0-pro"
export GOOGLE_API_KEY="..."
```

### DeepSeek (Advanced Reasoning)
```bash
export OPENCLAW_LLM_PROVIDER="deepseek"
export OPENCLAW_LLM_MODEL="deepseek-reasoner"
export DEEPSEEK_API_KEY="sk-..."
```

### Local (Ollama)
```bash
export OPENCLAW_LLM_PROVIDER="ollama"
export OPENCLAW_LLM_MODEL="mistral"
export OLLAMA_BASE_URL="http://localhost:11434"
```

### Use OpenClaw's LLM Proxy
```bash
export OPENCLAW_LLM_PROVIDER="openclaw"
export OPENCLAW_LLM_MODEL="gpt-4"  # your configured model
export OPENCLAW_BASE_URL="http://localhost:8000/v1"
```

## Usage

### Telegram/WhatsApp/Discord/Slack

```
You: "Analyze NVDA"
OpenClaw: 📊 Stock Analysis: NVDA (2024-05-10)
          ...detailed analysis...
          Recommendation: 🟢 Buy

You: "What about TSLA?"
OpenClaw: 📊 Stock Analysis: TSLA
          ...analysis...

You: "Trading setup for SPY as of 2024-01-15"
OpenClaw: 📊 Stock Analysis: SPY (2024-01-15)
          ...historical analysis...
```

### Commands

| Query | Result |
|-------|--------|
| "Analyze NVDA" | Full NVDA analysis |
| "TSLA trading setup" | TSLA analysis |
| "Market analysis for SPY" | SPY analysis |
| "AAPL as of 2024-01-15" | AAPL analysis on specific date |
| "What about BTC?" | Any ticker analysis |

## Configuration Files

### Persistent Setup

Create `~/.openclaw/.env.trading`:

```bash
export OPENCLAW_LLM_PROVIDER="openai"
export OPENCLAW_LLM_MODEL="gpt-4"
export OPENAI_API_KEY="sk-..."
```

Load when needed:
```bash
source ~/.openclaw/.env.trading
openclaw restart
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Trading Agents library not found" | `openclaw pip install tradingagents` |
| API key errors | Check env vars: `echo $OPENAI_API_KEY` |
| Skill not found | Restart OpenClaw: `openclaw restart` |
| Analysis is slow | Use faster model: `gpt-4-turbo` instead of `gpt-4` |
| Unknown ticker | Check spelling (e.g., "NVDA" not "NVIDIA") |

## Data Used

| Type | Source | Cost |
|------|--------|------|
| Stock Prices | yfinance | Free |
| Technical Indicators | yfinance/stockstats | Free |
| Fundamentals | yfinance | Free |
| News | yfinance | Free |
| LLM Analysis | Your provider (OpenAI/Anthropic/etc) | Varies |

## What You Get

✅ Multi-agent analysis (4+ specialized agents)
✅ Technical + Fundamental + News + Sentiment analysis
✅ Bull vs Bear debate synthesis
✅ Risk assessment
✅ Clear Buy/Hold/Sell recommendation
✅ Detailed rationale
✅ Works in all your favorite chat apps

## Resources

- **Full Guide**: [OPENCLAW_INTEGRATION.md](OPENCLAW_INTEGRATION.md)
- **Skill Docs**: [skills/openclaw/README.md](skills/openclaw/README.md)
- **OpenClaw**: https://openclaw.ai/
- **TradingAgents**: https://github.com/dkoul/TradingAgents

---

**Questions?** Check the full guide or open an issue on GitHub!
