# OpenClaw Integration Guide

**Make TradingAgents accessible to OpenClaw users via Telegram, WhatsApp, Discord, Slack, and more.**

## Overview

This guide explains how to integrate TradingAgents with [OpenClaw](https://openclaw.ai/), a personal AI assistant that runs on your machine and connects to your favorite chat apps.

With this integration:

- ✅ OpenClaw users can analyze stocks directly from their chat apps
- ✅ Share your LLM endpoint between TradingAgents and OpenClaw
- ✅ Get rich trading analysis via Telegram/WhatsApp/Discord
- ✅ Run locally or cloud-hosted
- ✅ Use any LLM provider (OpenAI, Anthropic, Google, DeepSeek, local models, etc.)

## Quick Start (5 minutes)

### 1. Install TradingAgents

```bash
pip install tradingagents
```

### 2. Install the OpenClaw Skill

From your OpenClaw machine:

```bash
openclaw skill add https://github.com/dkoul/TradingAgents/tree/main/skills/openclaw
```

Or manually:

```bash
cp -r skills/openclaw ~/.openclaw/skills/trading-analyzer
```

### 3. Configure Your LLM

Set environment variables on your OpenClaw machine:

**Option A: Use your OpenAI subscription**
```bash
export OPENCLAW_LLM_PROVIDER="openai"
export OPENCLAW_LLM_MODEL="gpt-4"
export OPENAI_API_KEY="sk-..."
```

**Option B: Use OpenClaw's own LLM proxy**
```bash
export OPENCLAW_LLM_PROVIDER="openclaw"
export OPENCLAW_LLM_MODEL="gpt-4"  # or your configured model
export OPENCLAW_BASE_URL="http://localhost:8000/v1"
```

### 4. Restart OpenClaw

```bash
openclaw restart
```

### 5. Ask in Telegram/WhatsApp/Discord

```
You: Analyze NVDA
OpenClaw: 📊 Stock Analysis: NVDA

Market Analysis
NVDA trading at $875, up 2.3% on strong AI earnings...

Recommendation: 🟢 Buy
...
```

Done! 🎉

## Detailed Setup

### Prerequisites

- **OpenClaw** running locally or cloud-hosted
- **Python 3.10+** environment
- **LLM API key** (OpenAI, Anthropic, Google, etc.)
- **Internet connection** (for data fetching and LLM calls)

### Architecture

```
┌─────────────────────────────────────────┐
│  Your Chat App (Telegram/WhatsApp/etc)  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│       OpenClaw Assistant                 │
│  (Persistent memory, automation, etc)   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Trading Analyzer Skill                  │
│  (skills/openclaw/trading_analyzer.py)  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│    TradingAgentsGraph                    │
│  ├─ Market Analyst                       │
│  ├─ Fundamentals Analyst                 │
│  ├─ News Analyst                         │
│  ├─ Sentiment Analyst                    │
│  ├─ Bull/Bear Researchers                │
│  ├─ Trader Agent                         │
│  ├─ Risk Managers                        │
│  └─ Portfolio Manager                    │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   LLM Provider (OpenAI/Anthropic/Google │
│   /DeepSeek/Ollama/OpenClaw/etc)        │
└─────────────────────────────────────────┘
```

### Step 1: Install TradingAgents Package

Install in your OpenClaw Python environment:

```bash
# If using OpenClaw's built-in Python
openclaw pip install tradingagents

# Or in a virtual environment
python -m venv trading_venv
source trading_venv/bin/activate
pip install tradingagents
```

### Step 2: Set Up LLM Provider

Choose your LLM provider and set API credentials:

#### Option A: OpenAI (GPT-4, GPT-4-turbo, etc.)

```bash
export OPENCLAW_LLM_PROVIDER="openai"
export OPENCLAW_LLM_MODEL="gpt-4"
export OPENAI_API_KEY="sk-..."
```

Get your API key from: https://platform.openai.com/account/api-keys

#### Option B: Anthropic (Claude)

```bash
export OPENCLAW_LLM_PROVIDER="anthropic"
export OPENCLAW_LLM_MODEL="claude-3-opus"
export ANTHROPIC_API_KEY="sk-ant-..."
```

Get your API key from: https://console.anthropic.com/

#### Option C: Google (Gemini)

```bash
export OPENCLAW_LLM_PROVIDER="google"
export OPENCLAW_LLM_MODEL="gemini-2.0-pro"
export GOOGLE_API_KEY="..."
```

Get your API key from: https://aistudio.google.com/app/apikey

#### Option D: DeepSeek (Reasoning Model)

```bash
export OPENCLAW_LLM_PROVIDER="deepseek"
export OPENCLAW_LLM_MODEL="deepseek-reasoner"
export DEEPSEEK_API_KEY="sk-..."
```

Get your API key from: https://platform.deepseek.com/

#### Option E: Use OpenClaw's LLM Proxy

If OpenClaw is already configured with an LLM, reuse that endpoint:

```bash
export OPENCLAW_LLM_PROVIDER="openclaw"
export OPENCLAW_LLM_MODEL="gpt-4"  # or whatever you configured
export OPENCLAW_BASE_URL="http://localhost:8000/v1"
```

### Step 3: Install the Skill

Navigate to the TradingAgents directory and copy the skill:

```bash
# Clone or download TradingAgents
git clone https://github.com/dkoul/TradingAgents.git
cd TradingAgents

# Copy the skill to OpenClaw
cp -r skills/openclaw ~/.openclaw/skills/trading-analyzer
```

Or use the CLI:

```bash
openclaw skill add https://github.com/dkoul/TradingAgents/tree/main/skills/openclaw
```

### Step 4: Verify Installation

Restart OpenClaw and verify the skill is loaded:

```bash
openclaw restart

# Check skill status
openclaw skill list | grep trading
```

### Step 5: Test the Skill

Send a message to OpenClaw via your chat app:

```
"Analyze NVDA"
```

Expected response:

```
📊 Stock Analysis: NVDA (2024-05-10)

Market Analysis
[NVDA technical indicators, price action...]

Fundamentals
[P/E ratio, revenue growth, margins...]

News & Events
[Recent announcements...]

Recommendation: 🟢 Buy
[Detailed rationale...]
```

## Usage Examples

### Basic Stock Analysis

```
You: "Analyze TSLA"
OpenClaw: 📊 Stock Analysis: TSLA
          [Full multi-agent analysis...]
```

### Historical Analysis (for backtesting)

```
You: "AAPL trading setup as of 2024-01-15"
OpenClaw: 📊 Stock Analysis: AAPL (2024-01-15)
          [Analysis as if today were 2024-01-15...]
```

### Quick Check

```
You: "What about MSFT?"
OpenClaw: 📊 Stock Analysis: MSFT
          [Current analysis...]
```

### Sector Analysis

```
You: "Analyze QQQ"  # or SPY, IWM, etc.
OpenClaw: 📊 Stock Analysis: QQQ
          [ETF analysis...]
```

## Configuration Files

### OpenClaw Trading Agent Profile

Create `~/.openclaw/profiles/trading_agent.json`:

```json
{
  "name": "Trading Agent",
  "description": "Your personal trading analyst",
  "system_prompt": "You are a trading analyst powered by TradingAgents. When asked about stocks, use the trading-analyzer skill to provide detailed multi-agent analysis.",
  "commands": {
    "analyze": "Analyze $1",
    "trading": "Trading outlook for $1",
    "market": "Market analysis for $1"
  }
}
```

### Environment Setup

Create `~/.openclaw/.env.trading`:

```bash
# LLM Configuration
OPENCLAW_LLM_PROVIDER=openai
OPENCLAW_LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...

# Optional: Custom startup
alias ta='openclaw message "Analyze"'
```

Load when needed:

```bash
source ~/.openclaw/.env.trading
openclaw run
```

## Advanced Scenarios

### Scenario 1: Share OpenAI Subscription Between Applications

**Goal**: Use your OpenAI API key for both OpenClaw's general tasks and TradingAgents analysis.

**Setup**:

```bash
# Set global OpenAI key
export OPENAI_API_KEY="sk-..."

# Configure TradingAgents to use OpenAI via OpenClaw
export OPENCLAW_LLM_PROVIDER="openai"
export OPENCLAW_LLM_MODEL="gpt-4"
```

OpenClaw will:
- Use the key for general tasks
- Pass it through to TradingAgents
- Centralize cost tracking in OpenAI dashboard

### Scenario 2: Run Locally Without External API Calls

**Goal**: Complete privacy - analyze stocks using local LLMs.

**Setup**:

1. Install local LLM via OpenClaw:
   ```bash
   openclaw model install ollama/mistral
   ```

2. Configure TradingAgents to use local model:
   ```bash
   export OPENCLAW_LLM_PROVIDER="ollama"
   export OPENCLAW_LLM_MODEL="mistral"
   export OLLAMA_BASE_URL="http://localhost:11434"
   ```

### Scenario 3: Advanced Reasoning with DeepSeek

**Goal**: Use DeepSeek's reasoning model for deeper analysis.

**Setup**:

```bash
export OPENCLAW_LLM_PROVIDER="deepseek"
export OPENCLAW_LLM_MODEL="deepseek-reasoner"
export DEEPSEEK_API_KEY="sk-..."
```

Results will include:
- Extended reasoning traces
- Deeper analysis of edge cases
- Longer processing time but better insights

## Troubleshooting

### Issue: "Trading Agents library not found"

**Solution**: Install in OpenClaw's environment
```bash
openclaw pip install tradingagents
```

### Issue: "API key not found" or "Unauthorized"

**Solution**: Verify environment variables are set
```bash
# Check if key is set
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY="sk-..."

# Restart OpenClaw
openclaw restart
```

### Issue: Analysis is very slow

**Solution**: Use a faster model
```bash
# Instead of:
export OPENCLAW_LLM_MODEL="gpt-4"

# Try:
export OPENCLAW_LLM_MODEL="gpt-4-turbo"
```

### Issue: Skill not recognized

**Solution**: Check skill installation
```bash
# List installed skills
openclaw skill list

# Re-install if needed
cp -r skills/openclaw ~/.openclaw/skills/trading-analyzer
openclaw restart
```

### Issue: "Unknown ticker"

The skill supports any publicly traded ticker. If a ticker isn't recognized:
1. Check spelling (e.g., "NVDA" not "NVIDIA")
2. Verify ticker is still active (not delisted)
3. Try with full company name: "NVIDIA Inc" -> Try "NVDA" instead

## Performance Tuning

### Faster Analysis

Reduce debate rounds (fewer agent discussions):

```python
# In TradingAgents config
config = {
    "max_debate_rounds": 1,  # default
    "max_risk_discuss_rounds": 1,  # default
}
```

### More Thorough Analysis

Increase debate rounds and use a reasoning model:

```python
config = {
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 2,
    "llm_provider": "deepseek",
    "deep_think_llm": "deepseek-reasoner",
}
```

### Cost Optimization

Use cheaper models for initial screening:

```bash
# Use mini model for fast analysis
export OPENCLAW_LLM_MODEL="gpt-4-turbo"  # 1/10th the cost of gpt-4

# Or use reasoning models only when requested
export OPENCLAW_LLM_MODEL="gpt-4"  # default
```

## Monitoring & Logging

### View OpenClaw Logs

```bash
openclaw logs tail -f
```

### Check Trading Agent Sessions

```bash
openclaw history --skill trading-analyzer
```

### Export Analysis History

```bash
openclaw export --skill trading-analyzer --format csv > trading_analyses.csv
```

## Further Integration

### Add to OpenClaw Memory

Let OpenClaw remember your trading preferences:

```
You: "I'm interested in tech stocks"
OpenClaw: [Remembers preference]

You: "Analyze the sector"
OpenClaw: [Analyzes NVDA, MSFT, AMD, AAPL...]
```

### Create Custom Workflows

Chain multiple analyses:

```
You: "Compare NVDA vs TSLA"
OpenClaw: [Analyzes both, compares]

You: "Which one should I buy first?"
OpenClaw: [Synthesizes recommendations]
```

### Scheduled Analyses

Set up periodic market analysis:

```bash
# In OpenClaw config
schedules:
  - "every Monday at 9 AM: Analyze SPY"
  - "every Friday at 4 PM: Market outlook for QQQ"
```

## Contributing

Found an issue or have ideas for improvements?

1. **Report bugs**: [GitHub Issues](https://github.com/dkoul/TradingAgents/issues)
2. **Suggest features**: [GitHub Discussions](https://github.com/dkoul/TradingAgents/discussions)
3. **Contribute code**: [GitHub Pull Requests](https://github.com/dkoul/TradingAgents/pulls)

## Resources

- **TradingAgents**: https://github.com/dkoul/TradingAgents
- **OpenClaw**: https://openclaw.ai/
- **OpenClaw Docs**: https://docs.openclaw.ai/
- **Research Paper**: arXiv:2412.20138

## Support

- 💬 **OpenClaw Discord**: https://discord.com/invite/clawd
- 🐛 **Report Issues**: https://github.com/dkoul/TradingAgents/issues
- 📝 **Documentation**: [skills/openclaw/README.md](skills/openclaw/README.md)

---

**Happy trading! 🚀**
