<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-TauricResearch-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/Join_GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
  <a href="https://deepwiki.com/TauricResearch/TradingAgents" target="_blank"><img alt="Deepwiki" src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</div>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README . -->
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

# TradingAgents: Multi-Agents LLM Financial Trading Framework

## News
- [2026-04] **TradingAgents v0.2.4** released with structured-output agents (Research Manager, Trader, Portfolio Manager), LangGraph checkpoint resume, persistent decision log, DeepSeek/Qwen/GLM/Azure provider support, Docker, and a Windows UTF-8 encoding fix. See [CHANGELOG.md](CHANGELOG.md) for the full list.
- [2026-03] **TradingAgents v0.2.3** released with multi-language support, GPT-5.4 family models, unified model catalog, backtesting date fidelity, and proxy support.
- [2026-03] **TradingAgents v0.2.2** released with GPT-5.4/Gemini 3.1/Claude 4.6 model coverage, five-tier rating scale, OpenAI Responses API, Anthropic effort control, and cross-platform stability.
- [2026-02] **TradingAgents v0.2.0** released with multi-provider LLM support (GPT-5.x, Gemini 3.x, Claude 4.x, Grok 4.x) and improved system architecture.
- [2026-01] **Trading-R1** [Technical Report](https://arxiv.org/abs/2509.11420) released, with [Terminal](https://github.com/TauricResearch/Trading-R1) expected to land soon.

<div align="center">
<a href="https://www.star-history.com/#TauricResearch/TradingAgents&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" />
   <img alt="TradingAgents Star History" src="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" style="width: 80%; height: auto;" />
 </picture>
</a>
</div>

> 🎉 **TradingAgents** officially released! We have received numerous inquiries about the work, and we would like to express our thanks for the enthusiasm in our community.
>
> So we decided to fully open-source the framework. Looking forward to building impactful projects with you!

<div align="center">

🚀 [TradingAgents](#tradingagents-framework) | ⚡ [Installation & CLI](#installation-and-cli) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho) | 📦 [Package Usage](#tradingagents-package) | 🤝 [Contributing](#contributing) | 📄 [Citation](#citation)

</div>

## TradingAgents Framework

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs trading decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> TradingAgents framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles. This ensures the system achieves a robust, scalable approach to market analysis and decision-making.

### Analyst Team
- Fundamentals Analyst: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- Sentiment Analyst: Analyzes social media and public sentiment using sentiment scoring algorithms to gauge short-term market mood.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent
- Composes reports from the analysts and researchers to make informed trading decisions. It determines the timing and magnitude of trades based on comprehensive market insights.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation and CLI

### Installation

Clone TradingAgents:
```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

Create a virtual environment in any of your favorite environment managers:
```bash
conda create -n tradingagents python=3.13
conda activate tradingagents
```

Install the package and its dependencies:
```bash
pip install .
```

### Docker

Alternatively, run with Docker:
```bash
cp .env.example .env  # add your API keys
docker compose run --rm tradingagents
```

For local models with Ollama:
```bash
docker compose --profile ollama run --rm tradingagents-ollama
```

### REST API + External Vault Runtime Keys

The API container can load API keys from HashiCorp Vault (KV v2) at startup and on demand.
Vault is intentionally decoupled from the main project compose so API rebuilds do not affect secret storage.

1. Start a persistent local Vault stack (separate compose project):

```bash
docker compose -f vault-local/docker-compose.yml up -d
```

2. Initialize/unseal and optionally seed provider keys:

```powershell
./scripts/vault_local_bootstrap.ps1
```

3. Set Vault variables in `.env` for the API container:

```bash
VAULT_ENABLED=true
VAULT_ADDR=http://host.docker.internal:8200
VAULT_TOKEN=<your-root-or-app-token>
VAULT_KV_MOUNT=secret
VAULT_KV_PATH=tradingagents/api-keys
VAULT_KEYS=GOOGLE_API_KEY,OPENROUTER_API_KEY
```

4. Start API:

```bash
docker compose up -d --build tradingagents-api
```

At startup, TradingAgents API refreshes configured keys from Vault. You can force refresh with:

```bash
curl -X POST http://localhost:9000/vault/refresh
```

To stop local Vault while keeping persistent data:

```bash
docker compose -f vault-local/docker-compose.yml down
```

Latest recommendation for a ticker:

```bash
curl "http://localhost:9000/recommendations/latest/NVDA"
```

### MCP Server for TradingAgents API

An MCP server is included to expose REST endpoints as MCP tools.

Run it with:

```bash
tradingagents-mcp
```

Environment variable:

```bash
TRADINGAGENTS_API_BASE_URL=http://localhost:9000
```

Exposed MCP tools include:
- `health_check`
- `submit_analysis`
- `get_request_status`
- `get_latest_recommendation`
- `force_vault_refresh`
- `get_runtime_env_value`

### Required APIs

TradingAgents supports multiple LLM providers. Set the API key for your chosen provider:

```bash
export OPENAI_API_KEY=...          # OpenAI (GPT)
export GOOGLE_API_KEY=...          # Google (Gemini)
export ANTHROPIC_API_KEY=...       # Anthropic (Claude)
export TENCENT_API_KEY=...         # Tencent Cloud LKEAP (Anthropic-compatible)
export XAI_API_KEY=...             # xAI (Grok)
export DEEPSEEK_API_KEY=...        # DeepSeek
export MOONSHOT_API_KEY=...        # Kimi (Moonshot AI) — https://platform.kimi.com/console/api-keys
export DASHSCOPE_API_KEY=...       # Qwen — International (dashscope-intl.aliyuncs.com)
export DASHSCOPE_CN_API_KEY=...    # Qwen — China (dashscope.aliyuncs.com)
export ZHIPU_API_KEY=...           # GLM via Z.AI (international)
export ZHIPU_CN_API_KEY=...        # GLM via BigModel (China, open.bigmodel.cn)
export MINIMAX_API_KEY=...         # MiniMax — Global (api.minimax.io, M2.x, 204K ctx)
export MINIMAX_CN_API_KEY=...      # MiniMax — China (api.minimaxi.com, M2.x, 204K ctx)
export OPENROUTER_API_KEY=...      # OpenRouter
export DEEPINFRA_API_KEY=...       # DeepInfra
export GITHUB_TOKEN=...            # GitHub Models / Copilot
export ALPHA_VANTAGE_API_KEY=...   # Alpha Vantage
```

> **Important for Kimi users**:
> - Keys from different Moonshot/Kimi consoles (e.g. `platform.moonshot.ai`, `platform.kimi.com`) are **not interchangeable**.
> - The provider defaults to `https://api.moonshot.ai/v1`.
> - If your key only works with a different endpoint (e.g. `https://api.moonshot.cn/v1`), set the `KIMI_BASE_URL` environment variable.
> - Generate your key from the console that matches your account.

#### Sign in with ChatGPT (OAuth)

Instead of an `OPENAI_API_KEY`, you can authenticate with your **ChatGPT**
subscription (Plus/Pro) using the same OAuth flow as the Codex CLI:

```bash
tradingagents login          # opens the browser for the OAuth sign-in
tradingagents                # then pick "OpenAI (ChatGPT OAuth)" as provider
```

Tokens are stored in `~/.tradingagents/oauth_openai.json` (mode `0600`) and
refreshed automatically; model calls go through the ChatGPT Codex backend.
Only Codex catalog models are available here (e.g. `gpt-5.3-codex`, `gpt-5.4`,
`gpt-5.5`) — generic API model IDs are rejected by that backend.

> ⚠️ Community/unofficial: this reuses Codex's public OAuth client and an
> undocumented backend that can change without notice. Whether using it from a
> non-Codex app complies with OpenAI's Terms is your responsibility.

See [`docs/openai-oauth.md`](docs/openai-oauth.md) for the full technical reference.

For enterprise providers (e.g. Azure OpenAI, AWS Bedrock), copy `.env.enterprise.example` to `.env.enterprise` and fill in your credentials.

For local models, configure Ollama with `llm_provider: "ollama"` in your config.

For AWS Bedrock, set `llm_provider: "bedrock"` and ensure your AWS credentials are configured (via `~/.aws/credentials`, environment variables, or IAM role). Set `AWS_DEFAULT_REGION` and optionally `AWS_PROFILE`. Install the extra dependency with `pip install .[bedrock]`. Use cross-region inference profile IDs (e.g. `us.anthropic.claude-opus-4-6-v1`) as model names.

### Optional APIs

```bash
export AGENTKEY_API_KEY=ak_...        # AgentKey — Chinese / international social sentiment
export AGENTKEY_BASE_URL=...          # optional, defaults to https://api.agentkey.app
```

- **What it does:** When set, the sentiment analyst enriches its report with Chinese / international social platforms via [AgentKey](https://agentkey.app/) — **Weibo** and **Zhihu** for every stock, plus **Xiaohongshu** and **Douyin** for consumer-brand sectors. This is most valuable for A-share / Hong Kong / China-listed or China-exposed tickers, whose sentiment the default US sources (StockTwits, Reddit) miss. Crypto tickers are unaffected.
- **Impact if unset:** Fully optional. With no key, the analyst silently skips these channels and runs exactly as before — no extra latency, no network calls, no cost. Existing US-only workflows are unchanged.
- **Cost:** AgentKey bills per successful call, so enabling this adds a small per-run cost (typically a few cents per analyzed ticker).
- **Where to get a key:** Sign up at [agentkey.app](https://agentkey.app/) and create an API key (format `ak_...`).

For Kimi (Moonshot AI), the provider defaults to `https://api.moonshot.ai/v1`. If your API key only works with a different endpoint (e.g. the China portal), set `KIMI_BASE_URL=https://api.moonshot.cn/v1`. Keys from different Moonshot/Kimi consoles are not interchangeable.

Alternatively, copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Alpha Vantage and data-vendor selection can also be configured from `.env`:
```bash
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
TRADINGAGENTS_DATA_VENDOR=alpha_vantage

# Optional per-category overrides:
TRADINGAGENTS_CORE_STOCK_VENDOR=alpha_vantage
TRADINGAGENTS_TECHNICAL_INDICATORS_VENDOR=alpha_vantage
TRADINGAGENTS_FUNDAMENTAL_DATA_VENDOR=alpha_vantage
TRADINGAGENTS_NEWS_DATA_VENDOR=alpha_vantage

# Optional per-tool override:
TRADINGAGENTS_TOOL_VENDOR_GET_STOCK_DATA=alpha_vantage
```

### CLI Usage

Launch the interactive CLI:
```bash
tradingagents          # installed command
python -m cli.main     # alternative: run directly from source
```
You will see a screen where you can select your desired tickers, analysis date, LLM provider, research depth, and more.

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### 🦞 OpenClaw Integration

Use TradingAgents directly from **Telegram, WhatsApp, Discord, Slack**, and other chat apps via [OpenClaw](https://openclaw.ai/), a personal AI assistant.

<details>
<summary>Click to expand OpenClaw integration guide</summary>

**Quick Start:**

1. Install the Trading Analyzer skill:
```bash
openclaw skill add https://github.com/TauricResearch/TradingAgents/tree/main/skills/openclaw
```

2. Set your LLM provider:
```bash
export OPENCLAW_LLM_PROVIDER="openai"
export OPENCLAW_LLM_MODEL="gpt-4"
export OPENAI_API_KEY="sk-..."
```

3. Chat with OpenClaw in your favorite app:
```
You: "Analyze NVDA"
OpenClaw: 📊 Stock Analysis: NVDA
          [Full multi-agent analysis with recommendation...]
```

**Features:**
- ✅ Full trading analysis via chat
- ✅ Share your LLM endpoint between apps
- ✅ Use any LLM provider (OpenAI, Anthropic, Google, DeepSeek, local models)
- ✅ Local/private or cloud-hosted

See [**OPENCLAW_INTEGRATION.md**](OPENCLAW_INTEGRATION.md) for detailed setup instructions.

</details>

## TradingAgents Package

### Implementation Details

We built TradingAgents with LangGraph to ensure flexibility and modularity. The framework supports multiple LLM providers: OpenAI, Google, Anthropic, Tencent Cloud LKEAP, xAI, DeepSeek, Kimi (Moonshot), Qwen (Alibaba DashScope, international and China endpoints), GLM (Zhipu), MiniMax (global + China), OpenRouter, DeepInfra, GitHub Models / Copilot, Ollama for local models, AWS Bedrock, and Azure OpenAI for enterprise.

### Python Usage

To use TradingAgents inside your code, you can import the `tradingagents` module and initialize a `TradingAgentsGraph()` object. The `.propagate()` function will return a decision. You can run `main.py`, here's also a quick example:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

# forward propagate
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

You can also adjust the default configuration to set your own choice of LLMs, debate rounds, etc.

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # openai, google, anthropic, tencent, xai, deepseek, kimi, qwen, qwen-cn, glm, glm-cn, minimax, minimax-cn, openrouter, deepinfra, github_copilot, ollama, bedrock, azure
config["deep_think_llm"] = "gpt-5.4"     # Model for complex reasoning
config["quick_think_llm"] = "gpt-5.4-mini" # Model for quick tasks
config["max_debate_rounds"] = 2

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

See `tradingagents/default_config.py` for all configuration options.

## Persistence and Recovery

TradingAgents persists two kinds of state across runs.

### Decision log

The decision log is always on. Each completed run appends its decision to `~/.tradingagents/memory/trading_memory.md`. On the next run for the same ticker, TradingAgents fetches the realised return (raw and alpha vs SPY), generates a one-paragraph reflection, and injects the most recent same-ticker decisions plus recent cross-ticker lessons into the Portfolio Manager prompt, so each analysis carries forward what worked and what didn't.

Override the path with `TRADINGAGENTS_MEMORY_LOG_PATH`.

### Checkpoint resume

Checkpoint resume is opt-in via `--checkpoint`. When enabled, LangGraph saves state after each node so a crashed or interrupted run resumes from the last successful step instead of starting over. On a resume run you will see `Resuming from step N for <TICKER> on <date>` in the logs; on a new run you will see `Starting fresh`. Checkpoints are cleared automatically on successful completion.

Per-ticker SQLite databases live at `~/.tradingagents/cache/checkpoints/<TICKER>.db` (override the base with `TRADINGAGENTS_CACHE_DIR`). Use `--clear-checkpoints` to reset all of them before a run.

```bash
tradingagents analyze --checkpoint           # enable for this run
tradingagents analyze --clear-checkpoints    # reset before running
```

```python
config = DEFAULT_CONFIG.copy()
config["checkpoint_enabled"] = True
ta = TradingAgentsGraph(config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
```

## Flint Shadow Service

This repository also includes an async shadow-analysis service used as a Flint-side comparator. It keeps state in this repo under `output/` and treats TradingAgents output as advisory evidence, not execution.

### What it adds

- `POST /v1/shadow-runs` creates an async run and returns `202 Accepted`
- `GET /v1/shadow-runs/{run_id}` returns run state and metadata
- `GET /v1/shadow-runs/{run_id}/events` returns the event log
- `GET /v1/shadow-runs/{run_id}/artifacts` returns persisted outputs
- `GET /v1/shadow-runs/{run_id}/decision` returns the normalized recommendation
- `GET /v1/shadow-runs/{run_id}/precedents` returns nearest prior runs
- `GET /v1/precedents` searches the precedent store
- `GET /v1/shadow-runs/{run_id}/report.md` returns a markdown report

### What gets stored

- run state and transition history
- raw tool provenance and telemetry
- artifact manifests and hashes
- evaluation records and quality gates
- precedent embeddings for retrieval against prior runs

### Local run shape

```bash
docker compose up postgres api worker
```

The service is designed for Flint ingestion workflows where the important output is a traceable evidence bundle, not an order.

## Contributing

We welcome contributions from the community! Whether it's fixing a bug, improving documentation, or suggesting a new feature, your input helps make this project better. If you are interested in this line of research, please consider joining our open-source financial AI research community [Tauric Research](https://tauric.ai/).

Past contributions, including code, design feedback, and bug reports, are credited per release in [`CHANGELOG.md`](CHANGELOG.md).

## Citation

Please reference our work if you find *TradingAgents* provides you with some help :)

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```
## Beginner Note
This project is a multi-agent trading system using AI models.

## Indian Market Support

TradingAgents can be used to analyze Indian stocks listed on NSE and BSE exchanges using their respective ticker symbols (e.g., RELIANCE.NS, TCS.NS).
