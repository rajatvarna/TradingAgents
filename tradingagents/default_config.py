import os

INVESTMENT_HORIZONS = {
    "1_day":        "Day trading / Intraday",
    "1_week":       "Swing trading / Short-term",
    "1_month":      "Medium-term trading",
    "6_months":     "Medium-term investing",
    "1_year":       "Long-term investing",
    "5_years_plus": "Long-term strategic allocation",
     "medium_term":  "Medium-term (default)",
}

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_DEEPSEEK_REASONING_EFFORT": "deepseek_reasoning_effort",
    "TRADINGAGENTS_IIC_DB_PATH":          "iic_db_path",
    "TRADINGAGENTS_IIC_DATA_DIR":         "iic_data_dir",
    "TRADINGAGENTS_COST_GUARD_ENABLED":   "cost_guard_enabled",
    "TRADINGAGENTS_ORCHESTRATOR_ENABLED": "orchestrator_enabled",
}


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # IIC-FORGE F1 — persistence + data layout
    "iic_db_path": os.path.join(_TRADINGAGENTS_HOME, "iic.db"),
    "iic_data_dir": os.path.join(_TRADINGAGENTS_HOME, "data"),
    # IIC-FORGE F1 — cost guards (coded but disabled by default — see
    # docs/superpowers/specs/2026-05-25-iic-forge-program-design.md Appendix A).
    "cost_guard_enabled": False,
    # IIC-FORGE F3 — always-on sensing + triage
    "sensing_redis_url": "redis://127.0.0.1:6379/0",
    "sensing_ingest_stream": "ingest:raw",
    "sensing_consumer_group": "triage",
    "sensing_dead_stream": "ingest:dead",
    "sensing_triage_consumers": 4,
    "sensing_triage_max_failures": 5,
    "sensing_dedupe_cosine_threshold": 0.92,
    "sensing_dedupe_window_hours": 24,
    "sensing_fingerprint_ttl_hours": 72,
    "sensing_watchlist_salience_threshold": 0.7,
    "sensing_watchlist_confidence_threshold": 0.8,
    "sensing_watchlist_ttl_days": 7,
    "sensing_watchlist_refresh_seconds": 60,
    "sensing_salience_cache_ttl_seconds": 86400,
    "sensing_embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
    "sensing_adapters_enabled": {
        "polygon_news": True,
        "telegram": True,
        "rss": True,
        "gdelt": True,
        "macro": True,
        "x": False,   # off by default per spec D8 / R-F3-3
    },
    # IIC-FORGE F4 — autonomous trigger loop (orchestrator)
    "orchestrator_enabled": False,
    "promoter_poll_interval_s": 10,
    "promoter_batch_size": 50,
    "alert_cooldown_min": 60,
    "alert_salience_threshold": 0.7,
    "alert_ticker_confidence_threshold": 0.8,
    "worker_poll_interval_s": 2,
    "worker_job_timeout_min": 20,
    "max_concurrent_jobs": 1,
    # Cost guards (program-spec Appendix A: enabled=False during F0–F5)
    "trigger_backpressure_enabled": False,
    "trigger_backpressure_max_pending": 20,
    "trigger_daily_rate_enabled": False,
    "trigger_daily_rate_max_jobs": 200,
    "daily_budget_enabled": False,
    "daily_budget_usd": 10.0,
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "google",
    "deep_think_llm": "gemini-2.5-pro",
    "quick_think_llm": "gemini-2.5-flash-lite",
    # T0.1 — deterministic generation defaults.  Reproducibility floor for
    # the audit trail: with these pinned, two runs against the same
    # (ticker, date, prompt, snapshot) should produce ε-close outputs, and
    # any drift becomes attributable to provider-side changes (captured
    # via system_fingerprint, T0.2) rather than to our sampling.
    #
    # NOTE: Reasoning models (GPT-5 + reasoning_effort, Claude opus/sonnet
    # 4+ with effort, Gemini 3 with thinking_level) ignore temperature
    # silently or reject it with 400.  The client pin is skipped
    # automatically in those cases; reproducibility there relies on seed
    # (if supported) plus prompt/data snapshots.  See
    # base_client.apply_determinism_kwargs.
    "llm_temperature": 0.0,
    "llm_seed": 42,
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": "max",    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    "llm_timeout": None,
    "deepseek_reasoning_effort": "max",
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": True,
    # T0.4 — audit archive of checkpoint rows. When True (default), on a
    # successful run the thread's checkpoint state is copied to
    # ``audit_dir/checkpoints/{TICKER}/{date}.db`` BEFORE the rows are
    # cleared from the active DB. Requires ``checkpoint_enabled=True`` to
    # have any effect (otherwise no rows ever exist to archive).
    # ``audit_dir=None`` uses ``~/.tradingagents/audit``.
    "audit_archive_checkpoints": True,
    "audit_dir": None,
    # T0.5 — per-call structured log of every LLM invocation written to
    # ``audit_dir/calls/{session_id}.jsonl``. Each line records the model,
    # provider fingerprint, token counts, latency, LangGraph node tag, and
    # call sequence number — enough for drift detection (T3.4) and the
    # ancestor of the full TraceCallback in Phase 1 (T1.2). Set to False
    # to revert to memory-only stats (pre-T0.5 behavior).
    "audit_jsonl_calls_enabled": True,
    # T1.2 — full prompt + response + tool I/O trace, one TraceRecord
    # per LangChain callback event, written to
    # ``audit_dir/traces/{session_id}.jsonl``. This is the substrate
    # T1.3 (hash chain), T1.7 (replay), and T2.3 (faithfulness
    # interventions) all build on. Disk usage: roughly +1-10 MB per
    # propagate() call depending on debate length. Disable for tight
    # disk budgets, but expect downstream audit features to no-op.
    "audit_full_trace_enabled": True,
    # T1.4 — per-agent prompt version pinning. Each key in this dict
    # corresponds to a file under tradingagents/prompts/, and the value
    # selects which ``.v{N}.txt`` file the agent renders.  To roll out
    # a new prompt: ship ``researchers/bull_researcher.v2.txt`` next
    # to ``v1.txt`` (NEVER delete v1 — historical traces must remain
    # replayable), then bump this dict.  The legacy 4 analyst agents
    # (market/sentiment/news/fundamentals) use ChatPromptTemplate and
    # are not yet on the registry — they will be added in T1.4b.
    "prompt_versions": {
        "researchers/bull_researcher": "v1",
        "researchers/bear_researcher": "v1",
        "managers/research_manager": "v1",
        "managers/portfolio_manager": "v1",
        "trader/trader_system": "v1",
        "trader/trader_user": "v1",
        "risk/aggressive": "v1",
        "risk/conservative": "v1",
        "risk/neutral": "v1",
    },
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Benchmark used for alpha calculation in deferred reflection.
    # When `benchmark_ticker` is set, it wins for every analysis.
    # Otherwise the longest matching suffix in `benchmark_map` is used,
    # falling back to the "" entry for tickers without a known suffix.
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS": "^NSEI",     # Nifty 50 (NSE India)
        ".BO": "^BSESN",    # Sensex (BSE India)
        ".T":  "^N225",     # Nikkei 225 (Japan)
        ".HK": "^HSI",      # Hang Seng (Hong Kong)
        ".L":  "^FTSE",     # FTSE 100 (London)
        ".TO": "^GSPTSE",   # TSX Composite (Toronto)
        "":    "SPY",       # default for US-listed tickers
    },
    # Debate and discussion settings
    "investment_horizon": "medium_term",
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "analyst_concurrency_limit": 5,
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # T0.3 — Snapshot the raw + formatted output of every news fetch so a
    # re-run on the same trade_date sees byte-identical context.  Enabled
    # by default; disable to skip the disk write entirely (e.g. CI runs
    # that only need stats).  ``news_force_refresh=True`` bypasses cache
    # reads, useful when intentionally collecting a new snapshot for
    # post-event analysis without removing the historical one (a new
    # snapshot is written and the older one stays on disk as evidence).
    "news_snapshot_enabled": True,
    "news_snapshot_dir": None,            # None → ~/.tradingagents/snapshots
    "news_force_refresh": False,
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # AgentKey (https://agentkey.app/) — optional. When AGENTKEY_API_KEY is set,
    # the sentiment analyst also pulls Chinese / international social channels
    # (Weibo, Zhihu, and — for consumer-brand sectors — Xiaohongshu, Douyin).
    # Leave the key empty to disable; existing US-only runs are unaffected.
    # Note: AgentKey bills per successful call, so enabling this adds per-run cost.
    "agentkey_api_key": os.getenv("AGENTKEY_API_KEY", ""),
    "agentkey_base_url": os.getenv("AGENTKEY_BASE_URL", "https://api.agentkey.app"),
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance, b3
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance, b3
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance, b3
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance, searxng, b3
        "options_data": "yfinance",          # Options: yfinance (Polygon/Futu via Epic B fallback chain)
        "osint_social": "telegram",          # Options: telegram (Telegram); X tool routes to "x" vendor automatically
    },
    # News parameters
    "ticker_news_count": 20,
    "global_news_look_back_days": 7,
    "global_news_limit": 10,        # yfinance global news article cap
    "av_global_news_limit": 50,     # Alpha Vantage global news article cap (historically 50)
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    "trade_filter_enabled": False,
    "trade_filter_threshold": 0.65,
    "futu_opend_host": "127.0.0.1",
    "futu_opend_port": 11111,
    "telegram_channels": [],
})
