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

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
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
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Request timeout in seconds for LLM API calls. None uses the provider
    # default. Increase for slow local inference (LM Studio, Ollama, vLLM).
    "llm_timeout": None,
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
    "analyst_concurrency_limit": 1,
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
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance, b3
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance, b3
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance, b3
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance, searxng, b3
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
}
