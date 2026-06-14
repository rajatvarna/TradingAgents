# TradingAgents/graph/trading_graph.py

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

from langgraph.prebuilt import ToolNode

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.analysts.valuation_analyst import (
    get_dcf_valuation,
    get_ddm_valuation,
    get_roic_analysis,
    get_scenario_analysis,
    get_wacc_components,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_news,
    get_stock_data,
    resolve_instrument_identity,
    resolve_risk_constraints,
)
from tradingagents.agents.utils.core_stock_tools import get_atr_stop_suggestion, get_peer_performance
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.run_cache import reset as reset_run_cache
from tradingagents.dataflows.run_cache import stats as run_cache_stats
from tradingagents.dataflows.symbol_utils import normalize_symbol
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id


def _precompute_monster_score(ticker: str, trade_date: str, config: dict) -> dict:
    """Pre-compute MonsterStockScore before the graph runs.

    Returns a serialised dict (via dataclasses.asdict) or {} on any failure so
    the graph is never blocked by a scoring error.
    """
    if not config.get("monster_stock_mode", True):
        return {}
    try:
        import dataclasses

        from tradingagents.dataflows.fundamentals_deep import fetch_deep_fundamentals
        from tradingagents.dataflows.market_health import fetch_market_health
        from tradingagents.dataflows.sector_groups import fetch_group_leadership
        from tradingagents.dataflows.technicals_deep import compute_deep_technicals
        from tradingagents.scoring.monster_stock_scorer import score_stock

        fund = fetch_deep_fundamentals(ticker)
        tech = compute_deep_technicals(ticker, trade_date)
        mkt = fetch_market_health(trade_date)
        grp = fetch_group_leadership(ticker, trade_date)
        score = score_stock(fund, tech, mkt, grp)
        return dataclasses.asdict(score)
    except Exception as exc:
        logger.warning("Monster Stock pre-score failed for %s on %s: %s", ticker, trade_date, exc)
        return {}
from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SIGNAL_CONVICTION_WEIGHTS, SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: dict[str, Any] = None,
        callbacks: list | None = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []
        self.selected_analysts = selected_analysts

        # IIC-FORGE F1: token accumulator — must be in self.callbacks BEFORE
        # the LLM clients are constructed, so the LLM clients pick it up.
        from tradingagents.graph.cost_callback import CostGuard, RunCostCallback
        _guard = CostGuard(
            per_run_token_budget=self.config.get("max_tokens_per_run", 500_000),
            enabled=self.config.get("cost_guard_enabled", False),
        )
        self._cost_cb = RunCostCallback(cost_guard=_guard)
        self.callbacks = list(self.callbacks or []) + [self._cost_cb]

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        self.memory_log = TradingMemoryLog(self.config)

        # IIC-FORGE F1: per-run id + persistence + Run Recorder
        import uuid

        from tradingagents.graph.run_recorder import RunRecorder, make_run_recorder_node
        from tradingagents.persistence.db import connect as _iic_connect

        self.run_id = uuid.uuid4().hex
        self._iic_conn = _iic_connect(self.config["iic_db_path"])
        self.run_recorder = RunRecorder(
            conn=self._iic_conn,
            data_dir=self.config["iic_data_dir"],
            run_id=self.run_id,
            persona_id=self.config.get("persona_id"),
            cost_callback=self._cost_cb,
            queue_job_id=self.config.get("queue_job_id"),
        )
        run_recorder_node = make_run_recorder_node(self.run_recorder)

        self.structured_output_cache: dict[str, str] = {}

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            self.structured_output_cache,
            analyst_concurrency_limit=self.config.get("analyst_concurrency_limit", 1),
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(
            selected_analysts, run_recorder_node=run_recorder_node
        )
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        # Sampling temperature is cross-provider: forward it whenever set.
        # float() here so a value coming from a TRADINGAGENTS_TEMPERATURE env
        # string ("0.2") works the same as a programmatic float.
        temperature = self.config.get("temperature")
        if temperature is not None and temperature != "":
            kwargs["temperature"] = float(temperature)

        # Forward timeout / retry budget to every provider's SDK. Without
        # these the SDK defaults are short enough that a single transient
        # APIConnectionError on a long depth-10 run takes down the graph.
        timeout = self.config.get("llm_timeout")
        if timeout is not None and timeout != "":
            kwargs["timeout"] = float(timeout)
        max_retries = self.config.get("llm_max_retries")
        if max_retries is not None and max_retries != "":
            kwargs["max_retries"] = int(max_retries)

        return kwargs

    def _risk_constraints_from_config(self) -> dict[str, Any]:
        """Return session risk constraints to persist outside message history."""
        return resolve_risk_constraints(self.config)

    def _create_tool_nodes(self) -> dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                    # Sector peer relative strength
                    get_peer_performance,
                    # ATR-based dynamic stop-loss suggestions
                    get_atr_stop_suggestion,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
            "valuation": ToolNode(
                [
                    get_wacc_components,
                    get_roic_analysis,
                    get_dcf_valuation,
                    get_ddm_valuation,
                    get_scenario_analysis,
                ]
            ),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()
        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark
        return benchmark_map.get("", "SPY")

    @staticmethod
    def _coerce_positive_int(value, default: int) -> int:
        """Coerce value to a positive int, falling back to default on failure."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = None,
        benchmark: str = "SPY",
    ) -> tuple[float | None, float | None, int | None]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        ``benchmark`` is the index used as the alpha baseline (resolved by the
        caller via ``_resolve_benchmark``). Returns ``(raw_return, alpha_return,
        actual_holding_days)`` or ``(None, None, None)`` if price data is
        unavailable (too recent, delisted, or network error).
        """
        if holding_days is None:
            cfg = getattr(self, "config", {}) or {}
            holding_days = self._coerce_positive_int(cfg.get("outcome_holding_days", 5), 5)
        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            yahoo_symbol = normalize_symbol(ticker)
            yahoo_bench = normalize_symbol(benchmark)
            stock = yf.Ticker(yahoo_symbol).history(start=trade_date, end=end_str)
            bench = yf.Ticker(yahoo_bench).history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(bench) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        benchmark = self._resolve_benchmark(ticker)
        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(
                ticker, entry["date"], benchmark=benchmark,
            )
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
                benchmark_name=benchmark,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def resolve_instrument_context(self, ticker: str, asset_type: str = "stock") -> str:
        """Resolve ticker identity once and return the full instrument context.

        Deterministic yfinance lookup (cached, fail-open) injected into a
        context string so every agent anchors to the real company instead of
        hallucinating one from the price chart (#814). Both the propagate()
        path and the CLI call this so the resolved identity reaches the whole
        graph regardless of entry point.
        """
        identity = resolve_instrument_identity(ticker)
        return build_instrument_context(ticker, asset_type, identity)

    def propagate(
        self,
        company_name,
        trade_date,
        asset_type: str = "stock",
        on_chunk=None,
        progress_callback=None,
        target_profile=None,
    ):
        """Run the trading agents graph for a company on a specific date.

        ``asset_type`` selects between the stock pipeline (default) and the
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name
        self.structured_output_cache.clear()

        # Reset per-run data cache at the start of each analysis run.
        reset_run_cache()

        # Apply market calendar guard — shift weekend/holiday dates to nearest trading day.
        from tradingagents.graph.market_calendar import nearest_trading_day
        exchange = self.config.get("benchmark_exchange", "NYSE")
        trade_date, adjusted = nearest_trading_day(str(trade_date), exchange)
        if adjusted:
            logger.warning("Analysis date shifted to nearest trading day: %s", trade_date)

        # Resolve any pending memory-log entries for this ticker before the pipeline runs.
        self._resolve_pending_entries(company_name)

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(
                company_name,
                trade_date,
                asset_type=asset_type,
                on_chunk=on_chunk,
                progress_callback=progress_callback,
                target_profile=target_profile,
            )
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def _run_graph(
        self,
        company_name,
        trade_date,
        asset_type: str = "stock",
        on_chunk=None,
        progress_callback=None,
        target_profile=None,
    ):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Initialize state — inject memory log context for PM and the
        # deterministically resolved instrument identity for all agents.
        past_context = self.memory_log.get_past_context(company_name)
        instrument_context = self.resolve_instrument_context(company_name, asset_type)
        risk_constraints = self._risk_constraints_from_config()

        monster_score = _precompute_monster_score(company_name, str(trade_date), self.config)

        init_agent_state = self.propagator.create_initial_state(
            company_name,
            trade_date,
            asset_type=asset_type,
            past_context=past_context,
            instrument_context=instrument_context,
            risk_constraints=risk_constraints,
            target_profile=target_profile,
        )
        if monster_score:
            init_agent_state["monster_stock_score"] = monster_score

        # Earnings calendar awareness — warn before analysis if earnings are near.
        from tradingagents.dataflows.earnings_calendar import get_earnings_warning
        earnings_warning = get_earnings_warning(
            company_name,
            str(trade_date),
            lookahead_days=self.config.get("earnings_lookahead_days", 7),
        )
        if earnings_warning["has_warning"]:
            logger.warning("Earnings warning for %s: %s", company_name, earnings_warning["message"])
        init_agent_state["earnings_warning"] = earnings_warning

        # IIC-FORGE F4: event-context injection — seed event text into state.
        # Empty string when not in event_alert mode (deep-dive path unchanged).
        init_agent_state["event_context_text"] = self.config.get("event_context", "") or ""

        args = self.propagator.get_graph_args(callbacks=self.callbacks)

        from datetime import datetime
        self.run_recorder.start(
            ticker=init_agent_state.get("company_of_interest", "UNKNOWN"),
            started_ts=datetime.now(UTC).isoformat(),
        )

        # Inject thread_id so same ticker+date resumes, different date starts fresh.
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug or on_chunk is not None:
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if on_chunk is not None:
                    on_chunk(chunk)
                elif chunk.get("messages"):
                    chunk["messages"][-1].pretty_print()
                trace.append(chunk)
            # stream_mode='values' yields cumulative state snapshots. Merging is
            # still harmless and keeps returned state parity with graph.invoke().
            final_state = {}
            for chunk in trace:
                final_state.update(chunk)
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection.
        self.curr_state = final_state

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run.
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )

        logger.info("run_cache stats: %s", run_cache_stats())

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        ids = final_state.get("investment_debate_state") or {}
        rds = final_state.get("risk_debate_state") or {}
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state.get("company_of_interest"),
            "trade_date": final_state.get("trade_date"),
            "market_report": final_state.get("market_report"),
            "sentiment_report": final_state.get("sentiment_report"),
            "news_report": final_state.get("news_report"),
            "fundamentals_report": final_state.get("fundamentals_report"),
            "risk_constraints": final_state.get("risk_constraints", {}),
            "investment_debate_state": {
                "bull_history": ids.get("bull_history"),
                "bear_history": ids.get("bear_history"),
                "history": ids.get("history"),
                "current_response": ids.get("current_response"),
                "judge_decision": ids.get("judge_decision"),
            },
            "trader_investment_decision": final_state.get("trader_investment_plan"),
            "risk_debate_state": {
                "aggressive_history": rds.get("aggressive_history"),
                "conservative_history": rds.get("conservative_history"),
                "neutral_history": rds.get("neutral_history"),
                "history": rds.get("history"),
                "judge_decision": rds.get("judge_decision"),
            },
            "investment_plan": final_state.get("investment_plan"),
            "final_trade_decision": final_state.get("final_trade_decision"),
        }

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)

    def propagate_portfolio(
        self,
        tickers: list[str],
        trade_date: str,
        max_workers: int = None,
    ) -> dict[str, Any]:
        """Analyze a basket of tickers in parallel and return a ranked summary.

        Each ticker is analysed independently by cloning the graph's config so
        that checkpoint state, memory log lookups, and YFinance cache remain
        isolated per ticker.  Results are ranked by conviction score: a blend
        of the Portfolio Manager's confidence and directional rating.

        Args:
            tickers: List of ticker symbols to analyse.
            trade_date: Analysis date in YYYY-MM-DD format.
            max_workers: Maximum parallel threads (default 4; keep below API
                rate-limit thresholds for your provider).  Set to 1 to run
                sequentially if your ``propagate`` implementation is not
                thread-safe (e.g. when using a shared SQLite checkpointer).

        Returns:
            A dict with keys:
                ``results`` — list of per-ticker dicts sorted by ``score`` desc,
                ``summary`` — high-level counts (buy/hold/sell) and top pick.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if max_workers is None:
            max_workers = self._coerce_positive_int(
                self.config.get("portfolio_propagation_max_workers", 4), 4
            )
        if not tickers:
            return {"results": [], "summary": {}}

        # Capture the propagate method reference before entering threads so that
        # test patches on self.propagate are honoured, and so that subclasses
        # overriding propagate are respected.  Under true concurrency, callers
        # must ensure their propagate implementation is thread-safe or pass
        # max_workers=1.
        _propagate = self.propagate

        def _analyse_one(ticker: str) -> dict[str, Any]:
            try:
                final_state, signal = _propagate(ticker, trade_date)
                decision_text = final_state.get("final_trade_decision", "")
                confidence = self._extract_confidence(decision_text)
                rating = self._extract_rating(decision_text)
                score = self._conviction_score(signal, confidence)
                return {
                    "ticker": ticker,
                    "signal": signal,
                    "rating": rating,
                    "confidence": confidence,
                    "score": score,
                    "decision": decision_text,
                    "error": None,
                }
            except Exception as exc:
                logger.error("Portfolio analysis failed for %s: %s", ticker, exc)
                return {
                    "ticker": ticker,
                    "signal": "ERROR",
                    "rating": None,
                    "confidence": 0.0,
                    "score": -999.0,
                    "decision": "",
                    "error": str(exc),
                }

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_analyse_one, t): t for t in tickers}
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: r["score"], reverse=True)

        # Assign rank after sorting
        for i, r in enumerate(results, start=1):
            r["rank"] = i

        buy_count = sum(1 for r in results if r["signal"].title() in ("Buy", "Overweight"))
        sell_count = sum(1 for r in results if r["signal"].title() in ("Sell", "Underweight"))
        hold_count = sum(1 for r in results if r["signal"].title() == "Hold")
        top_pick = results[0]["ticker"] if results else None

        return {
            "results": results,
            "summary": {
                "trade_date": trade_date,
                "total": len(results),
                "buy": buy_count,
                "hold": hold_count,
                "sell": sell_count,
                "top_pick": top_pick,
            },
        }

    @staticmethod
    def _extract_confidence(decision_text: str) -> float:
        """Parse **Confidence**: 0.XX from the Portfolio Manager's markdown."""
        import re
        match = re.search(r"\*\*Confidence\*\*:\s*([0-9.]+)", decision_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5  # neutral fallback

    @staticmethod
    def _extract_rating(decision_text: str) -> str | None:
        """Parse **Rating**: <value> from the Portfolio Manager's markdown."""
        import re
        match = re.search(r"\*\*Rating\*\*:\s*(\w+)", decision_text)
        return match.group(1) if match else None

    @staticmethod
    def _conviction_score(signal: str, confidence: float) -> float:
        """Blend signal direction with confidence into a scalar for ranking.

        Positive score -> bullish, negative -> bearish, magnitude = conviction.
        """
        # Normalise to title-case to match SIGNAL_CONVICTION_WEIGHTS keys.
        direction = SIGNAL_CONVICTION_WEIGHTS.get(signal.title(), 0.0)
        return direction * confidence
