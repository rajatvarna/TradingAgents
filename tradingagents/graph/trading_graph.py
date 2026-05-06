# TradingAgents/graph/trading_graph.py

import logging
import os
from pathlib import Path
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.stockstats_utils import yf_retry

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news,
)
from tradingagents.agents.utils.options_tools import get_options_data

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "sentiment", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
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
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
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

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
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
            "options": ToolNode(
                [
                    get_options_data,
                ]
            ),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Resolve the benchmark symbol for a ticker.

        Explicit ``benchmark_ticker`` config wins for every analysis. Otherwise
        the longest matching suffix in ``benchmark_map`` is picked (so ``.BO``
        beats ``""``), falling back to the ``""`` entry, and finally ``SPY``
        if neither config key is set.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map") or {}
        for suffix in sorted((s for s in benchmark_map if s), key=len, reverse=True):
            if ticker.endswith(suffix):
                return benchmark_map[suffix]
        return benchmark_map.get("", "SPY")

    def _fetch_returns(
        self,
        ticker: str,
        trade_date: str,
        holding_days: int = 5,
        benchmark: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        ``benchmark`` may be passed by callers that already resolved it (e.g.
        :meth:`_resolve_pending_entries` does so once per ticker), avoiding
        redundant resolution work in batch loops. When ``None`` it is resolved
        from the ticker.

        Returns (raw_return, alpha_return, actual_holding_days) or
        (None, None, None) if price data is unavailable (too recent, delisted,
        or network error).
        """
        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            if benchmark is None:
                benchmark = self._resolve_benchmark(ticker)
            stock = yf_retry(lambda: yf.Ticker(ticker).history(start=trade_date, end=end_str))
            # Skip the duplicate request when the user analyses the benchmark
            # itself (e.g. SPY vs SPY → alpha is 0 by definition).
            bench = stock if benchmark == ticker else yf_retry(
                lambda: yf.Ticker(benchmark).history(start=trade_date, end=end_str)
            )

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
                "Could not resolve outcome for %s on %s (will retry next run): %s",
                ticker, trade_date, e,
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
                ticker, entry["date"], benchmark=benchmark
            )
            if raw is None:
                continue  # price not available yet — try again next run
            try:
                reflection = self.reflector.reflect_on_final_decision(
                    final_decision=entry.get("decision", ""),
                    raw_return=raw,
                    alpha_return=alpha,
                    benchmark=benchmark,
                )
            except Exception:
                logger.warning(
                    "Failed to reflect on pending entry for %s on %s (will retry next run)",
                    ticker, entry["date"], exc_info=True,
                )
                continue
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

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date.

        When ``checkpoint_enabled`` is set in config, the graph is recompiled
        with a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name

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
            return self._run_graph(company_name, trade_date)
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def _run_graph(self, company_name, trade_date):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Initialize state — inject memory log context for PM.
        past_context = self.memory_log.get_past_context(company_name)
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, past_context=past_context
        )
        args = self.propagator.get_graph_args()

        # Inject thread_id so same ticker+date resumes, different date starts fresh.
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            final_state = None
            for chunk in self.graph.stream(init_agent_state, **args):
                if chunk.get("messages"):
                    chunk["messages"][-1].pretty_print()
                # Only keep the latest chunk — avoids unbounded memory in long runs.
                final_state = chunk
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

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "options_report": final_state.get("options_report", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def propagate_portfolio(
        self,
        tickers: List[str],
        trade_date: str,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """Analyze a basket of tickers in parallel and return a ranked summary.

        Each ticker is analysed independently by cloning the graph's config so
        that checkpoint state, memory log lookups, and YFinance cache remain
        isolated per ticker.  Results are ranked by conviction score: a blend
        of the Portfolio Manager's confidence and directional rating.

        Args:
            tickers: List of ticker symbols to analyse.
            trade_date: Analysis date in YYYY-MM-DD format.
            max_workers: Maximum parallel threads (default 4; keep below API
                rate-limit thresholds for your provider).

        Returns:
            A dict with keys:
                ``results`` — list of per-ticker dicts sorted by ``score`` desc,
                ``summary`` — high-level counts (buy/hold/sell) and top pick.
        """
        if not tickers:
            return {"results": [], "summary": {}}

        def _analyse_one(ticker: str) -> Dict[str, Any]:
            try:
                final_state, signal = self.propagate(ticker, trade_date)
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

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_analyse_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker_sym = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    # Last-resort catch: _analyse_one should never propagate, but
                    # guard against unexpected BaseException subclasses bubbling up.
                    logger.error("Unhandled thread failure for %s: %s", ticker_sym, exc)
                    results.append({
                        "ticker": ticker_sym,
                        "signal": "ERROR",
                        "rating": None,
                        "confidence": 0.0,
                        "score": -999.0,
                        "decision": "",
                        "error": f"Unhandled: {exc}",
                    })

        results.sort(key=lambda r: r["score"], reverse=True)

        # Assign rank after sorting
        for i, r in enumerate(results, start=1):
            r["rank"] = i

        buy_count = sum(1 for r in results if r["signal"] in ("BUY", "OVERWEIGHT"))
        sell_count = sum(1 for r in results if r["signal"] in ("SELL", "UNDERWEIGHT"))
        hold_count = sum(1 for r in results if r["signal"] == "HOLD")
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
        match = re.search(r"\*\*Confidence\*\*:\s*([0-9.]+)", decision_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                logger.debug("Malformed confidence value in decision text: %r", match.group(1))
        else:
            logger.debug("No **Confidence** field found in decision text; defaulting to 0.5")
        return 0.5  # neutral fallback

    @staticmethod
    def _extract_rating(decision_text: str) -> Optional[str]:
        """Parse **Rating**: <value> from the Portfolio Manager's markdown."""
        match = re.search(r"\*\*Rating\*\*:\s*(\w+)", decision_text)
        if not match:
            logger.debug("No **Rating** field found in decision text")
        return match.group(1) if match else None

    @staticmethod
    def _conviction_score(signal: str, confidence: float) -> float:
        """Blend signal direction with confidence into a scalar for ranking.

        Positive score → bullish, negative → bearish, magnitude = conviction.
        """
        direction = {
            "BUY": 2.0,
            "OVERWEIGHT": 1.0,
            "HOLD": 0.0,
            "UNDERWEIGHT": -1.0,
            "SELL": -2.0,
        }.get(signal.upper(), 0.0)
        return direction * confidence

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
