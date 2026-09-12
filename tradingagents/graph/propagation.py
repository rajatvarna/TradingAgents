# TradingAgents/graph/propagation.py

import hashlib
import json
from typing import Any

from tradingagents.agents.schemas import PortfolioContext
from tradingagents.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
)

#: Checkpoint-identity marker for runs started without a portfolio context.
#: Deliberately distinct from the fingerprint of any provided context —
#: including a known-empty one — so "not provided" never resumes (or is
#: resumed by) a portfolio-grounded run.
MISSING_PORTFOLIO_FINGERPRINT = "none"


def normalize_portfolio_context(
    portfolio_context: PortfolioContext | dict | None,
) -> dict | None:
    """Normalize a caller-supplied portfolio context to JSON-safe state data.

    Accepts a ``PortfolioContext`` or its ``dict`` form; ``None`` is passed
    through and means the context was not provided. Validation failures raise
    ``pydantic.ValidationError`` so a malformed snapshot fails at run start
    rather than silently degrading into an empty portfolio.
    """
    if portfolio_context is None:
        return None
    if isinstance(portfolio_context, PortfolioContext):
        context = portfolio_context
    else:
        context = PortfolioContext.model_validate(portfolio_context)
    return context.model_dump(mode="json")


def portfolio_context_fingerprint(
    portfolio_context: PortfolioContext | dict | None,
) -> str:
    """Return a short deterministic fingerprint of a portfolio context.

    The fingerprint folds the whole normalized snapshot (positions, capital,
    timestamp, source, currency) into the checkpoint identity: a crashed run
    resumes only when the snapshot is unchanged, while a changed snapshot
    routes to a different checkpoint thread and starts fresh — never a
    partial resume mixing two snapshots in one run.

    Canonicalization details:

    - ``None`` maps to ``MISSING_PORTFOLIO_FINGERPRINT`` (never equal to a
      provided context, not even a known-empty one).
    - dict key order is irrelevant (``sort_keys=True``).
    - position order is irrelevant (sorted by symbol, then full content).
    - only the 16-char hex digest is returned: no holdings leak into thread
      ids, filenames, or logs. Never uses ``hash()`` (process-randomized).
    """
    if portfolio_context is None:
        return MISSING_PORTFOLIO_FINGERPRINT
    normalized = normalize_portfolio_context(portfolio_context)
    positions = sorted(
        normalized.get("positions", []),
        key=lambda p: (p["symbol"].upper(), json.dumps(p, sort_keys=True)),
    )
    canonical = {**normalized, "positions": positions}
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        user_research: str = "",
        target_profile: dict[str, Any] | None = None,
        risk_constraints: dict[str, Any] | None = None,
        instrument_context: str = "",
        visual_report: str = "",
        strategy_rules: str = "",
        evidence_ledger: dict[str, Any] | None = None,
        evidence_summary: str = "",
        quantitative_anchors: list[dict[str, Any]] | None = None,
        math_guardrail_events: list[dict[str, Any]] | None = None,
        citation_verification: dict[str, Any] | None = None,
        evidence_warnings: list[str] | None = None,
        evidence_strict_mode: str = "warn",
        evidence_strict_blocked: bool = False,
        evidence_decision_status: str = "actionable",
        evidence_actionable: bool = True,
        evidence_blocking_reasons: list[str] | None = None,
        original_final_trade_decision: str | None = None,
        portfolio_context: PortfolioContext | dict | None = None,
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``TradingAgentsGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.

        ``portfolio_context`` is the optional broker-neutral portfolio
        snapshot (see ``PortfolioContext``). When omitted, the state records
        ``None`` so decision nodes can distinguish "not provided" from a
        known flat portfolio.
        """
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "portfolio_context": normalize_portfolio_context(portfolio_context),
            "user_research_report": user_research,
            "risk_constraints": risk_constraints or {},
            "visual_report": visual_report,
            "strategy_rules": strategy_rules,
            "tool_errors": [],
            "tool_call_count": 0,
            "error_count": 0,
            "data_quality": "unknown",
            "structured_valid": False,
            "research_manager_structured_valid": False,
            "trader_structured_valid": False,
            "portfolio_manager_structured_valid": False,
            "confidence_score": 0.0,
            "trade_levels": None,
            "trade_filter_score": 0.0,
            "trade_filter_pass": False,
            "trade_filter_reasons": [],
            "trade_filtered_out": False,
            "trade_filter_details": None,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "options_report": "",
            "esg_report": "",
            "derivatives_report": "",
            "investment_plan": "",
            "trader_investment_plan": "",
            "final_trade_decision": "",
            "sender": "",
            "macro_report": "",
            "source_objects": [],
            "source_registry": {},
            "claim_graph": {},
            "skill_registry": {},
            "recommendation_scorecard": {},
            "pre_synthesis_scope_audit": {},
            "raw_tool_outputs": [],
            "raw_tool_seen_ids": [],
            "target_profile": target_profile or {},
            # Monster Stock / TraderLion framework — pre-computed before graph runs
            "monster_stock_score": {},
            "forensic_score": {},
            "group_sector_report": "",
            "market_phase_report": "",
            "postmortem_report": "",
            "postmortem_past_recommendation": "",
            "postmortem_outcome_data": "",
            # Optional fields initialised to safe defaults
            "conflict_report": {},
            "high_uncertainty": False,
            "analyst_weights": {},
            "holdings_info": {},
            "trading_history_summary": {},
            "prior_pending_orders": [],
            "trading_mode": "live",
            "market_state": {},
            "structure_analysis": {},
            "feature_snapshot": {},
            "structured_strategy": {},
            "evidence_ledger": evidence_ledger or {"items": []},
            "evidence_summary": evidence_summary,
            "quantitative_anchors": list(quantitative_anchors or []),
            "math_guardrail_events": list(math_guardrail_events or []),
            "citation_verification": citation_verification,
            "evidence_warnings": list(evidence_warnings or []),
            "evidence_strict_mode": evidence_strict_mode,
            "evidence_strict_blocked": evidence_strict_blocked,
            "evidence_decision_status": evidence_decision_status,
            "evidence_actionable": evidence_actionable,
            "evidence_blocking_reasons": list(evidence_blocking_reasons or []),
            "original_final_trade_decision": original_final_trade_decision,
        }

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
