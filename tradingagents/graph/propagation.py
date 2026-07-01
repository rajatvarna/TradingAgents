# TradingAgents/graph/propagation.py

from typing import Any

from tradingagents.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
)


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
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``TradingAgentsGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.
        """
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
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
