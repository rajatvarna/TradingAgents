"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.claims import build_claim_graph
from tradingagents.agents.source_registry import build_source_registry
from tradingagents.agents.skills import build_skill_registry
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_scope_guard,
    format_risk_constraints,
    get_language_instruction,
)
from tradingagents.agents.utils.recommendation_audit import (
    build_pre_synthesis_scope_audit,
    build_raw_tool_source_objects,
    build_recommendation_scorecard,
    build_source_objects,
    render_raw_tool_sources_for_prompt,
    render_scorecard_for_prompt,
    render_scope_audit_for_prompt,
    render_sources_for_prompt,
)
from tradingagents.agents.utils.rating import RATINGS_5_TIER, extract_rating, parse_rating
from tradingagents.agents.utils.trade_filter import compute_trade_filter
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext_with_meta,
)
from tradingagents.prompts import load_prompt
from tradingagents.dataflows.config import get_config
from tradingagents.audit.prompt_registry import default_registry


def create_portfolio_manager(llm, cache=None, prompt_registry=None):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")
    registry = prompt_registry or default_registry()

    rating_order = list(RATINGS_5_TIER)

    def _data_quality(error_count: int) -> str:
        if error_count <= 0:
            return "high"
        if error_count <= 2:
            return "medium"
        return "low"

    def _shift_towards_hold(rating: str, steps: int = 1) -> str:
        if rating not in rating_order:
            return "Hold"
        idx = rating_order.index(rating)
        hold_idx = rating_order.index("Hold")
        for _ in range(max(0, steps)):
            if idx < hold_idx:
                idx += 1
            elif idx > hold_idx:
                idx -= 1
        return rating_order[idx]

    def _replace_rating(text: str, new_rating: str) -> str:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.lower().lstrip().startswith("**rating**") or line.lower().lstrip().startswith("rating"):
                if ":" in line:
                    prefix = line.split(":", 1)[0]
                    lines[i] = f"{prefix}: {new_rating}"
                    return "\n".join(lines)
        return f"**Rating**: {new_rating}\n\n{text}"

    def _compute_confidence(
        *,
        rating: str,
        data_quality: str,
        error_count: int,
        structured_valid: bool,
        research_plan: str,
        trader_plan: str,
    ) -> float:
        score = 0.85
        if data_quality == "medium":
            score -= 0.1
        elif data_quality == "low":
            score -= 0.25
        score -= 0.1 * min(max(error_count, 0), 5)
        if not structured_valid:
            score -= 0.15

        rm_rating = extract_rating(research_plan or "")
        if rm_rating and rm_rating == rating:
            score += 0.05

        action = None
        tp = (trader_plan or "").lower()
        if "final transaction proposal" in tp:
            if "**buy**" in tp or " buy" in tp:
                action = "Buy"
            elif "**sell**" in tp or " sell" in tp:
                action = "Sell"
            elif "**hold**" in tp or " hold" in tp:
                action = "Hold"
        if action and action == rating:
            score += 0.05

        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return float(score)

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        scope_guard = build_scope_guard(state["company_of_interest"])
        constraints_block = format_risk_constraints(state.get("risk_constraints", {}))

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        audit_state = {
            "market_report": state.get("market_report", ""),
            "news_report": state.get("news_report", ""),
            "sentiment_report": state.get("sentiment_report", ""),
            "fundamentals_report": state.get("fundamentals_report", ""),
            "risk_debate_state": risk_debate_state,
            "raw_tool_outputs": state.get("raw_tool_outputs", []),
            "macro_report": state.get("macro_report", ""),
            "target_profile": state.get("target_profile", {}),
        }
        source_objects = build_source_objects(audit_state)
        raw_tool_sources = build_raw_tool_source_objects(audit_state)
        source_registry = build_source_registry(audit_state, extra_sources=source_objects + raw_tool_sources)
        claim_graph = build_claim_graph(audit_state, source_registry)
        skill_registry = build_skill_registry(audit_state, source_registry, claim_graph)
        recommendation_scorecard = build_recommendation_scorecard(audit_state)
        pre_synthesis_scope_audit = build_pre_synthesis_scope_audit(
            state["company_of_interest"],
            audit_state,
        )

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        error_count = int(state.get("error_count", 0) or 0)
        data_quality = _data_quality(error_count)
        rm_structured_valid = bool(state.get("research_manager_structured_valid", False))
        trader_structured_valid = bool(state.get("trader_structured_valid", False))
        upstream_structured_valid = rm_structured_valid and trader_structured_valid
        trade_levels = state.get("trade_levels")

        reliability_constraints = ""
        if data_quality in ("low",) or error_count > 2:
            reliability_constraints = (
                "\n\n---\n\n"
                "**Reliability Constraints (must follow):**\n"
                "- If data_quality is low OR error_count > 2: "
                "you may NOT output Buy or Sell. Only Hold or Underweight are allowed.\n"
                "- If error_count >= 5: output Hold.\n"
            )

        version = state.get("prompt_versions", {}).get("managers/portfolio_manager", "v1")
        prompt, prompt_hash = registry.render(
            "managers/portfolio_manager",
            version=version,
            instrument_context=instrument_context,
            scope_guard=scope_guard,
            research_plan=research_plan,
            trader_plan=trader_plan,
            lessons_line=lessons_line,
            history=history,
            sources=render_sources_for_prompt(source_objects),
            source_registry=source_registry,
            claim_graph=claim_graph,
            skill_registry=skill_registry,
            raw_tool_sources=render_raw_tool_sources_for_prompt(raw_tool_sources),
            pre_synthesis_scope_audit=render_scope_audit_for_prompt(pre_synthesis_scope_audit),
            recommendation_scorecard=render_scorecard_for_prompt(recommendation_scorecard),
            language_instruction=get_language_instruction(),
        )

        prompt = constraints_block + prompt

        reliability_signals = (
            f"\n\n**Reliability Signals:**\n"
            f"- data_quality: {data_quality}\n"
            f"- error_count: {error_count}\n"
            f"- upstream_structured_valid: {upstream_structured_valid}\n"
        )
        prompt += reliability_signals + reliability_constraints

        final_trade_decision, pm_structured_valid = invoke_structured_or_freetext_with_meta(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
            cache=cache,
            config={
                "metadata": {
                    "prompt_key": "managers/portfolio_manager",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                }
            },
        )

        structured_valid = bool(upstream_structured_valid and pm_structured_valid)
        rating = parse_rating(final_trade_decision)

        trade_filter_result = None
        trade_filter_score = 0.0
        trade_filter_reasons = []
        trade_filter_pass = True
        trade_filtered_out = False
        cfg = get_config()
        if cfg.get("trade_filter_enabled"):
            rm_rating = extract_rating(research_plan or "")
            trader_action = extract_rating(trader_plan or "")
            trade_filter_result = compute_trade_filter(
                trade_levels=trade_levels,
                rating=rating,
                rm_rating=rm_rating,
                trader_action=trader_action,
                data_quality=data_quality,
                error_count=error_count,
                structured_valid=structured_valid,
                threshold=float(cfg.get("trade_filter_threshold", 0.65) or 0.65),
            )
            trade_filter_score = float(trade_filter_result.get("score", 0.0) or 0.0)
            trade_filter_pass = bool(trade_filter_result.get("pass", True))
            trade_filtered_out = bool(trade_filter_result.get("filtered_out", False))
            trade_filter_reasons = list(trade_filter_result.get("hard_reject_reasons", [])) + list(
                trade_filter_result.get("reasons", [])
            )
            if trade_filtered_out and rating in ("Buy", "Sell"):
                rating = "Hold"

        if error_count >= 5:
            rating = "Hold"
        elif (data_quality == "low") or (error_count > 2):
            if rating in ("Buy", "Sell"):
                rating = _shift_towards_hold(rating, steps=2)
            if rating not in ("Hold", "Underweight"):
                rating = _shift_towards_hold(rating, steps=1)

        confidence_score = _compute_confidence(
            rating=rating,
            data_quality=data_quality,
            error_count=error_count,
            structured_valid=structured_valid,
            research_plan=research_plan,
            trader_plan=trader_plan,
        )

        final_trade_decision = _replace_rating(final_trade_decision, rating)
        final_trade_decision = (
            f"{final_trade_decision}\n\n"
            f"**Data Quality**: {data_quality}\n"
            f"**Error Count**: {error_count}\n"
            f"**Structured Valid**: {structured_valid}\n"
            f"**Confidence**: {confidence_score:.2f}\n"
            f"**Trade Filter Score**: {trade_filter_score:.2f}\n"
            f"**Trade Filter Pass**: {trade_filter_pass}\n"
            f"**Trade Filtered Out**: {trade_filtered_out}"
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "source_objects": source_objects + raw_tool_sources,
            "source_registry": source_registry,
            "claim_graph": claim_graph,
            "skill_registry": skill_registry,
            "recommendation_scorecard": recommendation_scorecard,
            "pre_synthesis_scope_audit": pre_synthesis_scope_audit,
            "portfolio_manager_structured_valid": pm_structured_valid,
            "structured_valid": structured_valid,
            "data_quality": data_quality,
            "error_count": error_count,
            "confidence_score": confidence_score,
            "trade_levels": trade_levels,
            "trade_filter_score": trade_filter_score,
            "trade_filter_pass": trade_filter_pass,
            "trade_filter_reasons": trade_filter_reasons,
            "trade_filtered_out": trade_filtered_out,
            "trade_filter_details": trade_filter_result,
        }

    return portfolio_manager_node


from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


BROAD_INDEX_TICKERS = {
    "SPY",
    "VOO",
    "IVV",
    "QQQ",
    "QQQM",
    "DIA",
    "IWM",
    "VTI",
    "VT",
    "^GSPC",
    "^IXIC",
    "^DJI",
    "^RUT",
}


class PriceSizeBlock(BaseModel):
    price: Optional[float] = Field(default=None, description="Plain limit price, or null.")
    size_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class StopLossBlock(BaseModel):
    price: Optional[float] = Field(default=None, description="Plain stop price, or null.")


class PortfolioStrategy(BaseModel):
    schema_version: Literal["v3"] = "v3"
    ticker: str
    as_of_date: str = Field(description="YYYY-MM-DD analysis date.")
    action: Literal["BUY", "HOLD", "SELL"]
    entry: PriceSizeBlock
    add_position: PriceSizeBlock
    take_profit: PriceSizeBlock = Field(
        description="Sell on rise. size_pct=100 means full close."
    )
    reduce_stop: PriceSizeBlock = Field(
        description="Partial defensive sell on drop. Price must sit above stop_loss."
    )
    stop_loss: StopLossBlock = Field(
        description="Full-close stop. closes 100% of the position."
    )
    rationale_summary: str

    @model_validator(mode="after")
    def normalize_sell_orders(self):
        if self.action == "SELL":
            self.entry = PriceSizeBlock()
            self.add_position = PriceSizeBlock()
            self.take_profit = PriceSizeBlock()
            self.reduce_stop = PriceSizeBlock()
        return self

    @model_validator(mode="after")
    def validate_risk_levels(self):
        if self.action == "SELL":
            return self
        rs_price = self.reduce_stop.price
        sl_price = self.stop_loss.price
        if (
            rs_price is not None
            and sl_price is not None
            and self.reduce_stop.size_pct > 0
            and rs_price <= sl_price
        ):
            raise ValueError(
                f"reduce_stop.price ({rs_price}) must be ABOVE stop_loss.price ({sl_price})"
            )
        if self.action == "BUY" and self.entry.size_pct > 0 and sl_price is None:
            raise ValueError("BUY with a non-zero entry must define stop_loss.price.")
        if self.add_position.size_pct > 0 and sl_price is None:
            raise ValueError("add_position with size_pct > 0 must be paired with stop_loss.price.")
        return self


def _is_broad_index_instrument(ticker: str) -> bool:
    normalized = ticker.upper()
    return normalized in BROAD_INDEX_TICKERS or normalized.endswith((".INDEX", ".IDX"))


def _classify_volume_regime(volume_ratio: Optional[float]) -> str:
    if volume_ratio is None:
        return "unavailable"
    if volume_ratio >= 1.5:
        return "expanding"
    if volume_ratio < 0.7:
        return "shrinking"
    if volume_ratio >= 0.9:
        return "normal"
    return "soft"


def _clamp_size(block: dict, max_size: float) -> None:
    block["size_pct"] = min(float(block.get("size_pct") or 0.0), float(max_size))
    if block["size_pct"] <= 0:
        block["size_pct"] = 0.0
        block["price"] = None


def _distance_pct(price: Optional[float], current_price: Optional[float]) -> Optional[float]:
    if price is None or current_price in (None, 0):
        return None
    return abs(float(price) - float(current_price)) / float(current_price) * 100.0


def _append_rule_note(strategy: dict, note: str) -> None:
    rationale = strategy.get("rationale_summary") or ""
    if note not in rationale:
        strategy["rationale_summary"] = (rationale + " Rule adjustment: " + note).strip()


def _clear_entry_orders(strategy: dict) -> None:
    strategy["entry"] = PriceSizeBlock().model_dump()
    strategy["add_position"] = PriceSizeBlock().model_dump()


def _enforce_strategy_rules(strategy: dict, anchors: Optional[dict], constraints: dict, holdings_info: dict) -> dict:
    strategy = PortfolioStrategy.model_validate(strategy).model_dump()
    has_position = float(holdings_info.get("quantity") or 0.0) > 0
    current_price = anchors.get("current_price") if anchors else None

    if strategy["action"] not in constraints["allowed_actions"]:
        original_action = strategy["action"]
        strategy["action"] = "HOLD" if original_action == "SELL" or has_position else "BUY"
        if strategy["action"] not in constraints["allowed_actions"]:
            strategy["action"] = constraints["allowed_actions"][0]
        _append_rule_note(
            strategy,
            f"{original_action} was outside allowed_actions={constraints['allowed_actions']}; action set to {strategy['action']}.",
        )
        if original_action == "SELL":
            _clear_entry_orders(strategy)

    _clamp_size(strategy["entry"], constraints["max_entry_size_pct"])
    _clamp_size(strategy["add_position"], constraints["max_add_position_size_pct"])

    if constraints["entry_mode"] == "no_new_or_add":
        _clear_entry_orders(strategy)
        if not has_position and strategy["action"] == "BUY":
            strategy["action"] = "HOLD"
        _append_rule_note(strategy, "new entries and adds blocked by deterministic volume divergence rule.")

    entry_distance = _distance_pct(strategy["entry"].get("price"), current_price)
    if entry_distance is not None and entry_distance > 10:
        strategy["entry"]["size_pct"] = 0.0
        strategy["entry"]["price"] = None
        _append_rule_note(strategy, "entry more than 10% from current price was removed.")

    add_distance = _distance_pct(strategy["add_position"].get("price"), current_price)
    if add_distance is not None and add_distance > 10:
        strategy["add_position"]["size_pct"] = 0.0
        strategy["add_position"]["price"] = None
        _append_rule_note(strategy, "add-position level more than 10% from current price was removed.")

    if strategy["action"] == "BUY" and strategy["entry"]["size_pct"] <= 0 and strategy["add_position"]["size_pct"] <= 0:
        strategy["action"] = "HOLD"
        _append_rule_note(strategy, "BUY without an executable entry/add was converted to HOLD.")

    if strategy["action"] in ("BUY", "HOLD") and (
        strategy["entry"]["size_pct"] > 0 or strategy["add_position"]["size_pct"] > 0
    ):
        if strategy["stop_loss"]["price"] is None and anchors:
            reference = strategy["entry"]["price"] or current_price
            stop = min(
                anchors.get("nearest_support") or reference,
                float(reference) - 1.5 * float(anchors["atr14"]),
            )
            strategy["stop_loss"]["price"] = round(max(stop, 0.01), 4)
            _append_rule_note(strategy, "missing stop_loss was filled from support/ATR anchor.")

    return PortfolioStrategy.model_validate(strategy).model_dump()
