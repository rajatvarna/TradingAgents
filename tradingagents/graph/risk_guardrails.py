# TradingAgents/tradingagents/graph/risk_guardrails.py
#
# Deterministic safety net that runs AFTER the Portfolio Manager's LLM-based
# decision. Enforces hard, non-negotiable risk limits that no amount of
# LLM "reasoning" can override.
#
# Philosophy: LLMs are great at qualitative assessment (thesis, sentiment,
# edge-case reasoning). They are terrible at quantitative discipline
# (position sizing, max drawdown, correlation risk). Let the LLM debate
# the thesis. Let math protect the capital.

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """A single existing portfolio position for heat calculation."""

    ticker: str
    position_pct: float   # % of portfolio allocated to this position
    stop_loss_pct: float  # % loss from entry that triggers the stop


@dataclass
class GuardrailConfig:
    """Hard risk limits. These are NOT suggestions — they are circuit breakers.

    Set via config dict keys (all optional, sensible defaults):
        risk_guardrails_enabled: bool = False
        max_position_pct: float = 25.0
        max_single_loss_pct: float = 5.0
        require_stop_loss: bool = True
        blocked_ratings: list[str] = []  # e.g. ["Buy"] to prevent buys
        max_portfolio_heat_pct: float = 20.0  # max total portfolio heat (sum of position_pct * stop_loss_pct / 100)
        portfolio_positions: list[PortfolioPosition] = []  # existing open positions
    """

    enabled: bool = False
    max_position_pct: float = 25.0       # max % of portfolio in one position
    max_single_loss_pct: float = 5.0     # max loss per trade before forced exit
    require_stop_loss: bool = True       # reject Buy/Overweight without stop-loss
    blocked_ratings: list = field(default_factory=list)  # hard-block certain actions
    portfolio_tickers: list = field(default_factory=list)
    trade_date_str: str | None = None
    ticker: str | None = None
    max_portfolio_heat_pct: float = 20.0  # max aggregate portfolio heat
    portfolio_positions: list = field(default_factory=list)  # list of PortfolioPosition


@dataclass
class GuardrailResult:
    """Outcome of a guardrail check."""

    original_decision: str
    modified_decision: str
    was_modified: bool
    violations: list  # human-readable list of triggered rules
    clamped_fields: dict  # field → (original, clamped) pairs


class RiskGuardrails:
    """Deterministic post-PM safety layer.

    Runs after the Portfolio Manager outputs its decision. Parses the
    structured markdown, checks each field against hard limits, and
    either passes the decision through or clamps/overrides it.

    This node is intentionally NOT an LLM call — it is pure Python
    validation. LLMs cannot be trusted with capital preservation logic.
    """

    def __init__(self, config: dict):
        raw_positions = config.get("portfolio_positions") or []
        positions = []
        for p in raw_positions:
            if isinstance(p, PortfolioPosition):
                positions.append(p)
            elif isinstance(p, dict):
                try:
                    positions.append(
                        PortfolioPosition(
                            ticker=str(p["ticker"]),
                            position_pct=float(p["position_pct"]),
                            stop_loss_pct=float(p["stop_loss_pct"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    pass

        gc = GuardrailConfig(
            enabled=config.get("risk_guardrails_enabled", False),
            max_position_pct=config.get("max_position_pct", 25.0),
            max_single_loss_pct=config.get("max_single_loss_pct", 5.0),
            require_stop_loss=config.get("require_stop_loss", True),
            blocked_ratings=[
                r.lower() for r in config.get("blocked_ratings", [])
            ],
            portfolio_tickers=list(config.get("portfolio_tickers") or []),
            trade_date_str=config.get("trade_date_str"),
            ticker=config.get("ticker"),
            max_portfolio_heat_pct=config.get("max_portfolio_heat_pct", 20.0),
            portfolio_positions=positions,
        )
        self.gc = gc

    def check(self, final_trade_decision: str) -> GuardrailResult:
        """Validate the PM's decision against hard risk limits.

        Args:
            final_trade_decision: The markdown string from the Portfolio Manager

        Returns:
            GuardrailResult with original and (possibly modified) decision
        """
        if not self.gc.enabled:
            return GuardrailResult(
                original_decision=final_trade_decision,
                modified_decision=final_trade_decision,
                was_modified=False,
                violations=[],
                clamped_fields={},
            )

        violations = []
        clamped = {}
        decision = final_trade_decision

        # ── 1. Blocked ratings ──
        rating = self._extract_field(decision, "Rating")
        if rating and rating.lower() in self.gc.blocked_ratings:
            violations.append(
                f"BLOCKED: Rating '{rating}' is in blocked_ratings list. "
                f"Overriding to Hold."
            )
            decision = self._replace_field(decision, "Rating", "Hold")
            clamped["Rating"] = (rating, "Hold")

        # ── 2. Position sizing cap ──
        sizing = self._extract_field(decision, "Position Sizing")
        if sizing:
            pct = self._extract_percentage(sizing)
            # Apply correlation-aware size reduction before the hard cap check.
            if pct is not None:
                try:
                    from tradingagents.graph.correlation_guard import (
                        apply_correlation_sizing_adjustment,
                    )
                    existing = list(self.gc.portfolio_tickers)
                    if existing and self.gc.trade_date_str and self.gc.ticker:
                        original_pct = pct
                        pct, corr_note = apply_correlation_sizing_adjustment(
                            pct, self.gc.ticker, existing, self.gc.trade_date_str
                        )
                        if pct < original_pct:
                            corr_capped = (
                                f"{pct:.0f}% of portfolio "
                                f"(reduced by correlation guard from {original_pct:.0f}%)"
                            )
                            decision = self._replace_field(decision, "Position Sizing", corr_capped)
                            clamped["Position Sizing"] = (sizing, corr_capped)
                            if corr_note:
                                violations.append(corr_note)
                except Exception as exc:
                    logger.debug("Correlation sizing adjustment skipped due to error: %s", exc)
            if pct is not None and pct > self.gc.max_position_pct:
                capped = f"{self.gc.max_position_pct:.0f}% of portfolio (clamped from {pct:.0f}%)"
                violations.append(
                    f"CLAMPED: Position size {pct:.0f}% exceeds max "
                    f"{self.gc.max_position_pct:.0f}%. Reduced."
                )
                decision = self._replace_field(decision, "Position Sizing", capped)
                clamped["Position Sizing"] = (sizing, capped)

        # ── 3. Portfolio heat budget ──
        # Heat = position_pct * stop_loss_pct / 100 for each position.
        # stop_loss_pct is derived from entry and stop prices in the decision.
        rating_for_heat = self._extract_field(decision, "Rating") or ""
        if rating_for_heat.lower() in ("buy", "overweight"):
            sizing_after = self._extract_field(decision, "Position Sizing")
            new_pos_pct = self._extract_percentage(sizing_after or "") if sizing_after else None
            entry_price = self._extract_number(self._extract_field(decision, "Entry Price") or "")
            stop_price = self._extract_number(self._extract_field(decision, "Stop Loss") or "")
            new_stop_pct: float | None = None
            if entry_price and stop_price and entry_price > 0:
                new_stop_pct = abs(entry_price - stop_price) / entry_price * 100.0
            if new_pos_pct is not None and new_stop_pct is not None:
                existing_heat = sum(
                    (p.position_pct * p.stop_loss_pct / 100.0)
                    for p in self.gc.portfolio_positions
                )
                new_heat = new_pos_pct * new_stop_pct / 100.0
                total_heat = existing_heat + new_heat
                budget = self.gc.max_portfolio_heat_pct
                if total_heat > budget:
                    # Scale new position down so total heat stays within budget
                    remaining_budget = max(0.0, budget - existing_heat)
                    if new_stop_pct > 0:
                        max_new_pos = remaining_budget * 100.0 / new_stop_pct
                    else:
                        max_new_pos = 0.0
                    max_new_pos = round(max(0.0, max_new_pos), 2)
                    violation_msg = (
                        f"HEAT CAP: Adding {new_pos_pct:.1f}% position (heat={new_heat:.2f}%) "
                        f"would push total portfolio heat to {total_heat:.2f}%, "
                        f"exceeding budget of {budget:.1f}%. "
                        f"Position clamped to {max_new_pos:.1f}%."
                    )
                    violations.append(violation_msg)
                    heat_capped = f"{max_new_pos:.0f}% of portfolio (clamped by portfolio heat budget)"
                    decision = self._replace_field(decision, "Position Sizing", heat_capped)
                    clamped["Position Sizing (heat)"] = (f"{new_pos_pct:.1f}%", f"{max_new_pos:.1f}%")
                    logger.warning("Portfolio heat guardrail: %s", violation_msg)

        # ── 5. Stop-loss requirement ──
        if self.gc.require_stop_loss:
            stop_loss = self._extract_field(decision, "Stop Loss")
            rating_lower = (rating or "").lower()
            if rating_lower in ("buy", "overweight") and not stop_loss:
                violations.append(
                    f"WARNING: {rating} recommendation issued without a stop-loss. "
                    f"Risk guardrails require a stop-loss for directional positions."
                )
                # Append a warning to the decision rather than blocking
                decision += (
                    "\n\n**⚠️ Risk Guardrail Warning**: No stop-loss specified. "
                    "A stop-loss is strongly recommended before execution."
                )

        # ── 6. Loss-per-trade sanity check ──
        entry = self._extract_number(self._extract_field(decision, "Entry Price") or "")
        stop = self._extract_number(self._extract_field(decision, "Stop Loss") or "")
        if entry and stop and entry > 0:
            loss_pct = abs(entry - stop) / entry * 100
            if loss_pct > self.gc.max_single_loss_pct:
                violations.append(
                    f"ALERT: Stop-loss distance ({loss_pct:.1f}%) exceeds max "
                    f"single-loss limit ({self.gc.max_single_loss_pct:.1f}%). "
                    f"Consider tightening the stop."
                )
                # Don't override the stop — just warn. The trader may have reasons.
                decision += (
                    f"\n\n**⚠️ Risk Guardrail Alert**: Stop-loss distance "
                    f"({loss_pct:.1f}%) exceeds the configured maximum of "
                    f"{self.gc.max_single_loss_pct:.1f}%."
                )

        if violations:
            logger.warning(
                "Risk guardrails triggered %d violation(s):\n  %s",
                len(violations),
                "\n  ".join(violations),
            )

        return GuardrailResult(
            original_decision=final_trade_decision,
            modified_decision=decision,
            was_modified=decision != final_trade_decision,
            violations=violations,
            clamped_fields=clamped,
        )

    # ── Parsing helpers ──

    @staticmethod
    def _extract_field(text: str, field_name: str) -> str | None:
        """Extract the value after **Field Name**: from markdown."""
        pattern = rf"\*\*{re.escape(field_name)}\*\*:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _replace_field(text: str, field_name: str, new_value: str) -> str:
        """Replace a **Field Name**: value in markdown."""
        pattern = rf"(\*\*{re.escape(field_name)}\*\*:\s*).+?(?=\n|$)"
        return re.sub(
            pattern,
            lambda m: m.group(1) + new_value,
            text,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _extract_percentage(text: str) -> float | None:
        """Extract the first percentage number from a string."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        return float(match.group(1)) if match else None

    @staticmethod
    def _extract_number(text: str) -> float | None:
        """Extract the first number from a string."""
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else None


def create_guardrail_node(config: dict):
    """Create a LangGraph node that applies risk guardrails post-PM.

    Usage in setup.py:
        guardrail_node = create_guardrail_node(self.config)
        workflow.add_node("Risk Guardrails", guardrail_node)
        workflow.add_edge("Portfolio Manager", "Risk Guardrails")
        workflow.add_edge("Risk Guardrails", END)
    """
    guardrails = RiskGuardrails(config)

    def guardrail_node(state) -> dict:
        result = guardrails.check(state["final_trade_decision"])

        if result.was_modified:
            logger.info(
                "Risk guardrails modified the decision. Violations: %s",
                "; ".join(result.violations),
            )

        return {"final_trade_decision": result.modified_decision}

    return guardrail_node
