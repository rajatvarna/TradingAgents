"""
Valuation engine for TradingAgents.

Provides pure-math modules for ROIC, WACC, DCF, DDM, and scenario analysis.
"""

from tradingagents.valuation.dcf import margin_of_safety, revenue_dcf, roic_dcf
from tradingagents.valuation.ddm import gordon_growth_ddm, is_dividend_payer, multi_stage_ddm
from tradingagents.valuation.roic import (
    invested_capital,
    nopat,
    roic,
    roic_trend,
)
from tradingagents.valuation.scenarios import (
    ScenarioAssumptions,
    ScenarioResult,
    ScenarioSet,
    default_scenario_set,
    run_revenue_scenarios,
    run_roic_scenarios,
)
from tradingagents.valuation.wacc import (
    after_tax_cost_of_debt,
    cost_of_equity,
    value_spread,
    wacc,
)

__all__ = [
    # roic
    "nopat",
    "invested_capital",
    "roic",
    "roic_trend",
    # wacc
    "cost_of_equity",
    "after_tax_cost_of_debt",
    "wacc",
    "value_spread",
    # dcf
    "roic_dcf",
    "revenue_dcf",
    "margin_of_safety",
    # ddm
    "gordon_growth_ddm",
    "multi_stage_ddm",
    "is_dividend_payer",
    # scenarios
    "ScenarioAssumptions",
    "ScenarioSet",
    "ScenarioResult",
    "run_roic_scenarios",
    "run_revenue_scenarios",
    "default_scenario_set",
]
