"""Valuation Analyst — ROIC-DCF, Revenue DCF, DDM, and scenario analysis.

Follows the exact same pattern as fundamentals_analyst.py:
  - bind_tools_or_none / safe_tool_text for tool-free provider fallback
  - Reads ticker and trade_date from AgentState
  - Returns {"messages": [...], "valuation_report": str}
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.tool_fallback import bind_tools_or_none, safe_tool_text


# ---------------------------------------------------------------------------
# Helper: fetch and compute all valuation metrics for a ticker
# ---------------------------------------------------------------------------

def _compute_valuation_package(ticker: str) -> dict:
    """Fetch raw inputs and compute all valuation metrics. Returns a flat dict."""
    from tradingagents.dataflows.valuation_data import get_valuation_inputs
    from tradingagents.valuation.roic import (
        invested_capital as calc_ic,
        nopat as calc_nopat,
        roic as calc_roic,
    )
    from tradingagents.valuation.wacc import (
        after_tax_cost_of_debt,
        cost_of_equity,
        value_spread,
        wacc as calc_wacc,
    )
    from tradingagents.valuation.dcf import margin_of_safety, revenue_dcf, roic_dcf
    from tradingagents.valuation.ddm import (
        gordon_growth_ddm,
        is_dividend_payer,
        multi_stage_ddm,
    )
    from tradingagents.valuation.scenarios import (
        default_scenario_set,
        run_revenue_scenarios,
        run_roic_scenarios,
    )

    d = get_valuation_inputs(ticker)

    # ROIC
    nopat_val = calc_nopat(d["ebit"], d["tax_rate"])
    # Excess cash heuristic: cash beyond 2% of revenue
    excess_cash = max(d["cash_and_equivalents"] - 0.02 * d["revenue"], 0.0)
    ic_val = calc_ic(d["total_assets"], excess_cash, d["non_interest_current_liabilities"])
    roic_val = calc_roic(nopat_val, ic_val)

    # WACC
    ke = cost_of_equity(d["risk_free_rate"], d["beta"], d["equity_risk_premium"])
    kd = after_tax_cost_of_debt(d["interest_expense"], d["total_debt"], d["tax_rate"])
    wacc_val = calc_wacc(d["market_cap"], d["total_debt"], ke, kd)
    spread = value_spread(roic_val, wacc_val)

    # ROIC-DCF (base: reinvestment_rate derived from roic/growth assumption of 5%)
    reinvestment_rate = min(0.05 / max(roic_val, 0.01), 0.8)
    roic_iv = roic_dcf(
        nopat=nopat_val,
        roic_val=roic_val,
        wacc_val=wacc_val,
        reinvestment_rate=reinvestment_rate,
        projection_years=7,
        terminal_growth=0.025,
        shares_outstanding=d["shares_outstanding"],
    )

    # Revenue DCF (base: 5% revenue growth, current ebit margin, 7-yr horizon)
    ebit_margin = d["ebit"] / d["revenue"] if d["revenue"] > 0 else 0.10
    revenue_iv = revenue_dcf(
        revenue=d["revenue"],
        growth_rates=[0.05] * 7,
        ebit_margin=ebit_margin,
        tax_rate=d["tax_rate"],
        wacc_val=wacc_val,
        terminal_growth=0.025,
        shares_outstanding=d["shares_outstanding"],
        net_debt=d["net_debt"],
    )

    # DDM
    dividend_iv = None
    if is_dividend_payer(d["dividend_history"]):
        dividend_iv = gordon_growth_ddm(
            dps=d["dividends_per_share"],
            growth_rate=0.04,
            cost_of_equity=ke,
        )

    # Scenarios (ROIC-DCF)
    scenario_set = default_scenario_set(
        base_growth=0.05,
        base_terminal=0.025,
        base_margin=ebit_margin,
        base_reinvestment=reinvestment_rate,
    )
    roic_scenarios = run_roic_scenarios(
        nopat=nopat_val,
        roic_val=roic_val,
        wacc_val=wacc_val,
        shares_outstanding=d["shares_outstanding"],
        current_price=d["current_price"],
        scenario_set=scenario_set,
    )
    revenue_scenarios = run_revenue_scenarios(
        revenue=d["revenue"],
        shares_outstanding=d["shares_outstanding"],
        net_debt=d["net_debt"],
        tax_rate=d["tax_rate"],
        wacc_val=wacc_val,
        current_price=d["current_price"],
        scenario_set=scenario_set,
    )

    roic_mos = margin_of_safety(roic_iv, d["current_price"])
    rev_mos = margin_of_safety(revenue_iv, d["current_price"])

    return {
        **d,
        "nopat": nopat_val,
        "invested_capital": ic_val,
        "roic": roic_val,
        "ke": ke,
        "kd": kd,
        "wacc": wacc_val,
        "spread": spread,
        "reinvestment_rate": reinvestment_rate,
        "ebit_margin": ebit_margin,
        "roic_iv": roic_iv,
        "revenue_iv": revenue_iv,
        "dividend_iv": dividend_iv,
        "roic_mos": roic_mos,
        "rev_mos": rev_mos,
        "roic_scenarios": roic_scenarios,
        "revenue_scenarios": revenue_scenarios,
    }


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def get_wacc_components(ticker: str) -> str:
    """Return a detailed WACC breakdown for the given ticker.

    Includes risk-free rate, beta, equity risk premium, cost of equity (CAPM),
    cost of debt, capital structure weights, and the resulting WACC.
    """
    try:
        p = _compute_valuation_package(ticker)
    except Exception as exc:  # noqa: BLE001
        return f"WACC computation failed for {ticker}: {exc}"

    return (
        f"=== WACC Components for {ticker} ===\n"
        f"Risk-Free Rate (Rf):          {p['risk_free_rate']:.2%}\n"
        f"Beta:                         {p['beta']:.2f}\n"
        f"Equity Risk Premium (ERP):    {p['equity_risk_premium']:.2%}\n"
        f"Cost of Equity (Ke = CAPM):   {p['ke']:.2%}\n"
        f"Pre-tax Cost of Debt:         "
        f"{(p['interest_expense']/p['total_debt']):.2%}" if p['total_debt'] > 0
        else "Pre-tax Cost of Debt:         N/A (no debt)\n"
        f"\nAfter-tax Cost of Debt (Kd):  {p['kd']:.2%}\n"
        f"Market Cap (E):               ${p['market_cap']:,.0f}\n"
        f"Total Debt (D):               ${p['total_debt']:,.0f}\n"
        f"Equity Weight (E/V):          {p['market_cap']/(p['market_cap']+p['total_debt']):.1%}\n"
        f"Debt Weight (D/V):            {p['total_debt']/(p['market_cap']+p['total_debt']):.1%}\n"
        f"WACC:                         {p['wacc']:.2%}\n"
    )


@tool
def get_roic_analysis(ticker: str) -> str:
    """Compute ROIC, NOPAT, Invested Capital, and the ROIC vs WACC value spread.

    A positive spread (ROIC > WACC) means the company creates economic value.
    A negative spread signals value destruction.
    """
    try:
        p = _compute_valuation_package(ticker)
    except Exception as exc:  # noqa: BLE001
        return f"ROIC analysis failed for {ticker}: {exc}"

    verdict = "VALUE CREATING" if p["spread"] > 0 else "VALUE DESTROYING"
    strength = (
        "strong" if abs(p["spread"]) > 0.05
        else "moderate" if abs(p["spread"]) > 0.02
        else "marginal"
    )

    return (
        f"=== ROIC Analysis for {ticker} ===\n"
        f"EBIT:                         ${p['ebit']:,.0f}\n"
        f"Effective Tax Rate:           {p['tax_rate']:.1%}\n"
        f"NOPAT:                        ${p['nopat']:,.0f}\n"
        f"Total Assets:                 ${p['total_assets']:,.0f}\n"
        f"Excess Cash:                  ${max(p['cash_and_equivalents']-0.02*p['revenue'],0):,.0f}\n"
        f"Non-interest Current Liabs:   ${p['non_interest_current_liabilities']:,.0f}\n"
        f"Invested Capital:             ${p['invested_capital']:,.0f}\n"
        f"\nROIC:                         {p['roic']:.2%}\n"
        f"WACC:                         {p['wacc']:.2%}\n"
        f"Value Spread (ROIC − WACC):   {p['spread']:+.2%}\n"
        f"Verdict: {verdict} ({strength} spread)\n"
    )


@tool
def get_dcf_valuation(ticker: str) -> str:
    """Run both ROIC-driven DCF and Revenue DCF; return side-by-side intrinsic values.

    Assumptions use base-case inputs (5% revenue growth, 7-year horizon, 2.5% terminal growth).
    """
    try:
        p = _compute_valuation_package(ticker)
    except Exception as exc:  # noqa: BLE001
        return f"DCF valuation failed for {ticker}: {exc}"

    return (
        f"=== DCF Valuation for {ticker} ===\n"
        f"Current Price:                ${p['current_price']:.2f}\n\n"
        f"--- ROIC-Driven DCF ---\n"
        f"NOPAT:                        ${p['nopat']:,.0f}\n"
        f"ROIC:                         {p['roic']:.2%}\n"
        f"Reinvestment Rate:            {p['reinvestment_rate']:.1%}\n"
        f"Implied Growth (ROIC×Reinv):  {p['roic']*p['reinvestment_rate']:.2%}\n"
        f"WACC:                         {p['wacc']:.2%}\n"
        f"Terminal Growth:              2.50%\n"
        f"Projection Years:             7\n"
        f"Intrinsic Value (ROIC-DCF):   ${p['roic_iv']:.2f}\n"
        f"Margin of Safety:             {p['roic_mos']:+.1f}%\n\n"
        f"--- Revenue DCF ---\n"
        f"Revenue:                      ${p['revenue']:,.0f}\n"
        f"EBIT Margin:                  {p['ebit_margin']:.1%}\n"
        f"Revenue Growth Assumption:    5.00% p.a.\n"
        f"WACC:                         {p['wacc']:.2%}\n"
        f"Terminal Growth:              2.50%\n"
        f"Projection Years:             7\n"
        f"Net Debt:                     ${p['net_debt']:,.0f}\n"
        f"Intrinsic Value (Rev-DCF):    ${p['revenue_iv']:.2f}\n"
        f"Margin of Safety:             {p['rev_mos']:+.1f}%\n"
    )


@tool
def get_ddm_valuation(ticker: str) -> str:
    """Run the Dividend Discount Model if the company pays dividends.

    Uses Gordon Growth Model with a 4% perpetual dividend growth rate.
    Returns a clear message if the company does not pay dividends.
    """
    try:
        p = _compute_valuation_package(ticker)
    except Exception as exc:  # noqa: BLE001
        return f"DDM valuation failed for {ticker}: {exc}"

    if p["dividend_iv"] is None:
        return (
            f"=== DDM Valuation for {ticker} ===\n"
            f"Company does not pay dividends (or dividend history is insufficient). "
            f"DDM is not applicable. Use ROIC-DCF or Revenue DCF instead.\n"
        )

    from tradingagents.valuation.dcf import margin_of_safety
    mos = margin_of_safety(p["dividend_iv"], p["current_price"])

    return (
        f"=== DDM Valuation for {ticker} ===\n"
        f"Current Price:                ${p['current_price']:.2f}\n"
        f"Dividends Per Share:          ${p['dividends_per_share']:.4f}\n"
        f"Cost of Equity (Ke):          {p['ke']:.2%}\n"
        f"Assumed Dividend Growth:      4.00%\n"
        f"Intrinsic Value (Gordon DDM): ${p['dividend_iv']:.2f}\n"
        f"Margin of Safety:             {mos:+.1f}%\n"
    )


@tool
def get_scenario_analysis(ticker: str) -> str:
    """Run bear / base / bull scenarios for both ROIC-DCF and Revenue DCF.

    Bear: 60% of base growth, margin -20%, terminal -1pp.
    Base: 5% growth, current margin, 2.5% terminal.
    Bull: 140% of base growth, margin +20%, terminal +0.5pp.
    """
    try:
        p = _compute_valuation_package(ticker)
    except Exception as exc:  # noqa: BLE001
        return f"Scenario analysis failed for {ticker}: {exc}"

    rs = p["roic_scenarios"]
    vs = p["revenue_scenarios"]

    lines = [
        f"=== Scenario Analysis for {ticker} ===",
        f"Current Price: ${p['current_price']:.2f}",
        "",
        f"{'Scenario':<10} {'ROIC-DCF IV':>13} {'ROIC Upside':>13} {'Rev-DCF IV':>13} {'Rev Upside':>13}",
        "-" * 65,
    ]
    for label in ("bear", "base", "bull"):
        r = rs[label]
        v = vs[label]
        lines.append(
            f"{label.capitalize():<10} "
            f"${r.intrinsic_value:>11.2f} "
            f"{r.upside_pct:>+12.1f}% "
            f"${v.intrinsic_value:>11.2f} "
            f"{v.upside_pct:>+12.1f}%"
        )

    lines += [
        "",
        "Key Sensitivities:",
        f"  ROIC-DCF is most sensitive to: ROIC ({p['roic']:.2%}) and reinvestment rate ({p['reinvestment_rate']:.1%})",
        f"  Revenue DCF is most sensitive to: revenue growth rate and EBIT margin ({p['ebit_margin']:.1%})",
        f"  Both models share WACC sensitivity: current WACC = {p['wacc']:.2%}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_valuation_analyst(llm):
    """Create the Valuation Analyst node function for the LangGraph workflow."""

    tools = [
        get_wacc_components,
        get_roic_analysis,
        get_dcf_valuation,
        get_ddm_valuation,
        get_scenario_analysis,
    ]

    def _prefetch_valuation_data(ticker: str) -> str:
        """Pre-fetch all valuation outputs for tool-free providers."""
        wacc_text = safe_tool_text(
            "WACC components", lambda: get_wacc_components.func(ticker)
        )
        roic_text = safe_tool_text(
            "ROIC analysis", lambda: get_roic_analysis.func(ticker)
        )
        dcf_text = safe_tool_text(
            "DCF valuation", lambda: get_dcf_valuation.func(ticker)
        )
        ddm_text = safe_tool_text(
            "DDM valuation", lambda: get_ddm_valuation.func(ticker)
        )
        scenario_text = safe_tool_text(
            "scenario analysis", lambda: get_scenario_analysis.func(ticker)
        )
        return "\n\n".join([wacc_text, roic_text, dcf_text, ddm_text, scenario_text])

    def valuation_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        subject_label = "company" if asset_type == "stock" else "asset"
        ticker = str(state["company_of_interest"])
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            f"You are a Valuation Analyst specialising in intrinsic value estimation. "
            f"Your role is to determine whether this {subject_label} trades at a discount "
            f"or premium to its intrinsic value using multiple complementary methods.\n\n"
            f"Follow this sequence:\n"
            f"1. Call `get_wacc_components` and `get_roic_analysis` to establish the "
            f"   value-creation verdict (ROIC vs WACC spread).\n"
            f"2. Call `get_dcf_valuation` for quantitative intrinsic value estimates "
            f"   from two independent DCF lenses.\n"
            f"3. Call `get_ddm_valuation` — include the result whether or not the "
            f"   company pays dividends.\n"
            f"4. Call `get_scenario_analysis` for the full bear/base/bull range.\n"
            f"5. Synthesise into a structured memo covering:\n"
            f"   a. Value-creation verdict (ROIC vs WACC): is the company earning above "
            f"      its cost of capital?\n"
            f"   b. Intrinsic value triangulation: report all methods, note convergence "
            f"      or divergence.\n"
            f"   c. Margin of safety at the current price: state explicitly whether the "
            f"      stock offers a margin of safety (>15% discount) or is richly valued.\n"
            f"   d. Scenario range: summarise bear IV, base IV, bull IV and what drives "
            f"      each outcome.\n"
            f"   e. Single most sensitive assumption per method.\n"
            f"   f. PASS / WARN / FAIL verdict on the valuation case.\n\n"
            f"Append a Markdown summary table at the end."
            + get_language_instruction()
        )

        bound_llm = bind_tools_or_none(llm, tools, "Valuation Analyst")

        if bound_llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful AI assistant, collaborating with other assistants."
                        " Use the provided tools to progress towards answering the question."
                        " If you are unable to fully answer, that's OK; another assistant with"
                        " different tools will help where you left off. Execute what you can to"
                        " make progress."
                        " If you or any other assistant has the FINAL TRANSACTION PROPOSAL:"
                        " **BUY/HOLD/SELL** or deliverable, prefix your response with"
                        " FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                        " You have access to the following tools: {tool_names}.\n{system_message}"
                        " For your reference, the current date is {current_date}."
                        " {instrument_context}",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            prompt = prompt.partial(system_message=system_message)
            prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
            prompt = prompt.partial(current_date=current_date)
            prompt = prompt.partial(instrument_context=instrument_context)

            chain = prompt | bound_llm
            result = chain.invoke(state["messages"])

            report = ""
            if len(result.tool_calls) == 0:
                report = result.content

            return {
                "messages": [result],
                "valuation_report": report,
            }

        # Tool-free fallback: pre-fetch all data and inject into prompt
        valuation_data = _prefetch_valuation_data(ticker)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " The valuation data you need has ALREADY been gathered for you and is"
                    " included below; do NOT call any tools and disregard any instruction"
                    " below to call a tool — base your report only on the provided data."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL:"
                    " **BUY/HOLD/SELL** or deliverable, prefix your response with"
                    " FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}."
                    " {instrument_context}\n\n"
                    "=== Pre-fetched valuation data ===\n{valuation_data}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(valuation_data=valuation_data)

        formatted_messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(formatted_messages)

        return {
            "messages": [result],
            "valuation_report": result.content,
        }

    return valuation_analyst_node
