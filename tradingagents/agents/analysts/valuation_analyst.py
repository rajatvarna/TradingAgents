"""
Valuation Analyst agent.

Performs ROIC-driven DCF, Revenue DCF, DDM, and bear/base/bull scenario analysis
for a given ticker.  Follows the exact same pattern as fundamentals_analyst.py.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.tool_fallback import bind_tools_or_none, safe_tool_text

# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────────────────────────────────────


def _make_tools():
    """Build the valuation analyst tool list (lazy imports)."""
    from langchain_core.tools import tool

    from tradingagents.dataflows.valuation_data import get_valuation_inputs
    from tradingagents.valuation.dcf import margin_of_safety, revenue_dcf, roic_dcf
    from tradingagents.valuation.ddm import (
        gordon_growth_ddm,
        is_dividend_payer,
        multi_stage_ddm,
    )
    from tradingagents.valuation.roic import (
        invested_capital as calc_invested_capital,
    )
    from tradingagents.valuation.roic import (
        nopat as calc_nopat,
    )
    from tradingagents.valuation.roic import (
        roic as calc_roic,
    )
    from tradingagents.valuation.scenarios import (
        default_scenario_set,
        run_revenue_scenarios,
        run_roic_scenarios,
    )
    from tradingagents.valuation.wacc import (
        after_tax_cost_of_debt,
        cost_of_equity,
        value_spread,
    )
    from tradingagents.valuation.wacc import (
        wacc as calc_wacc,
    )

    @tool
    def get_wacc_components(ticker: str) -> str:
        """Return a detailed WACC breakdown for the given ticker.

        Includes risk-free rate, beta, equity risk premium, cost of equity (CAPM),
        cost of debt, and capital structure weights.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Formatted string with WACC components.
        """
        try:
            inputs = get_valuation_inputs(ticker)
        except Exception as exc:
            return f"Error fetching valuation inputs for {ticker}: {exc}"

        rf = inputs["risk_free_rate"]
        beta = inputs["beta"]
        erp = inputs["equity_risk_premium"]
        tax_rate = inputs["tax_rate"]
        total_debt = inputs.get("total_debt") or 0.0
        interest_expense = inputs.get("interest_expense") or 0.0
        current_price = inputs.get("current_price") or 0.0
        shares = inputs.get("shares_outstanding") or 0.0

        ke = cost_of_equity(rf, beta, erp)
        kd = after_tax_cost_of_debt(interest_expense, total_debt, tax_rate)

        equity_value = current_price * shares
        total_capital = equity_value + total_debt
        weight_e = equity_value / total_capital if total_capital > 0 else 1.0
        weight_d = total_debt / total_capital if total_capital > 0 else 0.0

        try:
            wacc_val = calc_wacc(equity_value, total_debt, ke, kd)
        except ValueError:
            wacc_val = ke  # Fallback to cost of equity if no debt

        lines = [
            f"=== WACC Components for {ticker.upper()} ===",
            f"Risk-Free Rate (Rf):          {rf*100:.2f}%",
            f"Beta:                         {beta:.2f}",
            f"Equity Risk Premium (ERP):    {erp*100:.2f}%",
            f"Cost of Equity (Ke = CAPM):   {ke*100:.2f}%",
            f"Pre-Tax Cost of Debt:         {(abs(interest_expense)/total_debt*100 if total_debt > 0 else 0):.2f}%",
            f"Tax Rate:                     {tax_rate*100:.1f}%",
            f"After-Tax Cost of Debt (Kd):  {kd*100:.2f}%",
            f"Weight of Equity:             {weight_e*100:.1f}%",
            f"Weight of Debt:               {weight_d*100:.1f}%",
            f"WACC:                         {wacc_val*100:.2f}%",
            "=== END WACC ===",
        ]
        return "\n".join(lines)

    @tool
    def get_roic_analysis(ticker: str) -> str:
        """Compute NOPAT, Invested Capital, ROIC, WACC, and value spread for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Formatted string with ROIC analysis including value-creation verdict.
        """
        try:
            inputs = get_valuation_inputs(ticker)
        except Exception as exc:
            return f"Error fetching valuation inputs for {ticker}: {exc}"

        ebit = inputs.get("ebit")
        if ebit is None:
            return f"EBIT data unavailable for {ticker} — cannot compute ROIC."

        tax_rate = inputs["tax_rate"]
        total_assets = inputs.get("total_assets") or 0.0
        cash = inputs.get("cash_and_equivalents") or 0.0
        nicl = inputs.get("non_interest_current_liabilities") or 0.0
        total_debt = inputs.get("total_debt") or 0.0
        interest_expense = inputs.get("interest_expense") or 0.0
        current_price = inputs.get("current_price") or 0.0
        shares = inputs.get("shares_outstanding") or 0.0
        rf = inputs["risk_free_rate"]
        beta = inputs["beta"]
        erp = inputs["equity_risk_premium"]

        nopat_val = calc_nopat(ebit, tax_rate)
        ic = calc_invested_capital(total_assets, cash, nicl)

        if ic == 0:
            return f"Invested Capital is zero for {ticker} — cannot compute ROIC."

        roic_val = calc_roic(nopat_val, ic)

        ke = cost_of_equity(rf, beta, erp)
        kd = after_tax_cost_of_debt(interest_expense, total_debt, tax_rate)
        equity_value = current_price * shares

        try:
            wacc_val = calc_wacc(equity_value, total_debt, ke, kd)
        except ValueError:
            wacc_val = ke

        spread = value_spread(roic_val, wacc_val)
        verdict = "VALUE CREATING" if spread > 0 else "VALUE DESTROYING"

        lines = [
            f"=== ROIC Analysis for {ticker.upper()} ===",
            f"EBIT:                  ${ebit/1e9:.2f}B",
            f"Tax Rate:              {tax_rate*100:.1f}%",
            f"NOPAT:                 ${nopat_val/1e9:.2f}B",
            f"Total Assets:          ${total_assets/1e9:.2f}B",
            f"Excess Cash:           ${cash/1e9:.2f}B",
            f"Non-Interest CL:       ${nicl/1e9:.2f}B",
            f"Invested Capital:      ${ic/1e9:.2f}B",
            f"ROIC:                  {roic_val*100:.2f}%",
            f"WACC:                  {wacc_val*100:.2f}%",
            f"Value Spread (ROIC-WACC): {spread*100:.2f}%  →  {verdict}",
            "=== END ROIC ANALYSIS ===",
        ]
        return "\n".join(lines)

    @tool
    def get_dcf_valuation(ticker: str) -> str:
        """Run ROIC-DCF and Revenue-DCF side by side for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Formatted string with both DCF intrinsic value estimates and margin of safety.
        """
        try:
            inputs = get_valuation_inputs(ticker)
        except Exception as exc:
            return f"Error fetching valuation inputs for {ticker}: {exc}"

        ebit = inputs.get("ebit")
        revenue = inputs.get("revenue")
        shares = inputs.get("shares_outstanding")
        current_price = inputs.get("current_price") or 0.0
        total_assets = inputs.get("total_assets") or 0.0
        cash = inputs.get("cash_and_equivalents") or 0.0
        nicl = inputs.get("non_interest_current_liabilities") or 0.0
        total_debt = inputs.get("total_debt") or 0.0
        interest_expense = inputs.get("interest_expense") or 0.0
        net_debt = inputs.get("net_debt") or 0.0
        tax_rate = inputs["tax_rate"]
        rf = inputs["risk_free_rate"]
        beta = inputs["beta"]
        erp = inputs["equity_risk_premium"]

        ke = cost_of_equity(rf, beta, erp)
        kd = after_tax_cost_of_debt(interest_expense, total_debt, tax_rate)
        equity_value = current_price * (shares or 0.0)

        try:
            wacc_val = calc_wacc(equity_value, total_debt, ke, kd)
        except ValueError:
            wacc_val = ke

        results = []

        # ROIC-DCF
        if ebit is not None and shares and shares > 0:
            nopat_val = calc_nopat(ebit, tax_rate)
            ic = calc_invested_capital(total_assets, cash, nicl)
            if ic != 0:
                roic_val = calc_roic(nopat_val, ic)
                reinvestment_rate = 0.30  # Base assumption
                try:
                    iv_roic = roic_dcf(
                        nopat=nopat_val,
                        roic_val=roic_val,
                        wacc_val=wacc_val,
                        reinvestment_rate=reinvestment_rate,
                        projection_years=10,
                        terminal_growth=0.025,
                        shares_outstanding=shares,
                    )
                    mos_roic = margin_of_safety(iv_roic, current_price) if iv_roic != 0 else 0.0
                    results.append(f"ROIC-DCF Intrinsic Value:    ${iv_roic:.2f}/share")
                    results.append(f"ROIC-DCF Margin of Safety:   {mos_roic:.1f}%")
                except ValueError as exc:
                    results.append(f"ROIC-DCF: {exc}")
            else:
                results.append("ROIC-DCF: Invested Capital is zero — skipped.")
        else:
            results.append("ROIC-DCF: EBIT or shares outstanding unavailable — skipped.")

        # Revenue-DCF
        if revenue is not None and shares and shares > 0:
            base_growth = 0.08  # 8% base assumption
            ebit_margin = ebit / revenue if ebit is not None and revenue != 0 else 0.10
            growth_rates = [base_growth] * 10
            try:
                iv_rev = revenue_dcf(
                    revenue=revenue,
                    growth_rates=growth_rates,
                    ebit_margin=ebit_margin,
                    tax_rate=tax_rate,
                    wacc_val=wacc_val,
                    terminal_growth=0.025,
                    shares_outstanding=shares,
                    net_debt=net_debt,
                )
                mos_rev = margin_of_safety(iv_rev, current_price) if iv_rev != 0 else 0.0
                results.append(f"Revenue-DCF Intrinsic Value: ${iv_rev:.2f}/share")
                results.append(f"Revenue-DCF Margin of Safety:{mos_rev:.1f}%")
            except ValueError as exc:
                results.append(f"Revenue-DCF: {exc}")
        else:
            results.append("Revenue-DCF: Revenue or shares outstanding unavailable — skipped.")

        lines = [
            f"=== DCF Valuation for {ticker.upper()} ===",
            f"Current Price:               ${current_price:.2f}",
            f"WACC:                        {wacc_val*100:.2f}%",
        ] + results + ["=== END DCF VALUATION ==="]
        return "\n".join(lines)

    @tool
    def get_ddm_valuation(ticker: str) -> str:
        """Run DDM valuation if the company pays dividends, otherwise explain why it is not applicable.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Formatted DDM intrinsic value string, or a message that DDM is not applicable.
        """
        try:
            inputs = get_valuation_inputs(ticker)
        except Exception as exc:
            return f"Error fetching valuation inputs for {ticker}: {exc}"

        dps = inputs.get("dividends_per_share") or 0.0
        div_history = inputs.get("dividend_history") or []

        if not is_dividend_payer(div_history if div_history else ([dps] if dps > 0 else [])):
            return f"Company does not pay dividends — DDM is not applicable for {ticker.upper()}."

        rf = inputs["risk_free_rate"]
        beta = inputs["beta"]
        erp = inputs["equity_risk_premium"]
        ke = cost_of_equity(rf, beta, erp)
        current_price = inputs.get("current_price") or 0.0

        growth_rate = 0.05  # Conservative 5% perpetual growth assumption

        lines = [f"=== DDM Valuation for {ticker.upper()} ==="]
        lines.append(f"Current DPS:               ${dps:.2f}")
        lines.append(f"Cost of Equity (Ke):       {ke*100:.2f}%")
        lines.append(f"Assumed Perpetual Growth:  {growth_rate*100:.1f}%")

        # Gordon Growth DDM
        try:
            iv_gordon = gordon_growth_ddm(dps, growth_rate, ke)
            mos = margin_of_safety(iv_gordon, current_price) if iv_gordon != 0 else 0.0
            lines.append(f"Gordon Growth DDM IV:      ${iv_gordon:.2f}/share")
            lines.append(f"Margin of Safety:          {mos:.1f}%")
        except ValueError as exc:
            lines.append(f"Gordon Growth DDM: {exc}")

        # Multi-stage DDM if we have history
        if len(div_history) >= 2:
            try:
                # Project 5 years at near-term growth then terminal
                near_term_growth = 0.07
                projected = [div_history[0] * ((1 + near_term_growth) ** y) for y in range(1, 6)]
                iv_multi = multi_stage_ddm(projected, growth_rate, ke)
                mos_multi = margin_of_safety(iv_multi, current_price) if iv_multi != 0 else 0.0
                lines.append(f"Multi-Stage DDM IV:        ${iv_multi:.2f}/share")
                lines.append(f"Multi-Stage MoS:           {mos_multi:.1f}%")
            except (ValueError, IndexError) as exc:
                lines.append(f"Multi-Stage DDM: {exc}")

        lines.append("=== END DDM VALUATION ===")
        return "\n".join(lines)

    @tool
    def get_scenario_analysis(ticker: str) -> str:
        """Run bear / base / bull scenario analysis using both ROIC-DCF and Revenue-DCF.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Formatted scenario table string.
        """
        try:
            inputs = get_valuation_inputs(ticker)
        except Exception as exc:
            return f"Error fetching valuation inputs for {ticker}: {exc}"

        ebit = inputs.get("ebit")
        revenue = inputs.get("revenue")
        shares = inputs.get("shares_outstanding")
        current_price = inputs.get("current_price") or 0.0
        total_assets = inputs.get("total_assets") or 0.0
        cash = inputs.get("cash_and_equivalents") or 0.0
        nicl = inputs.get("non_interest_current_liabilities") or 0.0
        total_debt = inputs.get("total_debt") or 0.0
        interest_expense = inputs.get("interest_expense") or 0.0
        net_debt = inputs.get("net_debt") or 0.0
        tax_rate = inputs["tax_rate"]
        rf = inputs["risk_free_rate"]
        beta = inputs["beta"]
        erp = inputs["equity_risk_premium"]

        ke = cost_of_equity(rf, beta, erp)
        kd = after_tax_cost_of_debt(interest_expense, total_debt, tax_rate)
        equity_value = current_price * (shares or 0.0)

        try:
            wacc_val = calc_wacc(equity_value, total_debt, ke, kd)
        except ValueError:
            wacc_val = ke

        ebit_margin = (ebit / revenue if ebit is not None and revenue and revenue != 0 else 0.10)

        scenario_set = default_scenario_set(
            base_growth=0.08,
            base_terminal=0.025,
            base_margin=ebit_margin,
            base_reinvestment=0.30,
        )

        lines = [
            f"=== Scenario Analysis for {ticker.upper()} ===",
            f"Current Price: ${current_price:.2f}  |  WACC: {wacc_val*100:.2f}%",
            "",
        ]

        # ROIC-DCF scenarios
        if ebit is not None and shares and shares > 0:
            nopat_val = calc_nopat(ebit, tax_rate)
            ic = calc_invested_capital(total_assets, cash, nicl)
            if ic != 0:
                roic_val = calc_roic(nopat_val, ic)
                try:
                    roic_results = run_roic_scenarios(
                        nopat=nopat_val,
                        roic_val=roic_val,
                        wacc_val=wacc_val,
                        shares_outstanding=shares,
                        scenario_set=scenario_set,
                        current_price=current_price,
                    )
                    lines.append("ROIC-DCF Scenarios:")
                    lines.append(f"  {'Scenario':<10} {'IV/Share':>10} {'Upside %':>10}")
                    lines.append(f"  {'-'*32}")
                    for label in ("bear", "base", "bull"):
                        r = roic_results[label]
                        lines.append(f"  {label.capitalize():<10} ${r.intrinsic_value:>9.2f} {r.upside_pct:>9.1f}%")
                    lines.append("")
                except (ValueError, Exception) as exc:
                    lines.append(f"ROIC-DCF Scenarios: {exc}")
            else:
                lines.append("ROIC-DCF Scenarios: Invested Capital is zero — skipped.")

        # Revenue-DCF scenarios
        if revenue is not None and shares and shares > 0:
            try:
                rev_results = run_revenue_scenarios(
                    revenue=revenue,
                    shares_outstanding=shares,
                    net_debt=net_debt,
                    wacc_val=wacc_val,
                    scenario_set=scenario_set,
                    tax_rate=tax_rate,
                    current_price=current_price,
                )
                lines.append("Revenue-DCF Scenarios:")
                lines.append(f"  {'Scenario':<10} {'IV/Share':>10} {'Upside %':>10}")
                lines.append(f"  {'-'*32}")
                for label in ("bear", "base", "bull"):
                    r = rev_results[label]
                    lines.append(f"  {label.capitalize():<10} ${r.intrinsic_value:>9.2f} {r.upside_pct:>9.1f}%")
            except (ValueError, Exception) as exc:
                lines.append(f"Revenue-DCF Scenarios: {exc}")

        lines.append("=== END SCENARIO ANALYSIS ===")
        return "\n".join(lines)

    return [
        get_wacc_components,
        get_roic_analysis,
        get_dcf_valuation,
        get_ddm_valuation,
        get_scenario_analysis,
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Pre-fetch helper (tool-free fallback)
# ──────────────────────────────────────────────────────────────────────────────


def _prefetch_valuation_data(ticker: str) -> str:
    """Pre-fetch valuation data for tool-less LLM providers."""
    tools = _make_tools()
    tool_map = {t.name: t for t in tools}

    sections = []
    for tool_name in (
        "get_wacc_components",
        "get_roic_analysis",
        "get_dcf_valuation",
        "get_ddm_valuation",
        "get_scenario_analysis",
    ):
        result = safe_tool_text(
            tool_name,
            lambda t=tool_map[tool_name]: t.func(ticker),
        )
        sections.append(f"### {tool_name}\n{result}")

    return "\n\n".join(sections)


# ──────────────────────────────────────────────────────────────────────────────
# Agent factory
# ──────────────────────────────────────────────────────────────────────────────


def create_valuation_analyst(llm, toolkit=None):
    """Create a Valuation Analyst node function.

    Follows the exact same pattern as create_fundamentals_analyst.

    Args:
        llm: Language model instance.
        toolkit: Ignored; included for API parity with other analyst factories.

    Returns:
        A LangGraph node function (state -> state_update dict).
    """

    def valuation_analyst_node(state):
        current_date = state["trade_date"]
        ticker = str(state["company_of_interest"])
        asset_type = state.get("asset_type", "stock")
        subject_label = "company" if asset_type == "stock" else "asset or protocol"
        instrument_context = get_instrument_context_from_state(state)

        tools = _make_tools()

        system_message = (
            f"You are a Valuation Analyst specializing in intrinsic value estimation "
            f"for {subject_label}s. Your analytical framework is grounded in economic "
            f"value creation: ROIC versus WACC (the value spread). "
            f"Your report must: "
            f"(1) call get_wacc_components and get_roic_analysis first to establish the "
            f"value-creation verdict — is ROIC above WACC (value-creating) or below (value-destroying)? "
            f"(2) call get_dcf_valuation for quantitative intrinsic value estimates from both "
            f"ROIC-DCF and Revenue-DCF perspectives, "
            f"(3) call get_ddm_valuation to assess dividend-based value if applicable, "
            f"(4) call get_scenario_analysis for a bear / base / bull valuation range, "
            f"(5) synthesize into a valuation memo covering: value spread verdict, intrinsic "
            f"value triangulation across methods, margin of safety, scenario range, and the "
            f"key sensitivities that most affect the valuation. "
            f"Conclude with an OVERVALUED / FAIRLY VALUED / UNDERVALUED verdict and the "
            f"primary driver of that verdict. "
            f"Include as much detail as possible. Provide specific, actionable insights "
            f"with supporting evidence."
            + " Make sure to append a Markdown table at the end of the report to organize "
            "key points in the report, organized and easy to read."
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
                        " If you are unable to fully answer, that's OK; another assistant with different tools"
                        " will help where you left off. Execute what you can to make progress."
                        " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                        " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                        " You have access to the following tools: {tool_names}.\n{system_message}"
                        "For your reference, the current date is {current_date}. {instrument_context}",
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

        # Tool-free fallback: pre-fetch all valuation data and inject into prompt
        valuation_data = _prefetch_valuation_data(ticker)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " The valuation data you need has ALREADY been gathered for you and is included below;"
                    " do NOT call any tools and disregard any instruction below to call a tool —"
                    " base your report only on the provided data."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}\n\n"
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
