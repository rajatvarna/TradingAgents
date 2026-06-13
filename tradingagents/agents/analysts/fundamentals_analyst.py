from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.tool_fallback import bind_tools_or_none, safe_tool_text


def _format_fundamental_monster_context(mss: dict) -> str:
    """Format Monster Stock fundamental + sponsorship scores for prompt injection."""
    if not mss or mss.get("composite_score") is None:
        return ""

    def _cs(key: str) -> str:
        cs = mss.get(key) or {}
        score = cs.get("score")
        score_str = f"{score:.0f}/10" if score is not None else "N/A"
        return f"{score_str} [{cs.get('pass_fail', '?')}] — {cs.get('rationale', '')}"

    blockers = mss.get("hard_blockers") or []
    strengths = mss.get("key_strengths") or []
    risks = mss.get("key_risks") or []

    lines = [
        "=== MONSTER STOCK SCORE — FUNDAMENTAL & SPONSORSHIP VIEW ===",
        f"COMPOSITE: {mss.get('composite_score', 0):.0f}/100  "
        f"Grade: {mss.get('composite_grade', '?')}  "
        f"Action: {mss.get('action_signal', '?').upper()}  "
        f"Stage: {mss.get('stage', '?')}",
    ]
    if blockers:
        lines.append(f"HARD BLOCKERS: {'; '.join(blockers)}")
    lines += [
        "",
        "FUNDAMENTAL SCORES (TraderLion/Boik criteria):",
        f"  EPS Growth (latest Q):       {_cs('eps_growth_score')}",
        f"  EPS Acceleration (8-Q trend):{_cs('eps_acceleration_score')}",
        f"  Revenue Growth:              {_cs('revenue_growth_score')}",
        f"  Revenue Acceleration:        {_cs('revenue_acceleration_score')}",
        f"  Annual EPS Trend (5-yr):     {_cs('annual_eps_trend_score')}",
        f"  ROE (≥17% guideline):        {_cs('roe_score')}",
        f"  After-Tax Margin Trend:      {_cs('margin_trend_score')}",
        f"  Forward Estimate:            {_cs('forward_estimate_score')}",
        "",
        "SPONSORSHIP SCORES:",
        f"  Fund Count Growth (8-Q):     {_cs('fund_count_growth_score')}",
        f"  Fund Count Acceleration:     {_cs('fund_count_acceleration_score')}",
        f"  Flagship Fund Presence:      {_cs('flagship_fund_score')}",
        f"  Institutional Quality:       {_cs('institutional_quality_score')}",
    ]
    if strengths:
        lines.append(f"\nKEY STRENGTHS: {', '.join(strengths)}")
    if risks:
        lines.append(f"KEY RISKS:     {', '.join(risks)}")
    lines += [
        "",
        "METHODOLOGY NOTES:",
        "  - EPS deceleration across 3+ consecutive quarters is a major red flag even if growth is still positive.",
        "  - A revenue-only story (no EPS) is acceptable only when fund count growth is strong and the sector theme is powerful.",
        "  - Use the pre-computed scores above as a structured starting point; your job is to confirm, challenge, or add context.",
        "=== END MONSTER STOCK SCORE ===",
        "",
    ]
    return "\n".join(lines)


def _prefetch_fundamentals_data(ticker: str, current_date: str) -> str:
    """Gather the fundamentals the tools would return, for tool-less providers."""
    fundamentals = safe_tool_text(
        "comprehensive fundamentals",
        lambda: get_fundamentals.func(ticker, current_date),
    )
    balance_sheet = safe_tool_text(
        "balance sheet",
        lambda: get_balance_sheet.func(ticker, curr_date=current_date),
    )
    cashflow = safe_tool_text(
        "cash flow statement",
        lambda: get_cashflow.func(ticker, curr_date=current_date),
    )
    income = safe_tool_text(
        "income statement",
        lambda: get_income_statement.func(ticker, curr_date=current_date),
    )

    return (
        "### Comprehensive fundamentals\n"
        f"{fundamentals}\n\n"
        "### Balance sheet\n"
        f"{balance_sheet}\n\n"
        "### Cash flow statement\n"
        f"{cashflow}\n\n"
        "### Income statement\n"
        f"{income}"
    )


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        subject_label = "company" if asset_type == "stock" else "asset or protocol"
        ticker = str(state["company_of_interest"])
        instrument_context = get_instrument_context_from_state(state)
        monster_context = _format_fundamental_monster_context(state.get("monster_stock_score") or {})

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            monster_context
            + f"You are a Fundamentals Analyst trained on the TraderLion / Boik Monster Stock methodology. "
            f"Analyze fundamental information about this {subject_label} against the scored criteria shown above. "
            f"Your report must: (1) confirm or challenge each scored criterion with additional context, "
            f"(2) identify the PRIMARY fundamental story (EPS story, revenue story, or theme story), "
            f"(3) flag any deceleration in the most recent quarter — this is a critical red flag, "
            f"(4) assess whether analysts expect growth to continue or slow in the next two fiscal years, "
            f"(5) conclude with a PASS / WARN / FAIL verdict on the fundamental case. "
            f"Include as much detail as possible. Provide specific, actionable insights with supporting evidence."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + get_language_instruction()
        )

        bound_llm = bind_tools_or_none(llm, tools, "Fundamentals Analyst")

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
            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
            prompt = prompt.partial(current_date=current_date)
            prompt = prompt.partial(instrument_context=instrument_context)

            chain = prompt | bound_llm

            result = chain.invoke(state["messages"])

            report = ""
            if len(result.tool_calls) == 0:
                report = result.content

            return {
                "messages": [result],
                "fundamentals_report": report,
            }

        # Tool-free fallback: pre-fetch the financial statements and inject them
        # into the prompt for providers (e.g. codex) that cannot bind tools.
        fundamentals_data = _prefetch_fundamentals_data(ticker, current_date)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " The fundamental data you need has ALREADY been gathered for you and is included below;"
                    " do NOT call any tools and disregard any instruction below to call a tool —"
                    " base your report only on the provided data."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}\n\n"
                    "=== Pre-fetched fundamentals ===\n{fundamentals_data}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(fundamentals_data=fundamentals_data)

        formatted_messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(formatted_messages)

        return {
            "messages": [result],
            "fundamentals_report": result.content,
        }

    return fundamentals_analyst_node
