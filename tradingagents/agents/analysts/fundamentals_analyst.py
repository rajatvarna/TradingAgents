from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_cacheable_system_content,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.tool_fallback import bind_tools_or_none, safe_tool_text
from tradingagents.audit.prompt_registry import default_registry


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


def _format_forensic_context(fs: dict) -> str:
    """Format the forensic accounting (earnings-quality) score for prompt injection."""
    if not fs or fs.get("composite_score") is None:
        return ""

    if not fs.get("data_available", True):
        narrative = fs.get("narrative_summary") or "Forensic accounting data unavailable for this ticker."
        return (
            "=== FORENSIC ACCOUNTING SCORE — EARNINGS QUALITY RED FLAGS ===\n"
            f"DATA UNAVAILABLE: {narrative}\n"
            "Do not treat this as a clean earnings-quality result — no forensic data could be fetched, "
            "so proceed with the financial statements available to you and note the gap in your report.\n"
            "=== END FORENSIC ACCOUNTING SCORE ===\n"
        )

    def _cs(key: str) -> str:
        cs = fs.get(key) or {}
        score = cs.get("score")
        score_str = f"{score:.1f}/10" if score is not None else "N/A"
        return f"{score_str} [{cs.get('pass_fail', '?')}] — {cs.get('rationale', '')}"

    blockers = fs.get("hard_blockers") or []

    lines = [
        "=== FORENSIC ACCOUNTING SCORE — EARNINGS QUALITY RED FLAGS ===",
        f"COMPOSITE: {fs.get('composite_score', 0):.0f}/100  Grade: {fs.get('composite_grade', '?')}",
    ]
    if blockers:
        lines.append(f"HARD BLOCKERS: {'; '.join(blockers)}")
    lines += [
        "",
        f"  Cash Flow / Net Income Divergence: {_cs('cf_ni_divergence_score')}",
        f"  Accruals Quality (Sloan 1996):      {_cs('accruals_quality_score')}",
        f"  Receivables Quality (DSO trend):    {_cs('receivables_quality_score')}",
        f"  SG&A Discipline vs Revenue:         {_cs('sga_discipline_score')}",
        "",
        "METHODOLOGY NOTES:",
        "  - Operating cash flow persistently below net income (OCF/NI < 0.8) means earnings are not converting to cash — a classic earnings-quality warning.",
        "  - Rising days-sales-outstanding alongside revenue growth can indicate channel stuffing or premature revenue recognition.",
        "  - Use the pre-computed scores above as a structured starting point for the forensic red-flag section of your report.",
        "=== END FORENSIC ACCOUNTING SCORE ===",
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


def create_fundamentals_analyst(llm, prompt_registry=None):
    registry = prompt_registry or default_registry()

    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        subject_label = "company" if asset_type == "stock" else "asset or protocol"
        ticker = str(state["company_of_interest"])
        instrument_context = get_instrument_context_from_state(state)
        monster_context = _format_fundamental_monster_context(state.get("monster_stock_score") or {})
        forensic_context = _format_forensic_context(state.get("forensic_score") or {})

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        version = state.get("prompt_versions", {}).get("analysts/fundamentals", "v1")
        rendered_message, prompt_hash = registry.render(
            "analysts/fundamentals",
            version=version,
            monster_context=monster_context,
            forensic_context=forensic_context,
            subject_label=subject_label,
            language_instruction=get_language_instruction(),
        )
        system_message = build_cacheable_system_content(rendered_message, llm)
        prompt_metadata = {
            "prompt_key": "analysts/fundamentals",
            "prompt_version": version,
            "prompt_hash": prompt_hash,
        }

        bound_llm = bind_tools_or_none(llm, tools, "Fundamentals Analyst")

        if bound_llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_message),
                    (
                        "human",
                        "You are a helpful AI assistant, collaborating with other assistants."
                        " Use the provided tools to progress towards answering the question."
                        " If you are unable to fully answer, that's OK; another assistant with different tools"
                        " will help where you left off. Execute what you can to make progress."
                        " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                        " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                        " You have access to the following tools: {tool_names}.\n"
                        "Analysis context:\n"
                        "- Current date: {current_date}\n"
                        "- Instrument context: {instrument_context}",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
            prompt = prompt.partial(current_date=current_date)
            prompt = prompt.partial(instrument_context=instrument_context)

            chain = prompt | bound_llm

            result = chain.invoke(state["messages"], config={"metadata": prompt_metadata})

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
                SystemMessage(content=system_message),
                (
                    "human",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " The fundamental data you need has ALREADY been gathered for you and is included below;"
                    " do NOT call any tools and disregard any instruction below to call a tool —"
                    " base your report only on the provided data."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\nAnalysis context:\n"
                    "- Current date: {current_date}\n"
                    "- Instrument context: {instrument_context}\n\n"
                    "=== Pre-fetched fundamentals ===\n{fundamentals_data}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(fundamentals_data=fundamentals_data)

        formatted_messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(formatted_messages, config={"metadata": prompt_metadata})

        return {
            "messages": [result],
            "fundamentals_report": result.content,
        }

    return fundamentals_analyst_node
