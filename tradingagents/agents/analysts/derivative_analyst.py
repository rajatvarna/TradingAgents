from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.derivatives_tools import (
    get_options_chain,
    get_options_overview,
)


def create_derivative_analyst(llm):

    def derivative_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(
            state["company_of_interest"], asset_type
        )

        tools = [get_options_overview, get_options_chain]

        system_message = (
            "You are a senior derivatives analyst. Analyze the options market for the instrument and "
            "explain what it implies for the underlying with the precision of an institutional derivatives "
            "desk memo — every claim must be tied to a specific number from the tools, not a generality. "
            "Start with get_options_overview to frame expirations, implied volatility, and the put/call "
            "open-interest ratio, then pull get_options_chain for the nearest (and one further) expiry to "
            "inspect skew, liquidity, and notable strikes.\n\n"
            "Cover, in order:\n"
            "(1) **IV level and term structure** — is IV currently elevated, depressed, or normal for this "
            "name, and how does it evolve across expirations (front-loaded IV suggests a near-term event "
            "is priced in);\n"
            "(2) **Skew** (put vs call IV) and what it says about hedging demand vs. speculative positioning — "
            "steep put skew implies fear/downside hedging demand, steep call skew implies speculative upside "
            "or gamma-squeeze risk;\n"
            "(3) **Put/call ratio and open-interest concentrations** — cite the actual ratio and identify the "
            "specific strikes with unusual volume or open interest, and what price levels those concentrations "
            "act as (support/resistance/gamma walls);\n"
            "(4) **Dealer positioning and gamma effects** — explain whether the current open-interest structure "
            "likely dampens (positive gamma near spot) or amplifies (negative gamma) moves in the underlying;\n"
            "(5) **Liquidity assessment** — comment on bid/ask spreads and open-interest depth; thin markets "
            "undermine confidence in the signals above;\n"
            "(6) **One or two concrete derivatives strategies** an investor could consider "
            "(e.g. covered call, protective put, vertical spread, calendar spread around an earnings date) with "
            "the specific directional/volatility thesis each expresses, approximate strikes, and expected payoff "
            "profile; and "
            "(7) **key risks** (assignment, theta decay, IV crush around events, liquidity risk on exit). "
            "Be specific and actionable; do not give generic options education, and do not fabricate strikes, "
            "IV values, or OI figures not present in the tool output — say explicitly when data is unavailable."
            " Make sure to append a Markdown table at the end summarizing key levels, IV, and the "
            "strategies you discuss."
            + get_language_instruction()
        )

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

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "derivatives_report": report,
        }

    return derivative_analyst_node
