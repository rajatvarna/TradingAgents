"""ESG analyst agent module."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.esg_data_tools import get_esg_news, get_esg_scores


def create_esg_analyst(llm):
    """Create an ESG analyst node for the trading graph."""

    def esg_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        tools = [get_esg_scores, get_esg_news]

        system_message = (
            "You are a senior ESG (Environmental, Social, Governance) analyst tasked "
            f"with analyzing a {asset_label}'s sustainability profile and ESG risk "
            "factors with the rigor of an institutional ESG research desk — your job is to "
            "identify financially material ESG risks, not to produce a generic CSR summary.\n\n"
            "Your report must cover, in order:\n"
            "1. **Environmental** — carbon footprint / emissions trend, environmental regulatory exposure "
            "(e.g. carbon pricing, emissions standards), physical climate risk (supply chain, facilities), "
            "and any environmental litigation or fines.\n"
            "2. **Social** — labor practices, supply chain/human rights exposure, product safety record, "
            "data privacy and cybersecurity posture, and community/customer relations.\n"
            "3. **Governance** — board independence and composition, executive compensation alignment with "
            "performance, ownership/control structure (dual-class shares, insider ownership), audit quality "
            "history, and related-party transaction risk.\n"
            "4. **Controversies** — any specific, dated controversies (lawsuits, investigations, scandals, "
            "recalls, greenwashing accusations) found in the news, with a materiality assessment for each.\n"
            "5. **Regulatory exposure** — pending or enacted regulation (environmental, labor, data privacy, "
            "antitrust) that could raise costs or restrict operations.\n"
            "6. **Trend direction** — is the ESG risk profile improving, stable, or deteriorating relative to "
            "the recent past, and why.\n"
            "7. **Materiality to valuation** — explicitly connect each major ESG finding to a plausible "
            "financial impact (cost of capital, regulatory fines, reputational/revenue risk, litigation "
            "exposure) rather than treating ESG as a separate, non-financial narrative.\n\n"
            "Use the available tools: "
            "`get_esg_scores` for current ESG ratings where point-in-time safe, and "
            "`get_esg_news` for ESG-related news and controversies up to the analysis "
            "date. If point-in-time ESG scores are unavailable, say so explicitly and "
            "do not infer current scores into historical analysis — reason instead from the news and "
            "controversy data available. Do not fabricate specific scores, ratings, or incidents that are "
            "not supported by tool output. Append a Markdown "
            "table summarizing key ESG signals, risk direction, evidence, materiality, and trading "
            "relevance."
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
        if not result.tool_calls:
            report = result.content

        return {
            "messages": [result],
            "esg_report": report,
        }

    return esg_analyst_node
