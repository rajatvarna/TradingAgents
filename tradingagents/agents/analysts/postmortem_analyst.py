"""
Post-Mortem Analyst.

Runs on past trading decisions (4-12 weeks old), computes outcomes,
and generates one lesson per trade for injection into the Portfolio Manager.

Implements Boik's principle: "Study your past trades honestly."
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.audit.prompt_registry import default_registry


def create_postmortem_analyst(llm, prompt_registry=None):
    """Create the Post-Mortem Analyst node for weekly review runs."""
    registry = prompt_registry or default_registry()

    def postmortem_analyst_node(state):
        past_rec = state.get("postmortem_past_recommendation", "No past recommendation provided.")
        outcome = state.get("postmortem_outcome_data", "No outcome data provided.")

        version = state.get("prompt_versions", {}).get("analysts/postmortem", "v1")
        system_message, prompt_hash = registry.render(
            "analysts/postmortem",
            version=version,
            past_recommendation=past_rec,
            outcome_data=outcome,
            language_instruction=get_language_instruction(),
        )

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Generate the post-mortem analysis and extract the lesson."),
        ])
        chain = prompt | llm
        result = chain.invoke(
            {},
            config={
                "metadata": {
                    "prompt_key": "analysts/postmortem",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                }
            },
        )
        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [HumanMessage(content=f"Post-Mortem Analyst:\n{report}")],
            "postmortem_report": report,
        }

    return postmortem_analyst_node
