"""
Post-Mortem Analyst.

Runs on past trading decisions (4-12 weeks old), computes outcomes,
and generates one lesson per trade for injection into the Portfolio Manager.

Implements Boik's principle: "Study your past trades honestly."
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage


POSTMORTEM_SYSTEM_PROMPT = """You are the Post-Mortem Analyst for the TradingAgents system.
Your role is to evaluate past trading recommendations with the benefit of hindsight,
identify what went right or wrong, and extract actionable lessons.

PAST RECOMMENDATION:
{past_recommendation}

OUTCOME DATA:
{outcome_data}

Answer these five questions concisely:

1. **Was the entry timing correct?** Was the stock in Setup or Breakout stage at entry?
   Was the market in an uptrend? Was the group confirming?

2. **Were sell signals missed?** List any sell signals that fired between entry and peak,
   with approximate dates. Would the Boik framework have triggered an exit?

3. **What was the maximum gain available?** What was the peak price and how many weeks
   after entry? Did the system capture a significant portion?

4. **Was the decline predictable?** If the stock is now lower than entry, identify the
   earliest sell signal that appeared — was it a 50-day MA break, climax run, or
   fundamental deceleration?

5. **What is the ONE lesson?** Write a single actionable lesson in this format:
   "When [setup condition] occurs with [market condition], the correct action is [action]
   because [reason]. Specifically for this stock, [what should have been done differently]."

Output exactly this structure:
- Entry Assessment: [correct/early/late/wrong stage]
- Missed Sell Signals: [list or "none"]
- Max Gain Available: [pct]% at [date]
- Decline Predictability: [early/mid/late warning vs actual exit]
- LESSON: [one paragraph]
"""


def create_postmortem_analyst(llm):
    """Create the Post-Mortem Analyst node for weekly review runs."""

    def postmortem_analyst_node(state):
        past_rec = state.get("postmortem_past_recommendation", "No past recommendation provided.")
        outcome = state.get("postmortem_outcome_data", "No outcome data provided.")

        system_message = POSTMORTEM_SYSTEM_PROMPT.format(
            past_recommendation=past_rec,
            outcome_data=outcome,
        )

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Generate the post-mortem analysis and extract the lesson."),
        ])
        chain = prompt | llm
        result = chain.invoke({})
        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [HumanMessage(content=f"Post-Mortem Analyst:\n{report}")],
            "postmortem_report": report,
        }

    return postmortem_analyst_node
