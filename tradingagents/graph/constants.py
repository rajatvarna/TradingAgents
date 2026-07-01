# Shared constants for the agent graph.

# Maps analyst type name → AgentState report field name.
# Must be updated whenever a new analyst is added.
ANALYST_REPORT_KEYS: dict[str, str] = {
    "market": "market_report",
    "sentiment": "sentiment_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
    "options": "options_report",
    "esg": "esg_report",
    "derivatives": "derivatives_report",
    "valuation": "valuation_report",
}

# Set of valid analyst names for input validation.
VALID_ANALYSTS: frozenset[str] = frozenset(ANALYST_REPORT_KEYS.keys())

# Maps analyst type → the key used to look up its ToolNode in GraphSetup.tool_nodes.
# "sentiment" / "social" analyst uses the "social" tool-node bucket.
TOOL_NODE_KEY: dict[str, str] = {
    "market": "market",
    "sentiment": "social",
    "social": "social",
    "news": "news",
    "fundamentals": "fundamentals",
    "options": "options",
    "esg": "esg",
    "derivatives": "derivatives",
    "valuation": "valuation",
}

# Node-name helpers — single source of truth so renaming only happens here.
def analyst_node_name(analyst_type: str) -> str:
    if analyst_type.lower() == "esg":
        return "ESG Analyst"
    if analyst_type.lower() == "social":
        return "Sentiment Analyst"
    return f"{analyst_type.capitalize()} Analyst"

def clear_node_name(analyst_type: str) -> str:
    if analyst_type.lower() == "esg":
        return "Msg Clear ESG"
    if analyst_type.lower() == "social":
        return "Msg Clear Sentiment"
    return f"Msg Clear {analyst_type.capitalize()}"

def tools_node_name(analyst_type: str) -> str:
    if analyst_type.lower() == "social":
        return "tools_sentiment"
    return f"tools_{analyst_type}"
