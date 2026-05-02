# Shared constants for the agent graph.

# Maps analyst type name → AgentState report field name.
# Must be updated whenever a new analyst is added.
ANALYST_REPORT_KEYS: dict[str, str] = {
    "market": "market_report",
    "sentiment": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
    "options": "options_report",
}
