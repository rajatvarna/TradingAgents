"""
Example: Using TradingAgents with OpenClaw LLM Client

This example demonstrates how to use the OpenClaw LLM client
to analyze stocks with TradingAgents.
"""

import os
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Example 1: Using OpenClaw's LLM proxy (local)
def example_local_openclaw():
    """Analyze stock using OpenClaw's local LLM proxy."""
    print("=" * 60)
    print("Example 1: OpenClaw LLM Proxy (Local)")
    print("=" * 60)

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "openclaw"
    config["deep_think_llm"] = "gpt-4"
    config["base_url"] = "http://localhost:8000/v1"  # OpenClaw endpoint

    ta = TradingAgentsGraph(debug=False, config=config)

    # Run analysis
    print("Analyzing NVDA...")
    state, decision = ta.propagate("NVDA", "2026-01-15")

    print("\nDecision:", decision)
    print("Type:", type(decision))


# Example 2: Using OpenClaw to proxy OpenAI subscription
def example_openclaw_openai_proxy():
    """Use OpenClaw to proxy your OpenAI API calls."""
    print("=" * 60)
    print("Example 2: OpenClaw Proxying OpenAI")
    print("=" * 60)

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "openai"  # Use OpenAI directly
    config["deep_think_llm"] = "gpt-4"

    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not set. Skipping example.")
        return

    ta = TradingAgentsGraph(debug=False, config=config)

    # Run analysis
    print("Analyzing TSLA...")
    state, decision = ta.propagate("TSLA", "2026-01-15")

    print("\nDecision:", decision)


# Example 3: Using OpenClaw programmatically from skill
def example_skill_usage():
    """Simulate how the OpenClaw skill calls TradingAgents."""
    print("=" * 60)
    print("Example 3: Skill Integration Pattern")
    print("=" * 60)

    from skills.openclaw.trading_analyzer import analyze_stock, parse_trading_command

    # Parse user input
    user_input = "Analyze AAPL as of 2024-05-10"
    ticker, date = parse_trading_command(user_input)

    print(f"Parsed ticker: {ticker}, date: {date}")

    # Run analysis with OpenClaw provider
    report = analyze_stock(
        ticker=ticker,
        date=date,
        llm_provider="openclaw",
        llm_model="gpt-4"
    )

    print("\nReport:")
    print(report)


# Example 4: Direct OpenClaw client usage
def example_openclaw_client():
    """Direct usage of OpenClawClient."""
    print("=" * 60)
    print("Example 4: Direct OpenClaw Client")
    print("=" * 60)

    from tradingagents.llm_clients.factory import create_llm_client

    # Create OpenClaw client
    client = create_llm_client(
        provider="openclaw",
        model="gpt-4",
        base_url="http://localhost:8000/v1"  # Your OpenClaw endpoint
    )

    # Get LLM instance
    llm = client.get_llm()

    print(f"LLM Client: {client.__class__.__name__}")
    print(f"Model: {client.model}")
    print(f"Base URL: {client.base_url}")
    print(f"LLM Type: {llm.__class__.__name__}")

    # Test a simple call
    response = llm.invoke("What is 2+2?")
    print(f"\nTest query response: {response.content[:100]}...")


# Example 5: Configuration flexibility
def example_configuration():
    """Show how to configure different scenarios."""
    print("=" * 60)
    print("Example 5: Configuration Scenarios")
    print("=" * 60)

    scenarios = [
        {
            "name": "Personal OpenAI subscription",
            "provider": "openai",
            "model": "gpt-4",
            "env": "OPENAI_API_KEY"
        },
        {
            "name": "Anthropic Claude",
            "provider": "anthropic",
            "model": "claude-3-opus",
            "env": "ANTHROPIC_API_KEY"
        },
        {
            "name": "Google Gemini",
            "provider": "google",
            "model": "gemini-2.0-pro",
            "env": "GOOGLE_API_KEY"
        },
        {
            "name": "DeepSeek Reasoning",
            "provider": "deepseek",
            "model": "deepseek-reasoner",
            "env": "DEEPSEEK_API_KEY"
        },
        {
            "name": "Local Ollama",
            "provider": "ollama",
            "model": "mistral",
            "env": "OLLAMA_BASE_URL"
        },
        {
            "name": "OpenClaw Proxy",
            "provider": "openclaw",
            "model": "gpt-4",
            "env": "OPENCLAW_BASE_URL (http://localhost:8000/v1)"
        },
    ]

    for scenario in scenarios:
        status = "✓" if os.getenv(scenario["env"]) else "✗"
        print(f"{status} {scenario['name']}")
        print(f"  Provider: {scenario['provider']}")
        print(f"  Model: {scenario['model']}")
        print(f"  Requires: {scenario['env']}\n")


# Run examples
if __name__ == "__main__":
    print("\n")
    print("🦞 TradingAgents + OpenClaw Examples")
    print("=" * 60)
    print()

    # Example 5: Show configurations
    example_configuration()

    # Example 4: Direct client usage
    try:
        example_openclaw_client()
    except Exception as e:
        print(f"⚠️ Error: {e}")
        print("Ensure OpenClaw is running at http://localhost:8000/v1\n")

    # Example 3: Skill usage
    try:
        example_skill_usage()
    except Exception as e:
        print(f"⚠️ Error: {e}\n")

    # Example 2: OpenAI
    try:
        example_openclaw_openai_proxy()
    except Exception as e:
        print(f"⚠️ Error: {e}\n")

    print("\n" + "=" * 60)
    print("For more information, see:")
    print("  - OPENCLAW_INTEGRATION.md")
    print("  - OPENCLAW_QUICK_START.md")
    print("  - skills/openclaw/README.md")
    print("=" * 60)
