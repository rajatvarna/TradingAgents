import questionary
from typing import List, Optional, Tuple, Dict

from rich.console import Console

from cli.models import AnalystType
from cli.preferences import load_preferences
from tradingagents.llm_clients.model_catalog import get_model_options

console = Console()

_prefs = load_preferences()

TICKER_INPUT_EXAMPLES = "Examples: SPY, CNC.TO, 7203.T, 0700.HK"

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Sentiment Analyst", AnalystType.SENTIMENT),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        f"Enter the exact ticker symbol to analyze ({TICKER_INPUT_EXAMPLES}):",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker)


def normalize_ticker_symbol(ticker: str) -> str:
    """Normalize ticker input while preserving exchange suffixes."""
    return ticker.strip().upper()


def get_batch_tickers() -> List[str]:
    """Prompt the user to enter multiple ticker symbols (comma-separated)."""
    tickers_input = questionary.text(
        f"Enter tickers separated by commas ({TICKER_INPUT_EXAMPLES}):",
        validate=lambda x: len(x.strip()) > 0 or "Please enter at least one ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not tickers_input:
        console.print("\n[red]No tickers provided. Exiting...[/red]")
        exit(1)

    raw_tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if not raw_tickers:
        console.print("\n[red]No valid tickers found. Exiting...[/red]")
        exit(1)

    normalized = [normalize_ticker_symbol(t) for t in raw_tickers]
    unique_tickers = list(dict.fromkeys(normalized))

    if len(unique_tickers) < len(normalized):
        console.print(f"[yellow]Removed duplicate tickers. {len(unique_tickers)} unique ticker(s) to analyze.[/yellow]")

    return unique_tickers


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def _fetch_openrouter_models() -> List[Tuple[str, str]]:
    """Fetch available models from the OpenRouter API."""
    import requests
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return [(m.get("name") or m["id"], m["id"]) for m in models]
    except Exception as e:
        console.print(f"\n[yellow]Could not fetch OpenRouter models: {e}[/yellow]")
        return []


def select_openrouter_model() -> str:
    """Select an OpenRouter model from the newest available, or enter a custom ID."""
    models = _fetch_openrouter_models()

    choices = [questionary.Choice(name, value=mid) for name, mid in models[:5]]
    choices.append(questionary.Choice("Custom model ID", value="custom"))

    choice = questionary.select(
        "Select OpenRouter Model (latest available):",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None or choice == "custom":
        return questionary.text(
            "Enter OpenRouter model ID (e.g. google/gemma-4-26b-a4b-it):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
        ).ask().strip()

    return choice


def _prompt_custom_model_id() -> str:
    """Prompt user to type a custom model ID."""
    return questionary.text(
        "Enter model ID:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
    ).ask().strip()


def _select_model(provider: str, mode: str) -> str:
    """Select a model for the given provider and mode (quick/deep)."""
    if provider.lower() == "openrouter":
        return select_openrouter_model()

    if provider.lower() == "custom_openai":
        saved_key = "shallow_thinker" if mode == "quick" else "deep_thinker"
        saved_model = _prefs.get(saved_key, "")
        model_name = questionary.text(
            f"Enter model name for {mode}-thinking (as shown in your server):",
            default=saved_model,
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model name.",
        ).ask()
        if model_name is None:
            console.print(f"\n[red]No {mode} thinking model name provided. Exiting...[/red]")
            exit(1)
        return model_name.strip()

    if provider.lower() == "azure":
        return questionary.text(
            f"Enter Azure deployment name ({mode}-thinking):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a deployment name.",
        ).ask().strip()

    if provider.lower() == "deepinfra":
        return questionary.text(
            f"Enter DeepInfra model ID ({mode}-thinking):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
        ).ask().strip()

    choice = questionary.select(
        f"Select Your [{mode.title()}-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, mode)
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(f"\n[red]No {mode} thinking llm engine selected. Exiting...[/red]")
        exit(1)

    if choice == "custom":
        return _prompt_custom_model_id()

    return choice


def select_shallow_thinking_agent(provider) -> str:
    """Select shallow thinking llm engine using an interactive selection."""
    return _select_model(provider, "quick")


def select_deep_thinking_agent(provider) -> str:
    """Select deep thinking llm engine using an interactive selection."""
    return _select_model(provider, "deep")

def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider and its API endpoint."""
    # (display_name, provider_key, base_url)
    PROVIDERS = [
        ("OpenAI", "openai", "https://api.openai.com/v1"),
        ("Google", "google", None),
        ("Anthropic", "anthropic", "https://api.anthropic.com/"),
        ("xAI", "xai", "https://api.x.ai/v1"),
        ("DeepSeek", "deepseek", "https://api.deepseek.com"),
        ("Qwen", "qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("GLM", "glm", "https://open.bigmodel.cn/api/paas/v4/"),
        ("OpenRouter", "openrouter", "https://openrouter.ai/api/v1"),
        ("DeepInfra", "deepinfra", "https://api.deepinfra.com/v1/openai"),
        ("Azure OpenAI", "azure", None),
        ("AWS Bedrock", "bedrock", None),
        ("GitHub Copilot", "github_copilot", "https://models.github.ai/inference"),
        ("Ollama", "ollama", "http://localhost:11434/v1"),
        ("Ollama Cloud", "ollama_cloud", "https://ollama.com/v1"),
        ("LM Studio", "lm-studio", None),
        ("Llama.cpp", "llama-cpp", None),
        ("Custom OpenAI-Compatible (vLLM, etc.)", "custom_openai", None),
    ]

    saved_provider = _prefs.get("llm_provider")
    provider_choices = [
        questionary.Choice(display, value=(provider_key, url))
        for display, provider_key, url in PROVIDERS
    ]
    default_choice = None
    if saved_provider:
        for display, provider_key, url in PROVIDERS:
            if provider_key == saved_provider:
                default_choice = (provider_key, url)
                break

    choice = questionary.select(
        "Select your LLM Provider:",
        choices=provider_choices,
        default=default_choice,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No LLM provider selected. Exiting...[/red]")
        exit(1)

    provider, url = choice

    if provider == "custom_openai":
        saved_url = _prefs.get("backend_url") or "http://localhost:1234/v1"
        url = questionary.text(
            "Enter base URL for your OpenAI-compatible endpoint:",
            default=saved_url,
            validate=lambda x: len(x.strip()) > 0 or "Please enter a base URL.",
            style=questionary.Style([
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]),
        ).ask()
        if not url:
            console.print("\n[red]No base URL provided. Exiting...[/red]")
            exit(1)
        url = url.strip()

    return provider, url


def ask_openai_reasoning_effort() -> str:
    """Ask for OpenAI reasoning effort level."""
    choices = [
        questionary.Choice("Medium (Default)", "medium"),
        questionary.Choice("High (More thorough)", "high"),
        questionary.Choice("Low (Faster)", "low"),
    ]
    return questionary.select(
        "Select Reasoning Effort:",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_anthropic_effort() -> str | None:
    """Ask for Anthropic effort level.

    Controls token usage and response thoroughness on Claude 4.5+ and 4.6 models.
    """
    return questionary.select(
        "Select Effort Level:",
        choices=[
            questionary.Choice("High (recommended)", "high"),
            questionary.Choice("Medium (balanced)", "medium"),
            questionary.Choice("Low (faster, cheaper)", "low"),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_gemini_thinking_config() -> str | None:
    """Ask for Gemini thinking configuration.

    Returns thinking_level: "high" or "minimal".
    Client maps to appropriate API param based on model series.
    """
    return questionary.select(
        "Select Thinking Mode:",
        choices=[
            questionary.Choice("Enable Thinking (recommended)", "high"),
            questionary.Choice("Minimal/Disable Thinking", "minimal"),
        ],
        style=questionary.Style([
            ("selected", "fg:green noinherit"),
            ("highlighted", "fg:green noinherit"),
            ("pointer", "fg:green noinherit"),
        ]),
    ).ask()


def ask_output_language() -> str:
    """Ask for report output language."""
    choice = questionary.select(
        "Select Output Language:",
        choices=[
            questionary.Choice("English (default)", "English"),
            questionary.Choice("Chinese (中文)", "Chinese"),
            questionary.Choice("Japanese (日本語)", "Japanese"),
            questionary.Choice("Korean (한국어)", "Korean"),
            questionary.Choice("Hindi (हिन्दी)", "Hindi"),
            questionary.Choice("Spanish (Español)", "Spanish"),
            questionary.Choice("Portuguese (Português)", "Portuguese"),
            questionary.Choice("French (Français)", "French"),
            questionary.Choice("German (Deutsch)", "German"),
            questionary.Choice("Arabic (العربية)", "Arabic"),
            questionary.Choice("Russian (Русский)", "Russian"),
            questionary.Choice("Custom language", "custom"),
        ],
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice == "custom":
        return questionary.text(
            "Enter language name (e.g. Turkish, Vietnamese, Thai, Indonesian):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a language name.",
        ).ask().strip()

    return choice


def ask_investment_horizon() -> str:
    """Ask for investment time horizon."""
    choice = questionary.select(
        "Select Investment Horizon:",
        choices=[
            questionary.Choice("Day trading / Intraday", "1_day"),
            questionary.Choice("Swing trading / Short-term (1 week)", "1_week"),
            questionary.Choice("Medium-term trading (1 month)", "1_month"),
            questionary.Choice("Medium-term investing (6 months)", "6_months"),
            questionary.Choice("Long-term investing (1 year)", "1_year"),
            questionary.Choice("Long-term strategic allocation (5+ years)", "5_years_plus"),
            questionary.Choice("Medium-term (default)", "medium_term"),
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No investment horizon selected. Exiting...[/red]")
        exit(1)

    return choice


def ask_llm_timeout() -> int | None:
    """Ask for LLM request timeout (seconds). Returns None for provider default."""
    TIMEOUT_OPTIONS = [
        ("Default (provider decides)", None),
        ("5 minutes (300s) - fast local models", 300),
        ("15 minutes (900s) - moderate local models", 900),
        ("30 minutes (1800s) - slow or large local models", 1800),
        ("Custom", "custom"),
    ]

    choice = questionary.select(
        "Select Request Timeout:",
        choices=[
            questionary.Choice(display, value=value) for display, value in TIMEOUT_OPTIONS
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()

    if choice == "custom":
        val = questionary.text(
            "Enter timeout in seconds:",
            validate=lambda x: x.strip().isdigit() and int(x.strip()) > 0
            or "Enter a positive integer.",
        ).ask()
        return int(val.strip()) if val else None

    return choice
