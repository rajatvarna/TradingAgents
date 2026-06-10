import questionary
import os
from typing import List, Optional, Tuple, Dict

from rich.console import Console

from cli.models import AnalystType, AssetType
from cli.preferences import load_preferences
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.custom_provider_config import get_custom_provider_choices
from tradingagents.llm_clients.oauth import (
    login as oauth_login,
    ensure_token,
    available_models as oauth_available_models,
    OAuthTokenStore,
    OAuthNotLoggedIn,
    OAuthError,
)
from tradingagents.llm_clients.url_validation import validate_custom_provider_base_url

console = Console()

_prefs = load_preferences()

TICKER_INPUT_EXAMPLES = "Examples: SPY, PETR4, VALE3, CNC.TO, 7203.T, 0700.HK, BTC-USD"

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Sentiment Analyst", AnalystType.SENTIMENT),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
    ("ESG Analyst", AnalystType.ESG),
    ("Derivatives Analyst (mandatory)", AnalystType.DERIVATIVES),
]

CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")


def detect_asset_type(ticker: str) -> AssetType:
    """Detect whether ticker symbol is a stock or a crypto asset."""
    normalized_ticker = ticker.strip().upper()
    if normalized_ticker.endswith(CRYPTO_SUFFIXES):
        return AssetType.CRYPTO
    return AssetType.STOCK


def filter_analysts_for_asset_type(
    analysts: List[AnalystType], asset_type: AssetType
) -> List[AnalystType]:
    """Filter out fundamentals analyst for crypto assets."""
    if asset_type != AssetType.CRYPTO:
        return analysts
    return [
        analyst
        for analyst in analysts
        if analyst != AnalystType.FUNDAMENTALS
    ]


def get_ticker(default_ticker: str = "SPY") -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        f"Enter the exact ticker symbol to analyze ({TICKER_INPUT_EXAMPLES}):",
        default=default_ticker,
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


def select_market() -> str:
    """Select the target market."""
    return questionary.select(
        "Select Target Market:",
        choices=[
            questionary.Choice([("fg:red", "US/Global (S&P 500, etc.)")], "US"),
            questionary.Choice([("fg:green", "B3 (Brazilian Stock Exchange)")], "B3"),
        ],
        style=questionary.Style([
            ("selected", "fg:blue noinherit"),
            ("highlighted", "fg:blue noinherit"),
            ("pointer", "fg:blue noinherit"),
        ]),
    ).ask()


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

    def validate_date(date_str: str) -> bool | str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return "Please enter a valid date in YYYY-MM-DD format."
        try:
            analysis_date = datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.today().date():
                return "Analysis date cannot be in the future."
            return True
        except ValueError:
            return "Please enter a valid date in YYYY-MM-DD format."

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        default=datetime.today().strftime("%Y-%m-%d"),
        validate=lambda x: validate_date(x.strip()),
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


def select_analysts(asset_type: AssetType = AssetType.STOCK) -> List[AnalystType]:
    """Select analysts using an interactive checkbox.

    The Derivatives Analyst is mandatory on every run (enforced in
    ``TradingAgentsGraph.__init__``); it is shown pre-checked here so the
    CLI's selection matches the graph-level guarantee, and re-added below
    in case the user unchecks it.
    """
    available_analysts = filter_analysts_for_asset_type(
        [value for _, value in ANALYST_ORDER],
        asset_type,
    )
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(
                display,
                value=value,
                checked=(value == AnalystType.DERIVATIVES),
            )
            for display, value in ANALYST_ORDER
            if value in available_analysts
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done\n- Note: Derivatives Analyst is mandatory and will be re-added if you uncheck it",
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

    if AnalystType.DERIVATIVES not in choices:
        choices.append(AnalystType.DERIVATIVES)

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


def _prefer_openrouter_free_models(models: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return models ordered with free OpenRouter models first."""
    free_models: List[Tuple[str, str]] = []
    paid_models: List[Tuple[str, str]] = []
    for name, mid in models:
        model_id = (mid or "").strip().lower()
        if model_id.endswith(":free"):
            free_models.append((name, mid))
        else:
            paid_models.append((name, mid))
    return free_models + paid_models


def select_openrouter_model(mode: str) -> str:
    """Select an OpenRouter model, preferring free-tier models first."""
    models = _prefer_openrouter_free_models(_fetch_openrouter_models())

    choices = [questionary.Choice(name, value=mid) for name, mid in models[:5]]
    choices.append(questionary.Choice("Custom model ID", value="custom"))

    choice = questionary.select(
        f"Select Your [{mode.title()}-Thinking] OpenRouter Model (free models first):",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None or choice == "custom":
        custom_id = questionary.text(
            "Enter OpenRouter model ID (e.g. deepseek/deepseek-r1-0528:free):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
        ).ask()
        return custom_id.strip() if custom_id else ""

    return choice


def _prompt_custom_model_id() -> str:
    """Prompt user to type a custom model ID."""
    return questionary.text(
        "Enter model ID:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
    ).ask().strip()


def _oauth_available_model_ids(refresh: bool = False):
    """Modelli usabili dall'account ChatGPT, o None se non scopribili.

    Sonda una sola volta (con cache): il primo _select_model fa la scoperta, il
    secondo legge la cache. In caso di errore ritorna None -> catalogo completo.
    """
    candidates = [
        value
        for mode in ("quick", "deep")
        for _, value in get_model_options("openai-oauth", mode)
    ]
    try:
        with console.status("[cyan]Rilevo i modelli disponibili per il tuo account ChatGPT...[/cyan]"):
            models = oauth_available_models(OAuthTokenStore(), candidates, refresh=refresh)
        return set(models) if models else None
    except Exception:
        return None
def _validate_custom_provider_url_input(value: str) -> bool | str:
    try:
        validate_custom_provider_base_url(value)
        return True
    except ValueError as exc:
        return str(exc)


def prompt_custom_provider_backend_url() -> str:
    """Prompt for a custom OpenAI-compatible provider base URL."""
    url = questionary.text(
        "Enter custom OpenAI-compatible base URL (e.g. https://api.example.com/v1):",
        validate=_validate_custom_provider_url_input,
    ).ask()
    if url is None:
        console.print("\n[red]No custom provider backend URL provided. Exiting...[/red]")
        exit(1)
    return validate_custom_provider_base_url(url)


def _select_model(provider: str, mode: str) -> str:
    """Select a model for the given provider and mode (quick/deep)."""
    if provider.lower() == "openrouter":
        return select_openrouter_model(mode)

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

    if provider.lower() == "custom":
        model = questionary.text(
            f"Enter custom provider model ID ({mode}-thinking):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
        ).ask()
        if model is None:
            console.print("\n[red]No model ID provided. Exiting...[/red]")
            exit(1)
        return model.strip()

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

    options = get_model_options(provider, mode)

    # ChatGPT OAuth: mostra solo i modelli realmente abilitati per il piano
    # dell'utente (gli altri verrebbero rifiutati dal backend con HTTP 400).
    if provider.lower() == "openai-oauth":
        available = _oauth_available_model_ids()
        if available:
            filtered = [(d, v) for d, v in options if v in available or v == "custom"]
            if filtered:
                options = filtered
        else:
            console.print(
                "[yellow]Impossibile rilevare i modelli abilitati; mostro l'intero "
                "catalogo (alcuni potrebbero non essere disponibili sul tuo piano).[/yellow]"
            )

    choice = questionary.select(
        f"Select Your [{mode.title()}-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in options
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

def _llm_provider_table() -> list[tuple[str, str, str | None]]:
    """(display_name, provider_key, base_url) for every supported provider.

    Shared by the interactive picker and by env-driven configuration so an
    env-set provider resolves to the same default endpoint the menu uses.
    Ollama users can point at a remote ollama-serve via OLLAMA_BASE_URL
    (convention from the broader Ollama ecosystem); falls back to the
    localhost default when unset.
    """
    # Resolve local-runtime URLs through the same function the LLM client uses
    # so default values and env-var overrides live in exactly one place.
    # Imported lazily to avoid pulling langchain_openai into CLI startup.
    from tradingagents.llm_clients.openai_client import resolve_provider_base_url
    ollama_url = resolve_provider_base_url("ollama")
    lmstudio_url = resolve_provider_base_url("lmstudio")
    # (display_name, provider_key, base_url)
    PROVIDERS = [
        ("OpenAI", "openai", "https://api.openai.com/v1"),
        ("OpenAI (ChatGPT OAuth)", "openai-oauth", None),
        ("Google", "google", None),
        ("Anthropic", "anthropic", "https://api.anthropic.com/"),
        (
            "Tencent Cloud LKEAP",
            "tencent",
            "https://api.lkeap.cloud.tencent.com/plan/anthropic",
        ),
        ("xAI", "xai", "https://api.x.ai/v1"),
        ("DeepSeek", "deepseek", "https://api.deepseek.com"),
        ("Qwen", "qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("GLM", "glm", "https://open.bigmodel.cn/api/paas/v4/"),
        ("Kimi", "kimi", os.environ.get("KIMI_BASE_URL") or "https://api.moonshot.ai/v1"),
        ("MiniMax", "minimax", "https://api.minimax.io/v1"),
        ("NVIDIA NIM", "nvidia_nim", "https://integrate.api.nvidia.com/v1"),
        ("OpenRouter", "openrouter", "https://openrouter.ai/api/v1"),
        ("Opencode", "opencode", os.environ.get("OPENCODE_BASE_URL") or "https://opencode.ai/zen/go/v1"),
        ("DeepInfra", "deepinfra", "https://api.deepinfra.com/v1/openai"),
        ("MiMo", "mimo", "https://token-plan-sgp.xiaomimimo.com/v1"),
        ("Custom OpenAI-compatible", "custom", None),
        ("Azure OpenAI", "azure", None),
        ("AWS Bedrock", "bedrock", None),
        ("GitHub Copilot", "github_copilot", "https://models.github.ai/inference"),
        ("Ollama", "ollama", ollama_url),
        ("Ollama Cloud", "ollama_cloud", "https://ollama.com/v1"),
        ("LM Studio", "lmstudio", lmstudio_url),
        ("LM Studio", "lmstudio", lmstudio_url),
        ("Llama.cpp", "llama-cpp", None),
        ("Custom OpenAI-Compatible (vLLM, etc.)", "custom_openai", None),
    ]

    existing_provider_keys = {provider_key for _, provider_key, _ in PROVIDERS}
    for display, provider_key, url in get_custom_provider_choices():
        if provider_key not in existing_provider_keys:
            PROVIDERS.append((display, provider_key, url))
            existing_provider_keys.add(provider_key)

    return PROVIDERS


def provider_default_url(provider_key: str) -> str | None:
    """Return the default backend URL for a provider key, or None if unknown."""
    key = provider_key.lower()
    for _, pk, url in _llm_provider_table():
        if pk == key:
            return url
    return None


def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider and its API endpoint."""
    PROVIDERS = _llm_provider_table()

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

    if provider == "custom":
        url = prompt_custom_provider_backend_url()
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


def ask_glm_region() -> tuple[str, str]:
    """Ask which GLM platform (Z.AI international vs BigModel China) to use.

    Zhipu serves the same GLM models under two brands with separate
    accounts; keys aren't interchangeable. Returns (provider_key, backend_url).
    """
    return questionary.select(
        "Select GLM platform:",
        choices=[
            questionary.Choice(
                "Z.AI — api.z.ai (international, uses ZHIPU_API_KEY)",
                value=("glm", "https://api.z.ai/api/paas/v4/"),
            ),
            questionary.Choice(
                "BigModel — open.bigmodel.cn (China, uses ZHIPU_CN_API_KEY)",
                value=("glm-cn", "https://open.bigmodel.cn/api/paas/v4/"),
            ),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_qwen_region() -> tuple[str, str]:
    """Ask which Qwen region (international vs China) to use.

    Alibaba DashScope exposes two endpoints with separate accounts —
    a key from one region does NOT authenticate against the other
    (fixes #758). Returns (provider_key, backend_url).
    """
    return questionary.select(
        "Select Qwen region:",
        choices=[
            questionary.Choice(
                "International — dashscope-intl.aliyuncs.com (uses DASHSCOPE_API_KEY)",
                value=("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            ),
            questionary.Choice(
                "China — dashscope.aliyuncs.com (uses DASHSCOPE_CN_API_KEY)",
                value=("qwen-cn", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_minimax_region() -> tuple[str, str]:
    """Ask which MiniMax region (global vs China) to use.

    MiniMax exposes two endpoints with separate accounts — a key from
    one region does NOT authenticate against the other. Returns
    (provider_key, backend_url).
    """
    return questionary.select(
        "Select MiniMax region:",
        choices=[
            questionary.Choice(
                "Global — api.minimax.io (uses MINIMAX_API_KEY)",
                value=("minimax", "https://api.minimax.io/v1"),
            ),
            questionary.Choice(
                "China — api.minimaxi.com (uses MINIMAX_CN_API_KEY)",
                value=("minimax-cn", "https://api.minimaxi.com/v1"),
            ),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def confirm_ollama_endpoint(url: str) -> None:
    """Show the resolved Ollama endpoint after provider selection.

    Surfaces three things the user benefits from seeing before model
    selection: which URL we'll actually hit, where it came from
    (``OLLAMA_BASE_URL`` vs default), and a soft warning if the URL is
    missing the scheme/port that ollama-serve expects. The warning is
    advisory only — we don't reject malformed input, since the user may
    be doing something deliberately unusual (e.g. a reverse-proxy path).
    """
    from_env = os.environ.get("OLLAMA_BASE_URL")
    origin = " (from OLLAMA_BASE_URL)" if from_env and from_env == url else ""
    console.print(f"[green]✓ Using Ollama at {url}{origin}[/green]")

    if not url.startswith(("http://", "https://")):
        console.print(
            f"[yellow]Note: {url!r} is missing a scheme. "
            f"Ollama-serve typically expects a URL like "
            f"http://<host>:11434/v1.[/yellow]"
        )
    elif ":11434" not in url and "://localhost" not in url and "://127.0.0.1" not in url:
        # Soft hint when the port differs from the ollama-serve default
        # and the host isn't local (where users sometimes proxy on :80).
        console.print(
            f"[yellow]Note: {url!r} doesn't include port 11434. "
            f"Make sure your remote ollama-serve listens on the port "
            f"shown above.[/yellow]"
        )


def confirm_lmstudio_endpoint(url: str) -> None:
    """Show the resolved LM Studio endpoint after provider selection.

    Mirrors confirm_ollama_endpoint: surfaces the URL, its origin
    (LMSTUDIO_BASE_URL vs default), and soft warnings for missing scheme
    or unexpected port. LM Studio's default server port is 1234.
    """
    from_env = os.environ.get("LMSTUDIO_BASE_URL")
    origin = " (from LMSTUDIO_BASE_URL)" if from_env and from_env == url else ""
    console.print(f"[green]✓ Using LM Studio at {url}{origin}[/green]")

    if not url.startswith(("http://", "https://")):
        console.print(
            f"[yellow]Note: {url!r} is missing a scheme. "
            f"LM Studio typically expects a URL like "
            f"http://<host>:1234/v1.[/yellow]"
        )
    elif ":1234" not in url and "://localhost" not in url and "://127.0.0.1" not in url:
        console.print(
            f"[yellow]Note: {url!r} doesn't include port 1234. "
            f"Make sure your LM Studio server listens on the port "
            f"shown above.[/yellow]"
        )


def ensure_api_key(provider: str) -> Optional[str]:
    """Make sure the API key for `provider` is available in the environment.

    If the env var is already set, returns its value untouched. Otherwise
    interactively prompts the user, persists the value to the project's
    .env file via python-dotenv's set_key (creating .env if needed), and
    exports it into os.environ so the current process picks it up.

    Returns None for providers that do not require a key (e.g. ollama)
    and for providers not found in the canonical mapping.
    """
    from tradingagents.llm_clients.api_key_env import get_api_key_env
    from dotenv import find_dotenv, set_key
    from pathlib import Path
    
    env_var = get_api_key_env(provider)
    if env_var is None:
        return None  # ollama / unknown — no key check possible

    existing = os.environ.get(env_var)
    if existing:
        return existing

    console.print(
        f"\n[yellow]{env_var} is not set in your environment.[/yellow]"
    )
    key = questionary.password(
        f"Paste your {env_var} (will be saved to .env):",
        style=questionary.Style([
            ("text", "fg:cyan"),
            ("highlighted", "noinherit"),
        ]),
    ).ask()
    if not key:
        console.print(
            f"[red]Skipped. API calls will fail until {env_var} is set.[/red]"
        )
        return None

    env_path = find_dotenv(usecwd=True) or str(Path.cwd() / ".env")
    Path(env_path).touch(exist_ok=True)
    set_key(env_path, env_var, key)
    os.environ[env_var] = key
    console.print(f"[green]Saved {env_var} to {env_path}[/green]")
    return key


def ensure_oauth_login(provider: str):
    """Garantisce un token OAuth valido per 'openai-oauth', altrimenti fa login.

    Per gli altri provider è un no-op. Se l'utente annulla il login, esce.
    Ritorna gli StoredTokens (con account_id) o None per provider non-OAuth.
    """
    if provider.lower() != "openai-oauth":
        return None
    try:
        return ensure_token()
    except OAuthNotLoggedIn:
        console.print(
            "\n[yellow]Nessun login ChatGPT trovato. Apro il browser per "
            "l'autenticazione OAuth (Sign in with ChatGPT)...[/yellow]"
        )
    except OAuthError as exc:
        console.print(f"[yellow]Token OAuth non utilizzabile ({exc}). Rifaccio il login...[/yellow]")
    try:
        tokens = oauth_login()
    except Exception as exc:  # OAuthLoginError e simili
        console.print(f"[red]Login OAuth fallito: {exc}[/red]")
        exit(1)
    acct = getattr(tokens, "account_id", None) or "(account sconosciuto)"
    console.print(f"[green]Login ChatGPT completato. Account: {acct}[/green]")
    # Aggiorna la cache dei modelli disponibili per il nuovo account.
    _oauth_available_model_ids(refresh=True)
    return tokens
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
            questionary.Choice("Italian (Italiano)", "Italian"),
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
