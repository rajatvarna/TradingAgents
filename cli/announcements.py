import os

import requests
from rich.console import Console
from rich.panel import Panel

from cli.config import CLI_CONFIG


def fetch_announcements(url: str = None, timeout: float = None) -> dict:
    """Fetch announcements from endpoint. Returns dict with announcements and settings."""
    # Allow disabling remote announcements via env var (see #1262)
    if os.getenv("TRADINGAGENTS_DISABLE_ANNOUNCEMENTS", "").strip().lower() in ("1", "true", "yes", "on"):
        return {
            "announcements": [],
            "require_attention": False,
        }
    endpoint = url or CLI_CONFIG["announcements_url"]
    timeout = timeout or CLI_CONFIG["announcements_timeout"]
    fallback = CLI_CONFIG["announcements_fallback"]

    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        # Handle untrusted content safely (clip length, strip control chars)
        announcements = data.get("announcements", [fallback])
        if not isinstance(announcements, list):
            announcements = [str(announcements)]
        # Clip each announcement to reasonable length to prevent display overflow
        safe = []
        for a in announcements:
            s = str(a)[:2000].replace("\r", "").strip()
            if s:
                safe.append(s)
        if not safe:
            safe = [fallback]
        return {
            "announcements": safe,
            "require_attention": bool(data.get("require_attention", False)),
        }
    except Exception:
        return {
            "announcements": [fallback],
            "require_attention": False,
        }


def display_announcements(console: Console, data: dict) -> None:
    """Display announcements panel. Prompts for Enter if require_attention is True."""
    announcements = data.get("announcements", [])
    require_attention = data.get("require_attention", False)

    if not announcements:
        return

    content = "\n".join(announcements)

    panel = Panel(
        content,
        border_style="cyan",
        padding=(1, 2),
        title="Announcements",
    )
    console.print(panel)

    if require_attention:
        input("Press Enter to continue...")
    else:
        console.print()
