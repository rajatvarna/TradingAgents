import os

import requests
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from cli.config import CLI_CONFIG

_DISABLE_ENV = "TRADINGAGENTS_DISABLE_ANNOUNCEMENTS"
_BOOL_TRUE = ("true", "1", "yes", "on")


def announcements_disabled() -> bool:
    """True when the user opted out of the remote announcements fetch."""
    return os.getenv(_DISABLE_ENV, "").strip().lower() in _BOOL_TRUE


def fetch_announcements(url: str = None, timeout: float = None) -> dict:
    """Fetch announcements from endpoint. Returns dict with announcements and settings.

    Set ``TRADINGAGENTS_DISABLE_ANNOUNCEMENTS=1`` to skip the network call and
    return an empty payload (no panel shown).
    """
    # Allow disabling remote announcements via env var (see #1262)
    if announcements_disabled():
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
        # Remote content is untrusted: coerce to plain strings, cap volume
        # (20 items) and length (2000 chars each) to prevent display overflow.
        raw_items = data.get("announcements", [fallback])
        if not isinstance(raw_items, list):
            raw_items = [str(raw_items)]
        safe = []
        for a in raw_items[:20]:
            if a is None:
                continue
            s = str(a)[:2000].replace("\r", "").strip()
            if s:
                safe.append(s)
        if not safe:
            return {
                "announcements": [fallback],
                "require_attention": False,
                "trusted_fallback": True,
            }
        return {
            "announcements": safe,
            # Never honor a remote require_attention flag — a compromised
            # endpoint must not stall the CLI.
            "require_attention": False,
            "trusted_fallback": False,
        }
    except Exception:
        return {
            "announcements": [fallback],
            "require_attention": False,
            "trusted_fallback": True,
        }


def display_announcements(console: Console, data: dict) -> None:
    """Display announcements panel.

    Remote announcement text is escaped so Rich markup/links in the payload
    cannot spoof branded UI. The local fallback string may contain trusted
    Rich markup and is rendered as-is.
    """
    announcements = data.get("announcements", [])
    require_attention = data.get("require_attention", False)

    if not announcements:
        return

    if data.get("trusted_fallback"):
        content = "\n".join(announcements)
    else:
        content = "\n".join(escape(str(a)) for a in announcements)

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
