"""Persist CLI selections between runs.

Saves LLM provider, base URL, and model choices to
~/.tradingagents/cli_preferences.json so the user doesn't
have to re-enter them every time.
"""

import json
import os
from typing import Any, Dict, Optional

_PREFS_PATH = os.path.join(
    os.path.expanduser("~"), ".tradingagents", "cli_preferences.json"
)

_SAVED_KEYS = (
    "llm_provider",
    "backend_url",
    "shallow_thinker",
    "deep_thinker",
)


def load_preferences() -> Dict[str, Any]:
    try:
        with open(_PREFS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_preferences(selections: Dict[str, Any]) -> None:
    data = {k: selections[k] for k in _SAVED_KEYS if k in selections}
    os.makedirs(os.path.dirname(_PREFS_PATH), exist_ok=True)
    with open(_PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
