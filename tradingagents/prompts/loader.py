"""Load agent prompt templates from YAML files.

YAML files live next to this module: tradingagents/prompts/<agent_name>.yaml
Each file has a top-level ``system`` key containing the prompt template.
Dynamic parts use Python's str.format placeholders: {variable_name}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str, **kwargs: Any) -> str:
    """Load a prompt template by agent name and optionally fill placeholders.

    Args:
        name: Agent name matching a YAML filename (without .yaml extension).
        **kwargs: Values to substitute for ``{placeholder}`` tokens in the template.

    Returns:
        The prompt string, with placeholders filled if kwargs were provided.

    Raises:
        FileNotFoundError: If no YAML file exists for the given name.
        KeyError: If the YAML file has no ``system`` key.
    """
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No prompt template found for agent '{name}' at {path}")

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    template: str = data["system"]

    if kwargs:
        return template.format(**kwargs)
    return template
