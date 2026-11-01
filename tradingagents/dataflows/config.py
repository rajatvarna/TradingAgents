from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import tradingagents.default_config as default_config

# Use default config but allow it to be overridden
_config: dict[str, Any] | None = None

# The config of the run in progress. A graph binds its own for the length of a
# run, so the data tools it calls read that graph's vendors even when several
# graphs share a process. LangGraph carries the context into tool calls.
_run_config: ContextVar[MappingProxyType[str, Any] | None] = ContextVar(
    "tradingagents_run_config", default=None
)


def _deep_freeze(mapping: dict[str, Any]) -> MappingProxyType[str, Any]:
    """Recursively freeze a dict so all nested levels are read-only.

    Lists are converted to tuples so they also become immutable.
    This avoids the cost of ``deepcopy`` on every read while guaranteeing
    that callers cannot accidentally mutate the shared configuration.
    """
    frozen: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            frozen[key] = _deep_freeze(value)
        elif isinstance(value, list):
            frozen[key] = tuple(value)
        else:
            frozen[key] = value
    return MappingProxyType(frozen)


def initialize_config() -> None:
    """Initialize the configuration with default values."""
    global _config
    if _config is None:
        _config = deepcopy(default_config.DEFAULT_CONFIG)


def _merge(base: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Merge ``config`` into ``base``: dict-valued keys one level deep, scalars replaced."""
    for key, value in deepcopy(config).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def set_config(config: dict[str, Any]) -> None:
    """Update the configuration with custom values.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update like ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    keeps the other nested keys from the default; scalar keys are replaced.
    """
    initialize_config()
    _merge(_config, config)


@contextmanager
def run_config(config: dict[str, Any]):
    """Serve ``config``, over the defaults, to every read made inside the block."""
    merged = _merge(deepcopy(default_config.DEFAULT_CONFIG), config)
    token = _run_config.set(_deep_freeze(merged))
    try:
        yield
    finally:
        _run_config.reset(token)


def get_config() -> MappingProxyType[str, Any]:
    """Get a read-only view of the configuration of the run in progress, else the process-wide one.

    Returns a recursively frozen mapping — any attempt to modify the returned
    value (or any nested dict/list inside it) raises ``TypeError``. This is
    both safer and faster than the previous ``deepcopy``-based approach.
    """
    scoped = _run_config.get()
    if scoped is not None:
        return scoped
    if _config is None:
        initialize_config()
    return _deep_freeze(_config)


# Initialize with default config
initialize_config()
