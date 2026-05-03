"""Trading Analyzer Skill for OpenClaw

Enables OpenClaw users to perform multi-agent stock trading analysis
via Telegram, WhatsApp, Discord, and other chat platforms.
"""

from .trading_analyzer import (
    handle_trading_query,
    analyze_stock,
    parse_trading_command,
    SKILL_MANIFEST,
)

__version__ = "1.0.0"
__all__ = [
    "handle_trading_query",
    "analyze_stock",
    "parse_trading_command",
    "SKILL_MANIFEST",
]
