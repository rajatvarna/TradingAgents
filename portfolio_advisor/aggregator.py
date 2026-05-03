from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Tuple

from .config import Aggressiveness, Config

# Map TradingAgents 5-tier rating to a numeric score for ranking within buckets.
_SIGNAL_SCORE: dict[str, int] = {
    "Buy": 5,
    "Overweight": 4,
    "Hold": 3,
    "Underweight": 2,
    "Sell": 1,
}


@dataclass
class Action:
    ticker: str
    action: str    # "BUY", "SELL", or "HOLD"
    signal: str    # raw TradingAgents 5-tier rating
    score: int     # higher = more conviction in the direction
    rationale: str # first substantive line from the PM decision
    owned: bool


def _is_buy(signal: str, aggressiveness: Aggressiveness) -> bool:
    if aggressiveness == Aggressiveness.AGGRESSIVE:
        return signal in ("Buy", "Overweight")
    return signal == "Buy"


def _is_sell(signal: str, aggressiveness: Aggressiveness) -> bool:
    if aggressiveness == Aggressiveness.AGGRESSIVE:
        return signal in ("Sell", "Underweight")
    return signal == "Sell"


def _extract_rationale(final_state: dict) -> str:
    text = final_state.get("final_trade_decision", "")
    for line in text.splitlines():
        stripped = line.strip("*# \t")
        if len(stripped) > 30:
            return stripped[:200]
    return text[:200]


def aggregate(
    results: List[Tuple[str, dict, str]],
    owned_tickers: set,
    cfg: Config,
    run_date: date,
) -> List[Action]:
    actions: List[Action] = []

    for ticker, final_state, signal in results:
        owned = ticker in owned_tickers
        score = _SIGNAL_SCORE.get(signal, 3)
        rationale = _extract_rationale(final_state)

        if _is_sell(signal, cfg.aggressiveness):
            action = "SELL"
        elif _is_buy(signal, cfg.aggressiveness):
            action = "BUY"
        else:
            action = "HOLD"

        actions.append(Action(
            ticker=ticker,
            action=action,
            signal=signal,
            score=score,
            rationale=rationale,
            owned=owned,
        ))

    # SELL first (highest urgency, lowest score = most bearish first),
    # then BUY (highest score = most bullish first), then HOLD.
    _order = {"SELL": 0, "BUY": 1, "HOLD": 2}

    def _sort_key(a: Action) -> tuple:
        if a.action == "SELL":
            return (_order[a.action], a.score)       # ascending score: worst first
        return (_order[a.action], -a.score)           # descending score: best first

    actions.sort(key=_sort_key)
    return actions
