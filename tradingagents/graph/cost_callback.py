"""Per-run token accumulator.

Cost guard policy (program-design Appendix A): measurement is unconditional
and always on. Enforcement is gated by ``cost_guard_enabled`` which ships
as False.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class RunCostCallback(BaseCallbackHandler):
    """Accumulates token counts grouped by model name for one run.

    Use one instance per ``TradingAgentsGraph`` run. The Run Recorder reads
    ``totals_by_model()`` when the run finishes and persists rows to the
    ``costs`` table.
    """

    def __init__(self, cost_guard: CostGuard | None = None) -> None:
        """Initialise with an optional CostGuard for budget enforcement."""
        self._totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"in_tokens": 0, "out_tokens": 0}
        )
        self._cost_guard = cost_guard

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Accumulate token counts from the completed LLM call and check budget."""
        info = response.llm_output or {}
        usage = info.get("token_usage") or {}
        model = info.get("model_name") or info.get("model") or "unknown"
        in_t = int(usage.get("prompt_tokens") or 0)
        out_t = int(usage.get("completion_tokens") or 0)
        if in_t == 0 and out_t == 0:
            return
        self._totals[model]["in_tokens"] += in_t
        self._totals[model]["out_tokens"] += out_t
        # Check budget after accumulating tokens
        if self._cost_guard is not None:
            self._cost_guard.check_or_raise(total_tokens=self.total_tokens())

    def totals_by_model(self) -> dict[str, dict[str, int]]:
        """Return per-model token counts as a plain dict."""
        return dict(self._totals)

    def total_tokens(self) -> int:
        """Return the total token count across all models for this run."""
        return sum(
            v["in_tokens"] + v["out_tokens"] for v in self._totals.values()
        )


class CostGuardExceeded(RuntimeError):
    """Raised when a run's token spend exceeds the configured per-run budget."""


class CostGuard:
    """Per-run token-budget enforcement.

    Per IIC-FORGE program design Appendix A, this ship with ``enabled=False``.
    Measurement (via ``RunCostCallback``) is always on; enforcement is gated.
    Flip ``enabled=True`` (or set ``TRADINGAGENTS_COST_GUARD_ENABLED=1``) only
    after collecting empirical cost data via the F5 dashboard.
    """

    def __init__(
        self,
        *,
        per_run_token_budget: int,
        enabled: bool = False,
    ) -> None:
        """Configure the token budget and whether enforcement is active."""
        self._budget = per_run_token_budget
        self._enabled = enabled

    def check_or_raise(self, *, total_tokens: int) -> None:
        """Raise CostGuardExceeded if enforcement is on and budget is exceeded."""
        if not self._enabled:
            return  # measurement only — no enforcement during F0–F5
        if total_tokens > self._budget:
            raise CostGuardExceeded(
                f"token spend {total_tokens} > budget {self._budget}"
            )
