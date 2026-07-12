"""Strategy primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Protocol

from ops.broker.types import Order
from ops.pipeline_adapter import PipelineAdapter, PipelineResult
from ops.universe import Candidate


@dataclass(frozen=True)
class StrategyOrder:
    order: Order
    reason: str
    candidate: Candidate
    pipeline: PipelineResult


@dataclass(frozen=True)
class ProposeOrdersResult:
    orders: list[StrategyOrder]
    # F3: symbols the daily LLM USD budget cut off before their pipeline
    # turn (never evaluated this cycle) — the orchestrator journals these
    # as deferred so they're re-offered, once, on the next trading day's
    # cycle ahead of fresh scan results.
    deferred_symbols: list[str] = field(default_factory=list)


class Strategy(Protocol):
    def propose_orders(
        self,
        *,
        candidates: list[Candidate],
        pipeline: PipelineAdapter,
        current_equity: Decimal,
        asof_date: date,
        live_max_position_cap: Decimal | None = None,
    ) -> ProposeOrdersResult: ...
