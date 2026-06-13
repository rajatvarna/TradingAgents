"""Monster Stock scoring engine — TraderLion / Boik framework."""

from .monster_stock_scorer import (
    CriterionScore,
    MonsterStockScore,
    score_stock,
    score_eps_growth,
    score_acceleration,
    score_sponsorship,
    score_rsnhbp,
    score_adr,
)
from .base_auditor import BaseAuditResult, audit_base_health
from .entry_gate import is_buyable

__all__ = [
    "MonsterStockScore",
    "CriterionScore",
    "score_stock",
    "score_eps_growth",
    "score_acceleration",
    "score_sponsorship",
    "score_rsnhbp",
    "score_adr",
    "BaseAuditResult",
    "audit_base_health",
    "is_buyable",
]
