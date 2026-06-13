"""Monster Stock scoring engine — TraderLion / Boik framework."""

from .base_auditor import BaseAuditResult, audit_base_health
from .entry_gate import is_buyable
from .monster_stock_scorer import (
    CriterionScore,
    MonsterStockScore,
    score_acceleration,
    score_adr,
    score_eps_growth,
    score_rsnhbp,
    score_sponsorship,
    score_stock,
)

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
