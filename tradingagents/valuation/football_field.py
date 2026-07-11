"""
Football-field valuation summary.

Aggregates intrinsic-value estimates from multiple methods (ROIC-DCF,
Revenue-DCF, DDM, financial model, sensitivity range, ...) into a single
low/high range per method plus an overall summary, for the classic
"football field" chart used to present a valuation range rather than a
single point estimate.

Pure math — no external dependencies, no LLM, no I/O. Callers are
responsible for computing each method's value(s) via the other valuation
modules and passing them in here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MethodRange:
    """Low/high intrinsic-value estimate for a single valuation method."""

    method: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            self.low, self.high = self.high, self.low


@dataclass
class FootballField:
    """Aggregated valuation range across multiple methods."""

    ticker: str
    ranges: list[MethodRange]
    current_price: float | None = None

    @property
    def overall_low(self) -> float:
        return min(r.low for r in self.ranges) if self.ranges else 0.0

    @property
    def overall_high(self) -> float:
        return max(r.high for r in self.ranges) if self.ranges else 0.0

    @property
    def overall_midpoint(self) -> float:
        return (self.overall_low + self.overall_high) / 2.0


def build_football_field(
    ticker: str,
    method_values: dict[str, float | tuple[float, float]],
    current_price: float | None = None,
) -> FootballField:
    """Assemble a football-field summary from per-method value(s).

    Args:
        ticker: Stock ticker symbol (for labelling only).
        method_values: Mapping of method name to either a single point
            estimate (treated as low == high) or a ``(low, high)`` tuple
            (e.g. a sensitivity-matrix range or a DCF/reverse-DCF pairing).
            Methods with non-finite or missing values should be omitted by
            the caller before calling this function.
        current_price: Optional current market price, shown alongside the range.

    Returns:
        :class:`FootballField` with one :class:`MethodRange` per method.
    """
    ranges: list[MethodRange] = []
    for method, value in method_values.items():
        if isinstance(value, tuple):
            low, high = value
        else:
            low = high = value
        ranges.append(MethodRange(method=method, low=round(low, 2), high=round(high, 2)))

    return FootballField(ticker=ticker.upper(), ranges=ranges, current_price=current_price)


def format_football_field(field: FootballField) -> str:
    """Render a :class:`FootballField` as a Markdown summary.

    Args:
        field: A populated FootballField instance.

    Returns:
        Multi-line Markdown string listing each method's range plus the
        overall low/high/midpoint and, if available, where the current
        price falls relative to that range.
    """
    if not field.ranges:
        return "(no valuation methods available for football field)"

    lines = [
        f"## Valuation Football Field — {field.ticker}",
        "",
        "| Method | Low | High |",
        "|---|---:|---:|",
    ]
    for r in field.ranges:
        lines.append(f"| {r.method} | ${r.low:,.2f} | ${r.high:,.2f} |")

    lines.extend([
        "",
        f"**Overall Range:** ${field.overall_low:,.2f} – ${field.overall_high:,.2f}  "
        f"(midpoint ${field.overall_midpoint:,.2f})",
    ])

    if field.current_price is not None and field.current_price > 0:
        price = field.current_price
        if price < field.overall_low:
            position = "BELOW all method ranges — potentially undervalued across the board"
        elif price > field.overall_high:
            position = "ABOVE all method ranges — potentially overvalued across the board"
        else:
            position = "WITHIN the overall valuation range"
        lines.append(f"**Current Price:** ${price:,.2f} — {position}")

    return "\n".join(lines)
