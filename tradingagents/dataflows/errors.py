"""Vendor data-error taxonomy.

A single hierarchy so the routing layer reacts by *behavior*, not by vendor:
every condition where a vendor cannot return usable data derives from
``VendorError``, and the router catches the base types. A new vendor raises
these (or a thin vendor-named subclass) and needs no new ``except`` clause.

    VendorError
    ├── DataVendorError            fork custom exception for vendor fallback
    ├── NoMarketDataError          no usable rows (empty result OR stale data)
    ├── VendorRateLimitError       transient throttle -> skip to next vendor
    ├── VendorNotConfiguredError   missing API key/config -> vendor unavailable
    └── VendorCapabilityError      vendor can't serve this request shape -> skip to next vendor
"""

from __future__ import annotations


class VendorError(Exception):
    """Base for any condition where a vendor could not return usable data."""


class DataVendorError(VendorError):
    """Raised by a vendor implementation when it cannot serve a request
    (auth/config missing, network error, daemon down). Signals route_to_vendor
    to try the next vendor in the fallback chain instead of failing the run."""


class NoMarketDataError(VendorError):
    """A vendor returned no usable rows for a symbol (empty result or stale data).

    Carries both the symbol the user requested and the canonical symbol the
    vendor was actually queried with, plus a free-text ``detail``, so callers
    can build a clear message instead of emitting a vendor-specific empty
    string into the data channel.
    """

    def __init__(self, symbol: str, canonical: str | None = None, detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class VendorRateLimitError(VendorError):
    """A vendor throttled the request; the router skips to the next vendor."""


class VendorNotConfiguredError(VendorError, ValueError):
    """A vendor was selected but its API key/configuration is missing.

    Also a ``ValueError`` so existing callers that catch ``ValueError`` keep
    working while the routing layer can treat it as "vendor unavailable".
    """


class VendorCapabilityError(VendorError):
    """A vendor was selected but cannot serve this specific request shape
    (e.g. an intraday ``interval`` it has no endpoint for).

    Distinct from ``VendorNotConfiguredError`` (missing credentials) and from
    the transient failures ``CircuitBreaker`` exists to protect against — a
    capability gap is permanent for this request shape, not a flaky vendor,
    so raising it does not trip the circuit breaker. The router treats it the
    same as "vendor unavailable": skip to the next vendor in the chain.
    """
