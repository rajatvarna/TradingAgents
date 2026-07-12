"""Futu OpenD vendor implementation.

Futu requires the **OpenD** gateway daemon running locally with a logged-in
Futu account; the Python SDK talks to OpenD over a local TCP socket
(default 127.0.0.1:11111). Treat Futu strictly as a fallback vendor —
never block a run on its availability.

To activate: start OpenD, install futu-api, set ``futu_opend_host`` and
``futu_opend_port`` in DEFAULT_CONFIG (defaults provided), and append
``"futu"`` to a vendor list (e.g. ``"core_stock_apis": "yfinance, futu"``).

Futu symbols are market-prefixed (e.g. "HK.00700", "US.AAPL", "SH.600000",
"SZ.000001") — ``_to_futu_symbol`` translates the framework's Yahoo-style
tickers (see ``symbol_utils.normalize_symbol``) before calling OpenD.

NOTE: this module has not been exercised against a live OpenD daemon (no
Futu account/gateway available in this environment) — request/response
shapes follow the documented ``futu-api`` surface, but treat it as
unverified until run against a real OpenD instance.
"""

from __future__ import annotations

import contextlib
import re

from .config import get_config
from .errors import DataVendorError
from .symbol_utils import crypto_base

_HK_RE = re.compile(r"^(\d{1,5})\.HK$", re.IGNORECASE)
_SH_RE = re.compile(r"^(\d{6})\.SS$", re.IGNORECASE)
_SZ_RE = re.compile(r"^(\d{6})\.SZ$", re.IGNORECASE)
_US_RE = re.compile(r"^[A-Z][A-Z\-]{0,9}$")

# Yahoo exchange suffixes for markets Futu does not serve under the bare
# US.<ticker> convention (Toronto, London, ASX, Paris, Xetra, Tokyo, ...).
# A bare US ticker never contains a dot, so any dotted suffix not already
# matched above (HK/SS/SZ) is an unsupported non-US market — reject it
# rather than let it fall through to the US.<ticker> guess.
_NON_US_SUFFIX_RE = re.compile(r"^[A-Z0-9]+\.[A-Z]{1,3}$")

# Yahoo suffixes/markers that indicate a non-equity instrument (forex, futures,
# indices) which Futu's US.<ticker> convention does not cover — never guess.
_NON_EQUITY_MARKERS = ("=X", "=F", "^")
_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
    }
)


def _to_futu_symbol(symbol: str) -> str:
    """Translate a Yahoo-style ticker to Futu's market-prefixed format.

    Examples: ``0700.HK`` -> ``HK.00700``, ``600000.SS`` -> ``SH.600000``,
    ``000001.SZ`` -> ``SZ.000001``, ``AAPL`` -> ``US.AAPL``.
    """
    s = symbol.strip().upper()

    m = _HK_RE.match(s)
    if m:
        return f"HK.{m.group(1).zfill(5)}"

    m = _SH_RE.match(s)
    if m:
        return f"SH.{m.group(1)}"

    m = _SZ_RE.match(s)
    if m:
        return f"SZ.{m.group(1)}"

    if any(marker in s for marker in _NON_EQUITY_MARKERS) or crypto_base(s) is not None:
        raise DataVendorError(f"futu: {symbol!r} is not a supported equity symbol")
    if len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        raise DataVendorError(f"futu: {symbol!r} looks like a forex pair, not an equity")
    if _NON_US_SUFFIX_RE.match(s):
        raise DataVendorError(
            f"futu: {symbol!r} carries an unsupported non-US/HK/CN exchange suffix"
        )

    if _US_RE.match(s):
        return f"US.{s}"

    raise DataVendorError(f"futu: cannot translate symbol {symbol!r} to a Futu market code")


def _ctx():
    try:
        from futu import OpenQuoteContext
    except ImportError as e:
        raise DataVendorError("futu-api not installed") from e
    cfg = get_config()
    try:
        return OpenQuoteContext(
            host=cfg.get("futu_opend_host", "127.0.0.1"),
            port=int(cfg.get("futu_opend_port", 11111)),
        )
    except Exception as e:
        raise DataVendorError(f"Cannot reach OpenD: {e}") from e


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Daily klines via ``ctx.request_history_kline(...)``, formatted as a CSV
    table matching ``get_YFin_data_online``'s output shape."""
    import pandas as pd
    from futu import RET_OK, AuType, KLType

    futu_symbol = _to_futu_symbol(symbol)
    ctx = _ctx()
    try:
        pages = []
        page_key = None
        while True:
            ret, data, page_key = ctx.request_history_kline(
                futu_symbol,
                start=start_date,
                end=end_date,
                ktype=KLType.K_DAY,
                autype=AuType.QFQ,
                max_count=1000,
                page_req_key=page_key,
            )
            if ret != RET_OK:
                raise DataVendorError(f"futu.get_stock_data({futu_symbol}): {data}")
            if data is not None and not data.empty:
                pages.append(data)
            if page_key is None:
                break

        if not pages:
            raise DataVendorError(f"futu.get_stock_data({futu_symbol}): no rows returned")
        data = pd.concat(pages, ignore_index=True)

        cols = ["time_key", "open", "high", "low", "close", "volume"]
        keep = [c for c in cols if c in data.columns]
        table = data[keep].rename(columns={"time_key": "Date"})
        csv_string = table.to_csv(index=False)
        return (
            f"## Daily klines for {futu_symbol} (from {symbol}), "
            f"{start_date} to {end_date}:\n\n{csv_string}"
        )
    finally:
        with contextlib.suppress(Exception):
            ctx.close()


def get_options_chain(symbol: str, expiration: str = "") -> str:
    """Options chain via ``ctx.get_option_expiration_date`` + ``ctx.get_option_chain``,
    with IV/greeks filled in from ``ctx.get_market_snapshot``. Formatted to match
    ``yfinance_options.get_options_chain``."""
    from futu import RET_OK, OptionCondType, OptionType

    futu_symbol = _to_futu_symbol(symbol)
    ctx = _ctx()
    try:
        ret, exp_data = ctx.get_option_expiration_date(futu_symbol)
        if ret != RET_OK:
            raise DataVendorError(f"futu.get_options_chain({futu_symbol}): {exp_data}")
        if exp_data is None or exp_data.empty:
            return f"No listed options found for {futu_symbol} (from {symbol})."

        exps = list(exp_data["strike_time"])
        exp = expiration if expiration in exps else exps[0]

        ret, chain = ctx.get_option_chain(
            futu_symbol,
            start=exp,
            end=exp,
            option_type=OptionType.ALL,
            option_cond_type=OptionCondType.ALL,
        )
        if ret != RET_OK:
            raise DataVendorError(f"futu.get_options_chain({futu_symbol}): {chain}")
        if chain is None or chain.empty:
            return f"No options contracts found for {futu_symbol} expiry {exp}."

        codes = list(chain["code"])
        iv_by_code: dict[str, float] = {}
        if codes:
            ret, snapshot = ctx.get_market_snapshot(codes)
            if ret == RET_OK and snapshot is not None and not snapshot.empty:
                for _, row in snapshot.iterrows():
                    iv = row.get("option_implied_volatility")
                    if iv is not None:
                        iv_by_code[row["code"]] = iv

        chain = chain.copy()
        chain["implied_volatility"] = chain["code"].map(iv_by_code)

        calls = chain[chain["option_type"] == OptionType.CALL].sort_values("strike_price")
        puts = chain[chain["option_type"] == OptionType.PUT].sort_values("strike_price")

        cols = ["strike_price", "lot_size", "implied_volatility"]

        def _fmt(df, n: int = 12) -> str:
            keep = [c for c in cols if c in df.columns]
            if len(df) <= n:
                return df[keep].to_markdown(index=False)
            # Keep the strikes nearest the median (proxy for ATM) rather than
            # naively taking the lowest-strike rows after an ascending sort.
            centered = df.iloc[(df["strike_price"] - df["strike_price"].median()).abs().argsort()[:n]]
            centered = centered.sort_values("strike_price")
            return centered[keep].to_markdown(index=False) + f"\n\n_(showing {n} of {len(df)} strikes)_"

        return (
            f"# Options chain for {futu_symbol} (from {symbol}) — expiry {exp}\n"
            f"Available expirations: {', '.join(exps[:8])}{' ...' if len(exps) > 8 else ''}\n\n"
            f"## Calls\n{_fmt(calls)}\n\n"
            f"## Puts\n{_fmt(puts)}\n"
        )
    finally:
        with contextlib.suppress(Exception):
            ctx.close()
