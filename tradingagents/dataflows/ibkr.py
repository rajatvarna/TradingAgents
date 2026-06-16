"""Interactive Brokers (IBKR) vendor implementation.

IBKR requires **Trader Workstation (TWS)** or **IB Gateway** running locally
with API access enabled. The Python SDK (``ib_insync``) talks to TWS/Gateway
over a local TCP socket.

Default ports:
  - TWS paper trading:    7497
  - TWS live trading:     7496
  - IB Gateway paper:     4002
  - IB Gateway live:      4001

To activate: start TWS or IB Gateway with API enabled, install ``ib_insync``,
set ``ibkr_host``, ``ibkr_port``, and ``ibkr_client_id`` in config (defaults
provided), and add ``"ibkr"`` to the relevant vendor list in config
(e.g. ``"core_stock_apis": "ibkr, yfinance"``).

Config keys (all optional — defaults shown):
  ibkr_host       = "127.0.0.1"
  ibkr_port       = 7497
  ibkr_client_id  = 10
"""

import contextlib
from collections.abc import Generator
from datetime import date, datetime

from .config import get_config
from .errors import DataVendorError


@contextlib.contextmanager
def _ctx() -> Generator:
    """Context manager that yields a connected ``ib_insync.IB`` instance.

    Raises ``DataVendorError`` if ``ib_insync`` is not installed or if the
    connection to TWS/Gateway cannot be established.
    """
    try:
        import ib_insync  # noqa: F401 — presence check
        from ib_insync import IB
    except ImportError as exc:
        raise DataVendorError(
            "ib_insync is not installed. Run: pip install ib_insync"
        ) from exc

    cfg = get_config()
    host = cfg.get("ibkr_host", "127.0.0.1")
    port = int(cfg.get("ibkr_port", 7497))
    client_id = int(cfg.get("ibkr_client_id", 10))

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
    except Exception as exc:
        raise DataVendorError(
            f"Cannot connect to IBKR TWS/Gateway at {host}:{port} — {exc}"
        ) from exc

    try:
        yield ib
    finally:
        with contextlib.suppress(Exception):
            ib.disconnect()


def _parse_date(value: str) -> date:
    """Parse an ISO date string (YYYY-MM-DD) or return today on failure."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return date.today()


def _duration_str(start: date, end: date) -> str:
    """Return an IBKR durationStr for the given inclusive date range.

    IBKR caps single-request history at certain limits; for simplicity we
    express the duration in calendar days ('D') up to 365, then switch to
    years ('Y').
    """
    days = max(1, (end - start).days + 1)
    if days <= 365:
        return f"{days} D"
    years = max(1, round(days / 365))
    return f"{years} Y"


def _md_table(headers: list, rows: list) -> str:
    """Render a simple Markdown table from a list of headers and row tuples."""
    sep = " | ".join("---" for _ in headers)
    header_row = " | ".join(headers)
    lines = [f"| {header_row} |", f"| {sep} |"]
    for row in rows:
        cells = " | ".join(str(c) for c in row)
        lines.append(f"| {cells} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Return daily OHLCV history for *symbol* as a Markdown table.

    Parameters
    ----------
    symbol:     Ticker symbol (e.g. ``"AAPL"``).
    start_date: Inclusive start date as ``"YYYY-MM-DD"``.
    end_date:   Inclusive end date as ``"YYYY-MM-DD"``.
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    duration = _duration_str(start, end)

    try:
        with _ctx() as ib:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end.strftime("%Y%m%d %H:%M:%S"),
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
    except DataVendorError:
        raise
    except Exception as exc:
        raise DataVendorError(f"IBKR get_stock_data RPC failed: {exc}") from exc

    if not bars:
        return (
            f"NOTICE: no data returned by IBKR for {symbol} "
            f"({start_date} → {end_date}). "
            "TWS may be offline or the symbol may be unrecognised."
        )

    headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = [
        (
            bar.date,
            f"{bar.open:.4f}",
            f"{bar.high:.4f}",
            f"{bar.low:.4f}",
            f"{bar.close:.4f}",
            f"{bar.volume:,.0f}",
        )
        for bar in bars
    ]

    table = _md_table(headers, rows)
    return (
        f"# {symbol} daily OHLCV ({start_date} → {end_date})\n\n"
        f"{table}\n"
    )


def get_options_chain(symbol: str, expiration: str = "") -> str:
    """Return an options chain for *symbol* as Markdown.

    Fetches option parameters via ``reqSecDefOptParams``, selects the
    expiration closest to *expiration* (nearest available if blank), then
    requests live snapshots for the ±20 strikes nearest the current price.

    Parameters
    ----------
    symbol:     Ticker symbol (e.g. ``"AAPL"``).
    expiration: Target expiration as ``"YYYYMMDD"`` or ``""`` for nearest.
    """
    try:
        with _ctx() as ib:
            from ib_insync import Option, Stock

            # ------------------------------------------------------------------ #
            # 1. Qualify the underlying to get its conId
            # ------------------------------------------------------------------ #
            underlying = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(underlying)
            if not qualified:
                return f"NOTICE: could not qualify {symbol} as a US stock on IBKR."
            con_id = qualified[0].conId

            # ------------------------------------------------------------------ #
            # 2. Fetch option parameter sets (exchanges, expirations, strikes)
            # ------------------------------------------------------------------ #
            opt_params = ib.reqSecDefOptParams(
                underlyingSymbol=symbol,
                futFopExchange="",
                underlyingSecType="STK",
                underlyingConId=con_id,
            )
            if not opt_params:
                return f"NOTICE: no option parameters found for {symbol} on IBKR."

            # Prefer SMART or the first exchange returned
            smart_params = [p for p in opt_params if p.exchange == "SMART"]
            params = smart_params[0] if smart_params else opt_params[0]

            expirations: list = sorted(params.expirations)
            strikes: list = sorted(params.strikes)

            if not expirations or not strikes:
                return f"NOTICE: IBKR returned empty expirations or strikes for {symbol}."

            # ------------------------------------------------------------------ #
            # 3. Choose expiration
            # ------------------------------------------------------------------ #
            if expiration and expiration in expirations:
                chosen_exp = expiration
            else:
                # Pick nearest expiration on or after today
                today_str = date.today().strftime("%Y%m%d")
                future_exps = [e for e in expirations if e >= today_str]
                chosen_exp = future_exps[0] if future_exps else expirations[0]

            # ------------------------------------------------------------------ #
            # 4. Get current price to narrow strike selection
            # ------------------------------------------------------------------ #
            ticker = ib.reqMktData(qualified[0], snapshot=True)
            ib.sleep(2)  # allow snapshot to populate
            mid_price = ticker.last or ticker.close or ticker.bid or 0.0

            if mid_price and strikes:
                # Keep ±20 strikes closest to the current price
                strikes_sorted = sorted(strikes, key=lambda s: abs(s - mid_price))
                selected_strikes = sorted(strikes_sorted[:40])
            else:
                selected_strikes = strikes[:40]

            # ------------------------------------------------------------------ #
            # 5. Request snapshots for each strike / right combination
            # ------------------------------------------------------------------ #
            call_rows: list = []
            put_rows: list = []

            for strike in selected_strikes:
                for right, bucket in (("C", call_rows), ("P", put_rows)):
                    opt = Option(symbol, chosen_exp, strike, right, "SMART")
                    try:
                        qualified_opts = ib.qualifyContracts(opt)
                    except Exception:
                        qualified_opts = []
                    if not qualified_opts:
                        continue
                    opt_ticker = ib.reqMktData(qualified_opts[0], genericTickList="", snapshot=True)
                    ib.sleep(0.1)
                    bucket.append(
                        (
                            strike,
                            _fmt_float(opt_ticker.last),
                            _fmt_float(opt_ticker.bid),
                            _fmt_float(opt_ticker.ask),
                            _fmt_int(opt_ticker.volume),
                            _fmt_int(getattr(opt_ticker, "openInterest", None)),
                            _fmt_float(getattr(opt_ticker, "impliedVol", None)),
                        )
                    )
    except DataVendorError:
        raise
    except Exception as exc:
        raise DataVendorError(f"IBKR get_options_chain RPC failed: {exc}") from exc

    headers = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
    exps_display = expirations[:8]
    exps_suffix = " ..." if len(expirations) > 8 else ""

    return (
        f"# Options chain for {symbol} — expiry {chosen_exp}\n"
        f"Available expirations: {', '.join(exps_display)}{exps_suffix}\n\n"
        f"## Calls\n{_md_table(headers, call_rows)}\n\n"
        f"## Puts\n{_md_table(headers, put_rows)}\n"
    )


def get_options_overview(symbol: str) -> str:
    """Return aggregated put/call statistics for *symbol* as Markdown.

    Reuses ``get_options_chain`` for the nearest expiration and aggregates
    open interest and implied volatility, matching the shape returned by
    ``yfinance_options.get_options_overview``.

    Parameters
    ----------
    symbol: Ticker symbol (e.g. ``"AAPL"``).
    """
    try:
        with _ctx() as ib:
            from ib_insync import Option, Stock

            underlying = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(underlying)
            if not qualified:
                return f"NOTICE: could not qualify {symbol} as a US stock on IBKR."
            con_id = qualified[0].conId

            opt_params = ib.reqSecDefOptParams(
                underlyingSymbol=symbol,
                futFopExchange="",
                underlyingSecType="STK",
                underlyingConId=con_id,
            )
            if not opt_params:
                return f"NOTICE: no option parameters found for {symbol} on IBKR."

            smart_params = [p for p in opt_params if p.exchange == "SMART"]
            params = smart_params[0] if smart_params else opt_params[0]
            expirations: list = sorted(params.expirations)
            strikes: list = sorted(params.strikes)

            if not expirations or not strikes:
                return f"NOTICE: IBKR returned empty expirations or strikes for {symbol}."

            today_str = date.today().strftime("%Y%m%d")
            future_exps = [e for e in expirations if e >= today_str]
            nearest_exp = future_exps[0] if future_exps else expirations[0]
            furthest_exp = expirations[-1]

            # Limit to a central slice of strikes for the overview
            ticker = ib.reqMktData(qualified[0], snapshot=True)
            ib.sleep(2)
            mid_price = ticker.last or ticker.close or ticker.bid or 0.0

            if mid_price:
                selected_strikes = sorted(
                    strikes, key=lambda s: abs(s - mid_price)
                )[:20]
                selected_strikes = sorted(selected_strikes)
            else:
                selected_strikes = strikes[:20]

            call_oi_total = 0.0
            put_oi_total = 0.0
            iv_samples: list = []

            for strike in selected_strikes:
                for right in ("C", "P"):
                    opt = Option(symbol, nearest_exp, strike, right, "SMART")
                    try:
                        q_opts = ib.qualifyContracts(opt)
                    except Exception:
                        q_opts = []
                    if not q_opts:
                        continue
                    opt_ticker = ib.reqMktData(q_opts[0], genericTickList="", snapshot=True)
                    ib.sleep(0.1)
                    oi = getattr(opt_ticker, "openInterest", None) or 0
                    iv = getattr(opt_ticker, "impliedVol", None)
                    if right == "C":
                        call_oi_total += float(oi)
                    else:
                        put_oi_total += float(oi)
                    if iv and iv > 0:
                        iv_samples.append(float(iv))
    except DataVendorError:
        raise
    except Exception as exc:
        raise DataVendorError(f"IBKR get_options_overview RPC failed: {exc}") from exc

    pcr = (put_oi_total / call_oi_total) if call_oi_total else float("nan")
    median_iv = _median(iv_samples) if iv_samples else float("nan")
    iv_display = f"{median_iv:.1%}" if median_iv == median_iv else "N/A"  # nan check
    pcr_display = f"{pcr:.2f}" if pcr == pcr else "N/A"

    return (
        f"# Derivatives overview for {symbol}\n"
        f"- Expirations available: {len(expirations)}"
        f" (nearest {nearest_exp}, furthest {furthest_exp})\n"
        f"- Nearest-expiry call OI: {call_oi_total:,.0f}"
        f" | put OI: {put_oi_total:,.0f}\n"
        f"- Put/Call OI ratio: {pcr_display}\n"
        f"- Median implied volatility (nearest expiry): {iv_display}\n"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_float(value: object) -> str:
    """Format a float for table output; return 'N/A' for None/nan."""
    if value is None:
        return "N/A"
    try:
        f = float(str(value))
        if f != f:  # nan check
            return "N/A"
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_int(value: object) -> str:
    """Format an integer for table output; return 'N/A' for None/nan."""
    if value is None:
        return "N/A"
    try:
        f = float(str(value))
        if f != f:  # nan check
            return "N/A"
        return f"{int(f):,}"
    except (TypeError, ValueError):
        return "N/A"


def _median(values: list) -> float:
    """Return the median of a non-empty list of floats."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
