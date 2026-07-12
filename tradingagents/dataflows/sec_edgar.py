"""SEC EDGAR full-text filing search vendor.

Wraps the SEC's free EDGAR full-text search API (efts.sec.gov) so the
fundamentals/news analysts can cite raw filing text and metadata (form
type, filing date, accession number) instead of relying solely on
pre-aggregated fundamentals. No API key is required, but the SEC requires
a descriptive User-Agent header identifying the requester on all requests.
"""
import logging
import os

import requests

logger = logging.getLogger(__name__)

EDGAR_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# Network timeout (seconds), mirroring the other free-tier vendors in this
# package so a stalled request can't hang the agents.
REQUEST_TIMEOUT = 30

# Rows cap for the rendered table: filings are sorted most-recent-first, and
# a long lookback can return far more hits than are useful in a report.
MAX_RESULTS = 20

# SEC EDGAR requires a descriptive User-Agent identifying the requester
# (https://www.sec.gov/os/accessing-edgar-data). Configurable via env var
# so operators can put their own contact info in it, with a generic
# fallback so the vendor still works out of the box.
DEFAULT_USER_AGENT = "TradingAgents research tool (contact: set SEC_EDGAR_USER_AGENT)"


def _user_agent() -> str:
    return os.getenv("SEC_EDGAR_USER_AGENT", DEFAULT_USER_AGENT)


def _request(params: dict) -> dict:
    """GET the EDGAR full-text search endpoint, surfacing errors clearly."""
    response = requests.get(
        EDGAR_FULLTEXT_SEARCH_URL,
        params=params,
        headers={"User-Agent": _user_agent()},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise ValueError(
            f"SEC EDGAR full-text search failed ({response.status_code}): {response.text[:500]}"
        )
    return response.json()


def search_filings(
    query: str,
    ticker: str | None = None,
    forms: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """Search SEC EDGAR full-text filing index and return a markdown report.

    Args:
        query: Free-text search query (e.g. "material weakness", "going concern").
        ticker: Optional ticker symbol to restrict results to one company.
        forms: Optional comma-separated form types (e.g. "10-K,10-Q,8-K").
        date_from: Optional start date (yyyy-mm-dd) for the filing date range.
        date_to: Optional end date (yyyy-mm-dd) for the filing date range.

    Returns:
        A markdown report listing matching filings: company, form type,
        filing date, and a link to the filing on sec.gov.
    """
    if not query or not query.strip():
        return "ERROR: query must be a non-empty search string."

    params: dict = {"q": query.strip()}
    if ticker:
        params["ciks"] = ticker.strip().upper()
    if forms:
        params["forms"] = forms.strip()
    if date_from:
        params["dateRange"] = "custom"
        params["startdt"] = date_from
    if date_to:
        params["dateRange"] = "custom"
        params["enddt"] = date_to

    payload = _request(params)
    hits = (payload.get("hits") or {}).get("hits") or []

    header = f"## SEC EDGAR Full-Text Search: \"{query}\"\n"
    if ticker:
        header += f"- Ticker filter: {ticker.upper()}\n"
    if forms:
        header += f"- Form types: {forms}\n"
    if date_from or date_to:
        header += f"- Date range: {date_from or 'earliest'} to {date_to or 'latest'}\n"

    if not hits:
        return header + "\nNo matching filings found."

    shown = hits[:MAX_RESULTS]
    note = ""
    if len(hits) > MAX_RESULTS:
        note = f"\n_(showing the most recent {MAX_RESULTS} of {len(hits)} matches)_\n"

    rows = []
    for hit in shown:
        source = hit.get("_source", {})
        company_names = source.get("display_names") or []
        company = company_names[0] if company_names else "Unknown"
        form_type = source.get("form") or source.get("file_type", "N/A")
        filed_date = source.get("file_date", "N/A")
        accession = (hit.get("_id") or "").split(":")[0]
        cik = (source.get("ciks") or [""])[0].lstrip("0")
        accession_nodash = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession}-index.htm"
            if cik and accession
            else "N/A"
        )
        rows.append(f"| {company} | {form_type} | {filed_date} | {url} |")

    table = (
        "\n| Company | Form | Filed | Link |\n| --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )
    return header + note + table
