"""
API interface for querying and analyzing stock holdings.
Provides the HoldingsAPI class and get_api_instance helper.
"""

import json
import os


class HoldingsAPI:
    """Read-only API layer for stock portfolio statistics and analysis."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}
        self.load_data()

    def load_data(self):
        """Load data from the JSON file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"holdings": [], "portfolio_summary": {}}
        else:
            self.data = {"holdings": [], "portfolio_summary": {}}

        if "holdings" not in self.data:
            self.data["holdings"] = []
        if "portfolio_summary" not in self.data:
            self.data["portfolio_summary"] = {}

    def get_all_holdings(self) -> list[dict]:
        """Get list of all stock holdings."""
        return self.data.get("holdings", [])

    def get_holding_by_symbol(self, symbol) -> dict | None:
        """Find a specific holding by its ticker symbol."""
        for h in self.data.get("holdings", []):
            if h.get("symbol", "").upper() == symbol.upper():
                return h
        return None

    def get_holdings_by_sector(self, sector) -> list[dict]:
        """Filter holdings by their economic sector."""
        result = []
        for h in self.data.get("holdings", []):
            if h.get("sector", "").strip().lower() == sector.strip().lower():
                result.append(h)
        return result

    def get_portfolio_summary(self) -> dict:
        """Get overall portfolio statistics."""
        return self.data.get("portfolio_summary", {})

    def calculate_individual_return(self, symbol) -> dict | None:
        """Calculate investment returns for a specific stock holding."""
        holding = self.get_holding_by_symbol(symbol)
        if not holding:
            return None

        qty = holding.get("quantity", 0)
        purch = holding.get("purchase_price", 0.0)
        curr = holding.get("current_price", 0.0)

        total_invested = qty * purch
        total_current_value = qty * curr
        gain_loss = total_current_value - total_invested
        return_percent = (gain_loss / total_invested * 100.0) if total_invested > 0.0 else 0.0

        return {
            "symbol": holding.get("symbol"),
            "total_invested": total_invested,
            "total_current_value": total_current_value,
            "gain_loss": gain_loss,
            "return_percent": return_percent,
        }

    def get_top_performers(self, limit=5) -> list[dict]:
        """Get holdings sorted by return percentage descending, up to limit."""
        holdings = self.get_all_holdings()
        decorated = []
        for h in holdings:
            h_copy = h.copy()
            purch = h_copy.get("purchase_price", 0.0)
            curr = h_copy.get("current_price", 0.0)
            ret = ((curr - purch) / purch * 100.0) if purch > 0.0 else 0.0
            h_copy["return_percent"] = ret
            decorated.append(h_copy)

        decorated.sort(key=lambda x: x["return_percent"], reverse=True)
        return decorated[:limit]

    def get_sector_distribution(self) -> dict[str, float]:
        """Get sector allocation as percentages of total portfolio current value."""
        holdings = self.get_all_holdings()
        if not holdings:
            return {}

        total_value = sum(h.get("quantity", 0) * h.get("current_price", 0.0) for h in holdings)
        if total_value <= 0.0:
            return {}

        sector_totals = {}
        for h in holdings:
            sector = h.get("sector", "Unknown")
            value = h.get("quantity", 0) * h.get("current_price", 0.0)
            sector_totals[sector] = sector_totals.get(sector, 0.0) + value

        return {
            sector: (val / total_value * 100.0)
            for sector, val in sector_totals.items()
        }

    def get_stats(self) -> dict:
        """Get comprehensive portfolio statistics."""
        holdings = self.get_all_holdings()
        total_quantity = sum(h.get("quantity", 0) for h in holdings)

        return {
            "total_holdings": len(holdings),
            "total_quantity": total_quantity,
            "portfolio_summary": self.get_portfolio_summary(),
            "sector_distribution": self.get_sector_distribution(),
            "top_performers": self.get_top_performers(),
        }


def get_api_instance(filepath=None) -> HoldingsAPI:
    """Helper factory function to get a HoldingsAPI instance."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "holdings.json")
    return HoldingsAPI(filepath)
