#!/usr/bin/env python3
"""
Stock holdings data management script.
Provides HoldingsManager class and automated update functionality via yfinance.
"""

import os
import json
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class HoldingsManager:
    """Manages stock holdings stored in a JSON database."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}
        self.load_holdings()

    def load_holdings(self) -> dict:
        """Load holdings data from JSON file or initialize a new structure."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as exc:
                logger.warning("Could not read holdings file, initializing empty: %s", exc)
                self.data = {"holdings": [], "portfolio_summary": {}}
        else:
            self.data = {"holdings": [], "portfolio_summary": {}}

        # Ensure correct structure
        if "holdings" not in self.data:
            self.data["holdings"] = []
        if "portfolio_summary" not in self.data:
            self.data["portfolio_summary"] = {}

        return self.data

    def save_holdings(self) -> None:
        """Save the current holdings data back to the JSON file."""
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save holdings to %s: %s", self.filepath, exc)

    def calculate_portfolio_summary(self) -> dict:
        """Compute and update aggregate summary metrics for the portfolio."""
        holdings = self.data.get("holdings", [])
        total_holdings = len(holdings)
        total_quantity = sum(h.get("quantity", 0) for h in holdings)
        total_invested = sum(h.get("quantity", 0) * h.get("purchase_price", 0.0) for h in holdings)
        total_current_value = sum(h.get("quantity", 0) * h.get("current_price", 0.0) for h in holdings)
        total_gain_loss = total_current_value - total_invested
        gain_loss_percentage = (total_gain_loss / total_invested * 100.0) if total_invested > 0.0 else 0.0

        summary = {
            "total_holdings": total_holdings,
            "total_quantity": total_quantity,
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current_value, 2),
            "total_gain_loss": round(total_gain_loss, 2),
            "gain_loss_percentage": round(gain_loss_percentage, 2),
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.data["portfolio_summary"] = summary
        self.save_holdings()
        return summary

    def add_holding(
        self,
        symbol,
        company_name,
        quantity,
        purchase_price,
        current_price,
        purchase_date,
        sector,
    ) -> dict:
        """Add a new stock holding to the portfolio."""
        holdings = self.data.get("holdings", [])
        next_id = max([h.get("id", 0) for h in holdings]) + 1 if holdings else 1

        new_holding = {
            "id": next_id,
            "symbol": symbol.upper(),
            "company_name": company_name,
            "quantity": int(quantity),
            "purchase_price": float(purchase_price),
            "current_price": float(current_price),
            "purchase_date": purchase_date,
            "sector": sector,
            "status": "active",
        }
        holdings.append(new_holding)
        self.data["holdings"] = holdings
        self.calculate_portfolio_summary()
        return new_holding

    def update_holding_price(self, symbol, new_price) -> bool:
        """Update the current price of a specific stock holding."""
        holdings = self.data.get("holdings", [])
        found = False
        for h in holdings:
            if h.get("symbol", "").upper() == symbol.upper():
                h["current_price"] = float(new_price)
                found = True
                break

        if found:
            self.calculate_portfolio_summary()
            return True
        return False

    def get_portfolio_summary(self) -> dict:
        """Get the current portfolio summary metrics."""
        return self.data.get("portfolio_summary", {})

    def get_all_holdings(self) -> list:
        """Get all active stock holdings."""
        return self.data.get("holdings", [])

    def get_holding_by_symbol(self, symbol) -> dict | None:
        """Retrieve a specific holding by its ticker symbol."""
        for h in self.data.get("holdings", []):
            if h.get("symbol", "").upper() == symbol.upper():
                return h
        return None

    def print_portfolio_report(self) -> None:
        """Format and print a comprehensive portfolio report to stdout."""
        summary = self.get_portfolio_summary()
        holdings = self.get_all_holdings()

        last_updated = summary.get("last_updated", "N/A")
        total_holdings = summary.get("total_holdings", 0)
        total_quantity = summary.get("total_quantity", 0)
        total_invested = summary.get("total_invested", 0.0)
        total_current_value = summary.get("total_current_value", 0.0)
        total_gain_loss = summary.get("total_gain_loss", 0.0)
        gain_loss_percentage = summary.get("gain_loss_percentage", 0.0)

        sign = "+" if total_gain_loss >= 0 else ""

        print("=" * 80)
        print("STOCK HOLDINGS PORTFOLIO REPORT")
        print("=" * 80)
        print(f"Last Updated: {last_updated}")
        print("-" * 80)
        print(f"Total Holdings: {total_holdings}")
        print(f"Total Quantity: {total_quantity} shares")
        print(f"Total Invested: ${total_invested:,.2f}")
        print(f"Current Value: ${total_current_value:,.2f}")
        print(f"Total Gain/Loss: ${total_gain_loss:,.2f} ({sign}{gain_loss_percentage:.2f}%)")
        print("-" * 80)
        print("")
        print("Detailed Holdings:")
        print(f"{'Symbol':<10}{'Company':<25}{'Qty':<8}{'Purchase':<12}{'Current':<12}{'Gain/Loss':<15}{'Return %'}")
        print("-" * 80)

        for h in holdings:
            sym = h.get("symbol", "")
            company = h.get("company_name", "")
            if len(company) > 23:
                company = company[:20] + "..."
            qty = h.get("quantity", 0)
            purch = h.get("purchase_price", 0.0)
            curr = h.get("current_price", 0.0)
            
            invested = qty * purch
            current_value = qty * curr
            gain = current_value - invested
            ret = (gain / invested * 100.0) if invested > 0.0 else 0.0
            ret_sign = "+" if gain >= 0 else ""

            print(
                f"{sym:<10}"
                f"{company:<25}"
                f"{qty:<8}"
                f"${purch:<11,.2f}"
                f"${curr:<11,.2f}"
                f"${gain:<14,.2f}"
                f"{ret_sign}{ret:.2f}%"
            )
        print("=" * 80)


if __name__ == "__main__":
    import yfinance as yf

    # Load default data/holdings.json path
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "holdings.json",
    )
    
    print(f"Loading portfolio holdings from: {default_path}")
    manager = HoldingsManager(default_path)
    holdings = manager.get_all_holdings()

    if not holdings:
        print("No active stock holdings found in portfolio database.")
    else:
        print("Updating real-time market prices via yfinance...")
        for holding in holdings:
            sym = holding.get("symbol")
            if not sym:
                continue
            try:
                ticker = yf.Ticker(sym)
                # Fetch latest day closing price
                hist = ticker.history(period="1d")
                if not hist.empty:
                    latest_price = float(hist["Close"].iloc[-1])
                    manager.update_holding_price(sym, latest_price)
                    print(f"Updated {sym}: ${latest_price:.2f}")
                else:
                    # Fallback to info regularMarketPrice
                    price = ticker.info.get("regularMarketPrice")
                    if price is not None:
                        manager.update_holding_price(sym, price)
                        print(f"Updated {sym} (info): ${price:.2f}")
                    else:
                        print(f"Could not retrieve price for {sym}, keeping existing.")
            except Exception as e:
                print(f"Error fetching price for {sym}: {e}, keeping existing.")

        manager.calculate_portfolio_summary()
        print("\nUpdate complete. Portfolio Report:")
        manager.print_portfolio_report()
