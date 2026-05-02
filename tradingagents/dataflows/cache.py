import yfinance as yf
from threading import Lock

class TickerCache:
    """Thread-safe in-memory cache for yfinance Ticker objects."""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TickerCache, cls).__new__(cls)
                cls._instance._tickers = {}
        return cls._instance

    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create a cached Ticker object for the given symbol."""
        key = symbol.upper()
        with self._lock:
            if key not in self._tickers:
                self._tickers[key] = yf.Ticker(key)
            return self._tickers[key]

    def clear(self):
        """Clear the cache. Useful between independent runs."""
        with self._lock:
            self._tickers.clear()

# Global instance
ticker_cache = TickerCache()
