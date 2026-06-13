import watchlistData from "@/data/watchlist.json";
import { Market, TickerMeta, Watchlist } from "@/types";

const watchlist = watchlistData as Watchlist;

export function getTickers(market: Market | "All"): TickerMeta[] {
  if (market === "All") {
    return (["US", "India", "UAE", "Saudi"] as Market[]).flatMap(
      (m) => watchlist[m].tickers.map((t) => ({ ...t, market: m }))
    );
  }
  return watchlist[market].tickers.map((t) => ({ ...t, market }));
}

export function getAllSectors(): string[] {
  const sectors = new Set<string>();
  for (const market of ["US", "India", "UAE", "Saudi"] as Market[]) {
    for (const ticker of watchlist[market].tickers) {
      sectors.add(ticker.sector);
    }
  }
  return Array.from(sectors).sort();
}
