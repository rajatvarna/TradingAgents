import watchlistData from "@/data/watchlist.json";
import { Market, TickerMeta, Watchlist } from "@/types";

const wl = watchlistData as Watchlist;

/** Returns all tickers for the given market(s), or all markets when "All" is requested. */
export function getTickers(market: Market | "All"): (TickerMeta & { market: Market })[] {
  const markets: Market[] = market === "All" ? ["US", "India", "UAE", "Saudi"] : [market];
  return markets.flatMap((m) =>
    (wl[m]?.tickers ?? []).map((t) => ({ ...t, market: m }))
  );
}

/** Returns all unique sector names across all markets. */
export function getAllSectors(): string[] {
  const sectors = new Set<string>();
  (["US", "India", "UAE", "Saudi"] as Market[]).forEach((m) => {
    wl[m]?.tickers.forEach((t) => sectors.add(t.sector));
  });
  return [...sectors].sort();
}
