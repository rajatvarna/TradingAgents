import type { Market, MarketState } from "@/types";

/**
 * Returns the current trading state for a market (approximate, UTC-based).
 * Does not account for public holidays.
 */
export function getMarketState(market: Market): MarketState {
  const now = new Date();
  const day = now.getUTCDay(); // 0=Sun … 6=Sat
  const timeMin = now.getUTCHours() * 60 + now.getUTCMinutes();

  switch (market) {
    case "US":
      if (day === 0 || day === 6) return "closed";
      if (timeMin >= 13 * 60 + 30 && timeMin < 21 * 60) return "open";
      if (timeMin >= 12 * 60 && timeMin < 13 * 60 + 30) return "pre-market";
      return "closed";
    case "India":
      if (day === 0 || day === 6) return "closed";
      if (timeMin >= 3 * 60 + 45 && timeMin < 10 * 60) return "open";
      if (timeMin >= 3 * 60 + 15 && timeMin < 3 * 60 + 45) return "pre-market";
      return "closed";
    case "UAE":
      if (day === 0 || day === 6) return "closed";
      if (day === 5 && timeMin >= 6 * 60 && timeMin < 9 * 60 + 50) return "open";
      if (day !== 5 && timeMin >= 6 * 60 && timeMin < 10 * 60 + 50) return "open";
      return "closed";
    case "Saudi":
      // Sun–Thu workweek; Fri(5) and Sat(6) are weekend
      if (day === 5 || day === 6) return "closed";
      if (timeMin >= 7 * 60 && timeMin < 12 * 60) return "open";
      if (timeMin >= 6 * 60 + 30 && timeMin < 7 * 60) return "pre-market";
      return "closed";
    default:
      return "closed";
  }
}

/**
 * Returns true if the given market is currently open (approximate, UTC-based).
 * Times are in UTC. This is a best-effort check — does not account for public holidays.
 */
export function isMarketOpen(market: string): boolean {
  const now = new Date();
  const day = now.getUTCDay();
  const h = now.getUTCHours();
  const m = now.getUTCMinutes();
  const timeMin = h * 60 + m;

  if (day === 0 || day === 6) return false;

  switch (market) {
    case "US":
      return timeMin >= 13 * 60 + 30 && timeMin < 21 * 60;
    case "India":
      return timeMin >= 3 * 60 + 45 && timeMin < 10 * 60;
    case "UAE":
      if (day === 5) return timeMin >= 6 * 60 && timeMin < 9 * 60 + 50;
      return timeMin >= 6 * 60 && timeMin < 10 * 60 + 50;
    case "Saudi":
      if (day === 5 || day === 0) return false;
      return timeMin >= 7 * 60 && timeMin < 12 * 60;
    default:
      return false;
  }
}
