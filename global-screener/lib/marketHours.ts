import type { Market, MarketState } from "@/types";

function getLocalTime(tz: string): { day: number; timeMin: number } {
  const now = new Date();
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    weekday: "short",
    hour: "numeric",
    minute: "numeric",
    hour12: false,
  });
  const parts = Object.fromEntries(fmt.formatToParts(now).map((p) => [p.type, p.value]));
  const dayMap: Record<string, number> = { Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 };
  const day = dayMap[parts.weekday] ?? 0;
  const h = parseInt(parts.hour, 10);
  const timeMin = h * 60 + parseInt(parts.minute, 10);
  return { day, timeMin };
}

/**
 * Returns the current trading state for a market. Does not account for public holidays.
 */
export function getMarketState(market: Market): MarketState {
  switch (market) {
    case "US": {
      const { day, timeMin } = getLocalTime("America/New_York");
      if (day === 0 || day === 6) return "closed";
      if (timeMin >= 9 * 60 + 30 && timeMin < 16 * 60) return "open";
      if (timeMin >= 8 * 60 && timeMin < 9 * 60 + 30) return "pre-market";
      return "closed";
    }
    case "India": {
      const { day, timeMin } = getLocalTime("Asia/Kolkata");
      if (day === 0 || day === 6) return "closed";
      if (timeMin >= 9 * 60 + 15 && timeMin < 15 * 60 + 30) return "open";
      if (timeMin >= 9 * 60 && timeMin < 9 * 60 + 15) return "pre-market";
      return "closed";
    }
    case "UAE": {
      const { day, timeMin } = getLocalTime("Asia/Dubai");
      if (day === 0 || day === 6) return "closed";
      if (timeMin >= 10 * 60 && timeMin < 14 * 60 + 50) return "open";
      if (timeMin >= 9 * 60 + 30 && timeMin < 10 * 60) return "pre-market";
      return "closed";
    }
    case "Saudi": {
      const { day, timeMin } = getLocalTime("Asia/Riyadh");
      // Sun–Thu workweek; Fri(5) and Sat(6) are weekend
      if (day === 5 || day === 6) return "closed";
      if (timeMin >= 10 * 60 && timeMin < 15 * 60) return "open";
      if (timeMin >= 9 * 60 + 30 && timeMin < 10 * 60) return "pre-market";
      return "closed";
    }
    default:
      return "closed";
  }
}

export function isMarketOpen(market: string): boolean {
  if (!["US", "India", "UAE", "Saudi"].includes(market)) return false;
  return getMarketState(market as Market) === "open";
}
