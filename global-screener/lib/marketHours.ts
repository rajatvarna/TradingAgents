/**
 * Returns true if the given market is currently open (approximate, UTC-based).
 * Times are in UTC. This is a best-effort check — does not account for public holidays.
 */
export function isMarketOpen(market: string): boolean {
  const now = new Date();
  const day = now.getUTCDay(); // 0=Sun, 1=Mon, ..., 5=Fri, 6=Sat
  const h = now.getUTCHours();
  const m = now.getUTCMinutes();
  const timeMin = h * 60 + m;

  // Weekends: no market open
  if (day === 0 || day === 6) return false;

  switch (market) {
    case "US":
      // NYSE/NASDAQ: 9:30–16:00 ET = 14:30–21:00 UTC (EST, Nov-Mar) / 13:30–20:00 UTC (EDT, Mar-Nov)
      // Use a simplified range: 13:30–21:00 UTC to cover both DST states
      return timeMin >= 13 * 60 + 30 && timeMin < 21 * 60;

    case "India":
      // NSE/BSE: 9:15–15:30 IST = 3:45–10:00 UTC
      return timeMin >= 3 * 60 + 45 && timeMin < 10 * 60;

    case "UAE":
      // DFM/ADX: 10:00–14:50 GST = 6:00–10:50 UTC; also Mon–Fri (Fri close 13:50)
      // Friday the DFM closes at 13:50 GST = 9:50 UTC
      if (day === 5) return timeMin >= 6 * 60 && timeMin < 9 * 60 + 50;
      return timeMin >= 6 * 60 && timeMin < 10 * 60 + 50;

    case "Saudi":
      // Tadawul: 10:00–15:00 AST (UTC+3) = 7:00–12:00 UTC; Sun–Thu
      if (day === 5 || day === 0) return false; // Fri/Sat are weekend for Saudi
      // Remap: Saudi working week is Sun–Thu
      return timeMin >= 7 * 60 && timeMin < 12 * 60;

    default:
      return false;
  }
}
