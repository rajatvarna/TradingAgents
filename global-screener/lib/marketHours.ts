import { MarketSchedule } from "@/types";

const SCHEDULES: Record<string, MarketSchedule> = {
  US:     { open: "09:30", close: "16:00", timezone: "America/New_York",  tradingDays: [1,2,3,4,5] },
  India:  { open: "09:15", close: "15:30", timezone: "Asia/Kolkata",      tradingDays: [1,2,3,4,5] },
  UAE:    { open: "10:00", close: "15:00", timezone: "Asia/Dubai",        tradingDays: [1,2,3,4,5] },
  Saudi:  { open: "10:00", close: "15:00", timezone: "Asia/Riyadh",       tradingDays: [1,2,3,4,5] },
};

/** Returns whether the named market is currently within its regular trading session. */
export function isMarketOpen(market: string): boolean {
  const sched = SCHEDULES[market];
  if (!sched) return false;

  const now = new Date();
  const localStr = now.toLocaleString("en-US", { timeZone: sched.timezone });
  const local = new Date(localStr);

  const dayOfWeek = local.getDay(); // 0=Sun, 6=Sat
  if (!sched.tradingDays.includes(dayOfWeek)) return false;

  const hhmm = (d: Date) => d.getHours() * 60 + d.getMinutes();
  const [openH, openM] = sched.open.split(":").map(Number);
  const [closeH, closeM] = sched.close.split(":").map(Number);

  const nowMin = hhmm(local);
  return nowMin >= openH * 60 + openM && nowMin < closeH * 60 + closeM;
}

/** Returns all currently open markets. */
export function getOpenMarkets(): string[] {
  return Object.keys(SCHEDULES).filter(isMarketOpen);
}

/** Returns the local time string for a given market's timezone. */
export function getMarketLocalTime(market: string): string {
  const sched = SCHEDULES[market];
  if (!sched) return "";
  return new Date().toLocaleTimeString("en-US", {
    timeZone: sched.timezone,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}
