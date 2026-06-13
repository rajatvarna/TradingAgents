import { Market, PerformanceMetrics } from "@/types";

/**
 * Finds the closest close price at or before a target date string (YYYY-MM-DD).
 * closes must be sorted oldest→newest.
 */
function findCloseAtOrBefore(
  closes: Array<{ date: string; close: number }>,
  targetDate: string
): number | null {
  // Binary search for last entry <= targetDate
  let lo = 0;
  let hi = closes.length - 1;
  let result: number | null = null;

  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (closes[mid].date <= targetDate) {
      result = closes[mid].close;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  return result;
}

function pctChange(from: number | null, to: number): number | null {
  if (from === null || from === 0) return null;
  return ((to - from) / from) * 100;
}

function toDateStr(d: Date): string {
  return d.toISOString().split("T")[0];
}

export function computePerformance(
  closes: Array<{ date: string; close: number }>,
  currentPrice: number,
  market?: Market
): PerformanceMetrics {
  if (!closes.length) {
    return {
      daily: null,
      wtd: null,
      mtd: null,
      ytd: null,
      one_y: null,
      three_y: null,
      five_y: null,
    };
  }

  const now = new Date();

  // Daily: last close vs current price
  const lastClose = closes[closes.length - 1]?.close ?? null;
  const daily = pctChange(lastClose, currentPrice);

  // WTD: UAE/Saudi trade Sun–Thu so their week starts Sunday;
  // all other markets use the Mon–Fri convention (week starts Monday).
  let wtdRef: number | null;
  if (market === "UAE" || market === "Saudi") {
    // Anchor to Sunday of the current week (day 0)
    const sunday = new Date(now);
    sunday.setDate(now.getDate() - now.getDay()); // rewind to Sun
    sunday.setHours(0, 0, 0, 0);
    // Last close before the Sun–Thu week = Thursday of previous week → search at Sat
    const prevSat = new Date(sunday);
    prevSat.setDate(sunday.getDate() - 1);
    wtdRef = findCloseAtOrBefore(closes, toDateStr(prevSat));
  } else {
    // Anchor to Monday of current week
    const monday = new Date(now);
    monday.setDate(now.getDate() - ((now.getDay() + 6) % 7));
    monday.setHours(0, 0, 0, 0);
    // Last close before the Mon–Fri week = Friday of previous week → search at Sun
    const prevSunday = new Date(monday);
    prevSunday.setDate(monday.getDate() - 1);
    wtdRef = findCloseAtOrBefore(closes, toDateStr(prevSunday));
  }
  const wtd = pctChange(wtdRef, currentPrice);

  // MTD: last trading day of previous month
  const firstOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  const lastDayPrevMonth = new Date(firstOfMonth);
  lastDayPrevMonth.setDate(0);
  const mtdRef = findCloseAtOrBefore(closes, toDateStr(lastDayPrevMonth));
  const mtd = pctChange(mtdRef, currentPrice);

  // YTD: last trading day of previous year
  const lastDayPrevYear = new Date(now.getFullYear() - 1, 11, 31);
  const ytdRef = findCloseAtOrBefore(closes, toDateStr(lastDayPrevYear));
  const ytd = pctChange(ytdRef, currentPrice);

  // 1Y
  const oneYAgo = new Date(now);
  oneYAgo.setFullYear(now.getFullYear() - 1);
  const oneYRef = findCloseAtOrBefore(closes, toDateStr(oneYAgo));
  const one_y = pctChange(oneYRef, currentPrice);

  // 3Y
  const threeYAgo = new Date(now);
  threeYAgo.setFullYear(now.getFullYear() - 3);
  const threeYRef = findCloseAtOrBefore(closes, toDateStr(threeYAgo));
  const three_y = pctChange(threeYRef, currentPrice);

  // 5Y
  const fiveYAgo = new Date(now);
  fiveYAgo.setFullYear(now.getFullYear() - 5);
  const fiveYRef = findCloseAtOrBefore(closes, toDateStr(fiveYAgo));
  const five_y = pctChange(fiveYRef, currentPrice);

  return { daily, wtd, mtd, ytd, one_y, three_y, five_y };
}
