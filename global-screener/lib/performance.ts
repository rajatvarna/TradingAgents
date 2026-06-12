import { PerformanceMetrics } from "@/types";

interface DailyClose {
  date: string;
  close: number;
}

/** Returns Monday of the current week (YYYY-MM-DD) */
function getLastMonday(): string {
  const d = new Date();
  const day = d.getDay(); // 0=Sun
  const diff = day === 0 ? 6 : day - 1;
  d.setDate(d.getDate() - diff);
  return d.toISOString().split("T")[0];
}

/** Returns the first calendar day of the current month (YYYY-MM-DD) */
function getFirstDayOfMonth(): string {
  const d = new Date();
  return new Date(d.getFullYear(), d.getMonth(), 1).toISOString().split("T")[0];
}

/** Returns the last calendar day of the given year (YYYY-MM-DD) */
function getLastDayOfYear(year: number): string {
  return new Date(year, 11, 31).toISOString().split("T")[0];
}

/**
 * Finds the closest trading day close at or before a target date.
 * Assumes `closes` is sorted ascending by date.
 */
function findClosestClose(
  closes: DailyClose[],
  targetDate: string
): number | null {
  let result: number | null = null;
  for (const c of closes) {
    if (c.date <= targetDate) result = c.close;
    else break;
  }
  return result;
}

/**
 * Calculates percentage return from base to current price.
 * Returns null when base is missing or zero (e.g. insufficient history).
 */
function calcReturn(current: number, base: number | null): number | null {
  if (base === null || base === 0) return null;
  return Math.round(((current - base) / base) * 10000) / 100;
}

/**
 * Computes all 7 performance metrics from a sorted array of daily closes.
 *
 * @param closes - Array of {date, close} sorted ascending by date
 * @param todayPrice - Current live price (may differ from last close during session)
 * @returns PerformanceMetrics with null for periods lacking sufficient history
 */
export function computePerformance(
  closes: DailyClose[],
  todayPrice: number
): PerformanceMetrics {
  if (!closes.length) {
    return {
      daily: null, wtd: null, mtd: null, ytd: null,
      one_y: null, three_y: null, five_y: null,
    };
  }

  const sorted = [...closes].sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
  );

  const lastMonday = getLastMonday();
  const firstOfMonth = getFirstDayOfMonth();
  const lastYearEnd = getLastDayOfYear(new Date().getFullYear() - 1);

  const prevClose = sorted.length >= 2 ? sorted[sorted.length - 2].close : null;
  const wtdBase = findClosestClose(sorted, lastMonday);
  const mtdBase = findClosestClose(sorted, firstOfMonth);
  const ytdBase = findClosestClose(sorted, lastYearEnd);

  // Rolling lookbacks: approximate 252 trading days per year
  const oneYBase = sorted.at(-253)?.close ?? null;
  const threeYBase = sorted.at(-757)?.close ?? null;
  const fiveYBase = sorted.at(-1261)?.close ?? null;

  return {
    daily:   calcReturn(todayPrice, prevClose),
    wtd:     calcReturn(todayPrice, wtdBase),
    mtd:     calcReturn(todayPrice, mtdBase),
    ytd:     calcReturn(todayPrice, ytdBase),
    one_y:   calcReturn(todayPrice, oneYBase),
    three_y: calcReturn(todayPrice, threeYBase),
    five_y:  calcReturn(todayPrice, fiveYBase),
  };
}

export type { DailyClose };
