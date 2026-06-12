import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Formats a percentage value with sign and 2 decimal places. */
export function fmtPct(v: number | null): string {
  if (v === null) return "N/A";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

/** Formats market-cap in billions/trillions with 2 sig figs. */
export function fmtMarketCap(v: number | null): string {
  if (!v) return "N/A";
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  return `$${v.toLocaleString()}`;
}

/** Returns a Tailwind text-colour class for a % change value. */
export function pctColor(v: number | null): string {
  if (v === null) return "text-slate-500";
  if (v > 0) return "text-emerald-400";
  if (v < 0) return "text-red-400";
  return "text-slate-400";
}

/** Market flag emoji */
export const MARKET_FLAG: Record<string, string> = {
  US: "🇺🇸",
  India: "🇮🇳",
  UAE: "🇦🇪",
  Saudi: "🇸🇦",
};
