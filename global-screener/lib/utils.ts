import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { Market } from "@/types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtPct(v: number | null): string {
  if (v === null || v === undefined) return "N/A";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

export function fmtMarketCap(v: number | null): string {
  if (v === null || v === undefined) return "N/A";
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9)  return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6)  return `$${(v / 1e6).toFixed(2)}M`;
  return `$${v.toLocaleString()}`;
}

export function pctColor(v: number | null): string {
  if (v === null || v === undefined || v === 0) return "text-slate-400";
  return v > 0 ? "text-emerald-400" : "text-red-400";
}

export const MARKET_FLAG: Record<Market, string> = {
  US: "🇺🇸",
  India: "🇮🇳",
  UAE: "🇦🇪",
  Saudi: "🇸🇦",
};
