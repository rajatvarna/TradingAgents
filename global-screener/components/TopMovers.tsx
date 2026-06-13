"use client";

import { useMemo } from "react";
import { StockData } from "@/types";
import { cn } from "@/lib/utils";

interface Props {
  stocks: StockData[];
  onSelectStock: (s: StockData) => void;
}

export default function TopMovers({ stocks, onSelectStock }: Props) {
  const { gainers, losers } = useMemo(() => {
    const valid = stocks.filter((s) => s.performance.daily !== null);
    const sorted = [...valid].sort((a, b) => (b.performance.daily ?? 0) - (a.performance.daily ?? 0));
    return { gainers: sorted.slice(0, 5), losers: sorted.slice(-5).reverse() };
  }, [stocks]);

  if (stocks.length === 0) return null;

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 overflow-x-auto">
      <div className="flex items-center gap-4 min-w-max">
        <span className="text-xs font-semibold text-emerald-400 whitespace-nowrap">▲ Top Gainers</span>
        {gainers.map((s) => (
          <button
            key={s.symbol}
            onClick={() => onSelectStock(s)}
            className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-900/40 border border-emerald-700/50 hover:bg-emerald-900/70 transition-colors"
          >
            <span className="text-xs font-semibold text-white">{s.symbol}</span>
            <span className="text-xs font-mono text-emerald-400">
              +{s.performance.daily!.toFixed(2)}%
            </span>
          </button>
        ))}

        <span className="text-slate-700">│</span>

        <span className="text-xs font-semibold text-red-400 whitespace-nowrap">▼ Top Losers</span>
        {losers.map((s) => (
          <button
            key={s.symbol}
            onClick={() => onSelectStock(s)}
            className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-red-900/40 border border-red-700/50 hover:bg-red-900/70 transition-colors"
          >
            <span className="text-xs font-semibold text-white">{s.symbol}</span>
            <span className={cn("text-xs font-mono text-red-400")}>
              {s.performance.daily!.toFixed(2)}%
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
