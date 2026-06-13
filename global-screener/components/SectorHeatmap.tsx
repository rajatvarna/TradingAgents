"use client";

import { useMemo } from "react";
import { StockData } from "@/types";
import { cn } from "@/lib/utils";

interface Props {
  stocks: StockData[];
  onSelectStock: (s: StockData) => void;
}

function dailyColor(pct: number | null): string {
  if (pct === null) return "bg-slate-800 text-slate-400";
  if (pct >= 5)   return "bg-emerald-600 text-white";
  if (pct >= 3)   return "bg-emerald-700 text-white";
  if (pct >= 1)   return "bg-emerald-800 text-emerald-100";
  if (pct >= 0)   return "bg-emerald-900/60 text-emerald-300";
  if (pct >= -1)  return "bg-red-900/60 text-red-300";
  if (pct >= -3)  return "bg-red-800 text-red-100";
  if (pct >= -5)  return "bg-red-700 text-white";
  return "bg-red-600 text-white";
}

export default function SectorHeatmap({ stocks, onSelectStock }: Props) {
  const sectors = useMemo(() => {
    const map = new Map<string, StockData[]>();
    for (const s of stocks) {
      if (!s.sector) continue;
      const arr = map.get(s.sector) ?? [];
      arr.push(s);
      map.set(s.sector, arr);
    }
    // Sort sectors by total market cap descending
    return Array.from(map.entries())
      .sort((a, b) => {
        const capA = a[1].reduce((s, x) => s + (x.marketCap ?? 0), 0);
        const capB = b[1].reduce((s, x) => s + (x.marketCap ?? 0), 0);
        return capB - capA;
      })
      .map(([sector, sectorStocks]) => ({
        sector,
        stocks: sectorStocks.sort((a, b) => (b.marketCap ?? 0) - (a.marketCap ?? 0)),
        totalCap: sectorStocks.reduce((s, x) => s + (x.marketCap ?? 0), 0),
      }));
  }, [stocks]);

  if (stocks.length === 0) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-900 p-8 text-center text-slate-500 text-sm">
        Loading sector data…
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Sector Heatmap</h2>
        <div className="flex gap-3 text-xs text-slate-500">
          <span>Cell size = market cap · Color = 1D %</span>
          <div className="flex gap-1 items-center">
            {[["bg-emerald-600", "+5%+"], ["bg-emerald-800", "+1%"], ["bg-slate-800", "0%"], ["bg-red-800", "-1%"], ["bg-red-600", "-5%-"]].map(([cls, label]) => (
              <span key={label} className={cn("px-1.5 py-0.5 rounded text-white text-[10px]", cls)}>{label}</span>
            ))}
          </div>
        </div>
      </div>
      <div className="p-3 space-y-3">
        {sectors.map(({ sector, stocks: sectorStocks }) => (
          <div key={sector}>
            <div className="text-xs text-slate-500 font-semibold mb-1 px-1">{sector}</div>
            <div className="flex flex-wrap gap-1">
              {sectorStocks.map((stock) => {
                const pct = stock.performance.daily;
                const cap = stock.marketCap ?? 0;
                // Normalize cell size within sector (min 48px, max 120px proportional to cap)
                const sectorMax = sectorStocks[0]?.marketCap ?? 1;
                const relSize = sectorMax > 0 ? Math.max(0.1, cap / sectorMax) : 0.1;
                const w = Math.round(48 + relSize * 72);
                const h = Math.round(36 + relSize * 24);
                return (
                  <button
                    key={stock.symbol}
                    onClick={() => onSelectStock(stock)}
                    title={`${stock.symbol} · ${stock.name} · ${pct !== null ? (pct >= 0 ? "+" : "") + pct.toFixed(2) + "%" : "N/A"}`}
                    className={cn(
                      "rounded flex flex-col items-center justify-center text-center transition-opacity hover:opacity-80 cursor-pointer",
                      dailyColor(pct)
                    )}
                    style={{ width: w, height: h }}
                  >
                    <span className="text-xs font-bold leading-tight">{stock.symbol}</span>
                    {pct !== null && (
                      <span className="text-[10px] leading-tight opacity-90">
                        {pct >= 0 ? "+" : ""}{pct.toFixed(1)}%
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
