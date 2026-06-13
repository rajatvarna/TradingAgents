"use client";

import { StockData } from "@/types";
import { cn } from "@/lib/utils";

function cellColor(daily: number | null): string {
  if (daily === null) return "bg-slate-700";
  if (daily < -3) return "bg-red-900";
  if (daily < -1) return "bg-red-700";
  if (daily < 0) return "bg-red-500/40";
  if (daily < 1) return "bg-emerald-500/40";
  if (daily < 3) return "bg-emerald-600";
  return "bg-emerald-800";
}

export function SectorHeatmap({
  stocks,
  onSelectStock,
}: {
  stocks: StockData[];
  onSelectStock: (s: StockData) => void;
}) {
  // Group by sector
  const sectorMap = new Map<string, StockData[]>();
  for (const s of stocks) {
    const sector = s.sector || "Other";
    if (!sectorMap.has(sector)) sectorMap.set(sector, []);
    sectorMap.get(sector)!.push(s);
  }

  const sectors = Array.from(sectorMap.entries()).sort((a, b) => b[1].length - a[1].length);

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-4 space-y-4">
      <h2 className="text-sm font-semibold text-slate-300">Sector Heatmap</h2>
      {sectors.map(([sector, sectorStocks]) => {
        const totalCap = sectorStocks.reduce((s, d) => s + (d.marketCap ?? 0), 0);
        return (
          <div key={sector}>
            <div className="text-xs text-slate-500 mb-1 font-medium">{sector}</div>
            <div className="flex flex-wrap gap-1">
              {sectorStocks.map((stock) => {
                const capShare = totalCap > 0 ? (stock.marketCap ?? 0) / totalCap : 0;
                const minW = 60;
                const maxW = 200;
                const width = Math.round(minW + capShare * (maxW - minW) * sectorStocks.length);
                return (
                  <button
                    key={stock.symbol}
                    onClick={() => onSelectStock(stock)}
                    title={`${stock.name} · ${stock.performance.daily !== null ? (stock.performance.daily > 0 ? "+" : "") + stock.performance.daily.toFixed(2) + "%" : "N/A"}`}
                    style={{ minWidth: Math.min(width, 200), maxWidth: 200 }}
                    className={cn(
                      "flex flex-col items-center justify-center px-2 py-2 rounded text-center cursor-pointer transition-opacity hover:opacity-80",
                      cellColor(stock.performance.daily)
                    )}
                  >
                    <span className="text-xs font-bold text-white truncate w-full text-center">
                      {stock.symbol}
                    </span>
                    <span className="text-xs text-white/80">
                      {stock.performance.daily !== null
                        ? (stock.performance.daily > 0 ? "+" : "") + stock.performance.daily.toFixed(2) + "%"
                        : "—"}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
