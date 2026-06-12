"use client";

import { useMemo } from "react";
import { StockData } from "@/types";
import { fmtPct } from "@/lib/utils";

interface Props {
  data: StockData[];
}

function heatColor(pct: number): string {
  if (pct >= 3)    return "bg-emerald-700 text-emerald-100";
  if (pct >= 1.5)  return "bg-emerald-800 text-emerald-200";
  if (pct >= 0.5)  return "bg-emerald-900 text-emerald-300";
  if (pct >= 0)    return "bg-slate-800 text-slate-300";
  if (pct >= -0.5) return "bg-red-950 text-red-300";
  if (pct >= -1.5) return "bg-red-900 text-red-200";
  return "bg-red-800 text-red-100";
}

/** Sector heatmap — average daily % change per sector across visible stocks. */
export default function SectorHeatmap({ data }: Props) {
  const sectors = useMemo(() => {
    const map = new Map<string, { sum: number; count: number }>();
    for (const s of data) {
      const d = s.performance.daily;
      if (d === null) continue;
      const entry = map.get(s.sector) ?? { sum: 0, count: 0 };
      entry.sum += d;
      entry.count += 1;
      map.set(s.sector, entry);
    }
    return [...map.entries()]
      .map(([sector, { sum, count }]) => ({ sector, avg: sum / count }))
      .sort((a, b) => b.avg - a.avg);
  }, [data]);

  if (!sectors.length) return null;

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-4">
      <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
        Sector Heatmap — Avg Daily %
      </h2>
      <div className="flex flex-wrap gap-2">
        {sectors.map(({ sector, avg }) => (
          <div
            key={sector}
            className={`rounded-lg px-3 py-2 text-center min-w-[110px] flex-1 ${heatColor(avg)}`}
          >
            <div className="text-xs font-medium truncate">{sector}</div>
            <div className="text-sm font-bold tabular-nums mt-0.5">{fmtPct(avg)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
