"use client";

import { useQuery } from "@tanstack/react-query";
import { cn } from "@/lib/utils";

interface VixData { price: number | null; changePct: number | null; fetchedAt: string; }

function getMood(vix: number | null): { label: string; color: string; barColor: string; pct: number } {
  if (!vix) return { label: "—", color: "text-slate-400", barColor: "#64748b", pct: 50 };
  if (vix < 15) return { label: "Calm",         color: "text-emerald-400", barColor: "#22c55e", pct: Math.round(vix / 50 * 100) };
  if (vix < 20) return { label: "Neutral",       color: "text-slate-300",   barColor: "#94a3b8", pct: Math.round(vix / 50 * 100) };
  if (vix < 30) return { label: "Cautious",      color: "text-amber-400",   barColor: "#f59e0b", pct: Math.round(vix / 50 * 100) };
  if (vix < 40) return { label: "Fearful",       color: "text-orange-400",  barColor: "#f97316", pct: Math.round(vix / 50 * 100) };
  return           { label: "Extreme Fear",  color: "text-red-400",     barColor: "#ef4444", pct: 100 };
}

export default function FearGreed() {
  const { data, isLoading } = useQuery<VixData>({
    queryKey: ["vix"],
    queryFn: () => fetch("/api/vix").then((r) => r.json()),
    staleTime: 5 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000,
  });

  if (isLoading) {
    return <div className="h-16 w-40 bg-slate-800 rounded-xl animate-pulse" />;
  }

  const mood = getMood(data?.price ?? null);

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 min-w-[160px]">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-500 font-semibold uppercase tracking-wide">VIX</span>
        <span className={cn("text-lg font-bold font-mono tabular-nums", mood.color)}>
          {data?.price?.toFixed(2) ?? "—"}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-700 overflow-hidden mb-1">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${mood.pct}%`, background: mood.barColor }}
        />
      </div>
      <div className="flex items-center justify-between">
        <span className={cn("text-xs font-semibold", mood.color)}>{mood.label}</span>
        {data?.changePct !== null && data?.changePct !== undefined && (
          <span className={cn("text-xs font-mono", data.changePct >= 0 ? "text-red-400" : "text-emerald-400")}>
            {data.changePct >= 0 ? "+" : ""}{data.changePct.toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}
