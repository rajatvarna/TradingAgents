"use client";

import { useQuery } from "@tanstack/react-query";

interface VixData {
  vix: number;
  label: string;
  timestamp: string;
}

async function fetchVix(): Promise<VixData> {
  const res = await fetch("/api/vix");
  if (!res.ok) throw new Error(`VIX fetch error ${res.status}`);
  return res.json();
}

function vixColor(vix: number): { bar: string; text: string; bg: string } {
  if (vix < 15)  return { bar: "bg-emerald-500", text: "text-emerald-400", bg: "bg-emerald-900/20" };
  if (vix < 20)  return { bar: "bg-yellow-400",  text: "text-yellow-400",  bg: "bg-yellow-900/20" };
  if (vix < 28)  return { bar: "bg-orange-500",  text: "text-orange-400",  bg: "bg-orange-900/20" };
  if (vix < 35)  return { bar: "bg-red-500",     text: "text-red-400",     bg: "bg-red-900/20" };
  return              { bar: "bg-red-900",       text: "text-red-300",     bg: "bg-red-950/40" };
}

/** Clamp VIX to 0–50 for the gauge bar width */
function gaugeWidth(vix: number): number {
  return Math.min(100, Math.max(0, (vix / 50) * 100));
}

export default function FearGreed() {
  const { data, isLoading, isError } = useQuery<VixData>({
    queryKey: ["vix"],
    queryFn: fetchVix,
    refetchInterval: 5 * 60 * 1000, // 5 minutes
    staleTime: 5 * 60 * 1000,
  });

  if (isLoading) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 flex items-center gap-4 animate-pulse">
        <div className="h-4 w-24 bg-slate-800 rounded" />
        <div className="h-3 w-48 bg-slate-800 rounded flex-1" />
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-xs text-slate-500">
        VIX unavailable
      </div>
    );
  }

  const colors = vixColor(data.vix);
  const width = gaugeWidth(data.vix);

  return (
    <div
      className={`rounded-xl border border-slate-700 ${colors.bg} px-4 py-3 flex flex-wrap items-center gap-4`}
    >
      {/* Label */}
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-xs text-slate-400 font-medium">VIX</span>
        <span className={`text-2xl font-bold tabular-nums ${colors.text}`}>
          {data.vix.toFixed(2)}
        </span>
        <span className={`text-sm font-semibold ${colors.text}`}>{data.label}</span>
      </div>

      {/* Gauge bar */}
      <div className="flex-1 min-w-[160px]">
        <div className="relative h-3 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`absolute left-0 top-0 h-full rounded-full transition-all duration-700 ${colors.bar}`}
            style={{ width: `${width}%` }}
          />
        </div>
        {/* Scale labels */}
        <div className="flex justify-between text-[10px] text-slate-600 mt-0.5 px-0.5">
          <span>0</span>
          <span>Calm</span>
          <span>Fear</span>
          <span>50+</span>
        </div>
      </div>

      <span className="text-[10px] text-slate-600 shrink-0">
        as of {new Date(data.timestamp).toLocaleTimeString()}
      </span>
    </div>
  );
}
