"use client";
import { useEffect, useState } from "react";
import { isMarketOpen, getMarketLocalTime } from "@/lib/marketHours";

const MARKETS = [
  { key: "US", label: "US (NYSE/NASDAQ)", flag: "🇺🇸" },
  { key: "India", label: "India (NSE)", flag: "🇮🇳" },
  { key: "UAE", label: "UAE (DFM/ADX)", flag: "🇦🇪" },
  { key: "Saudi", label: "Saudi (TADAWUL)", flag: "🇸🇦" },
];

export default function MarketStatusBar() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 60_000);
    return () => clearInterval(id);
  }, []);
  void tick;

  return (
    <div className="flex flex-wrap gap-2 px-2 py-2">
      {MARKETS.map(({ key, label, flag }) => {
        const open = isMarketOpen(key);
        const time = getMarketLocalTime(key);
        return (
          <div
            key={key}
            className="flex items-center gap-2 rounded-full px-3 py-1 text-xs border"
            style={{
              borderColor: open ? "#16a34a" : "#475569",
              backgroundColor: open ? "rgb(22 163 74 / 0.1)" : "rgb(15 23 42 / 0.5)",
            }}
          >
            <span>{flag}</span>
            <span className="text-slate-300 hidden sm:inline">{label}</span>
            <span className="text-slate-400 sm:hidden">{key}</span>
            <span className="font-mono text-slate-300">{time}</span>
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: open ? "#22c55e" : "#64748b" }}
            />
            <span style={{ color: open ? "#4ade80" : "#94a3b8" }}>
              {open ? "OPEN" : "CLOSED"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
