"use client";

import { useEffect, useState } from "react";
import { FilterState, Market, MarketState, PresetName, SortField } from "@/types";
import { getAllSectors } from "@/lib/watchlist";
import { MARKET_FLAG } from "@/lib/utils";
import { getMarketState } from "@/lib/marketHours";

const MARKETS: Market[] = ["US", "India", "UAE", "Saudi"];
const TOP_N_OPTIONS = [10, 25, 50, 100, 0]; // 0 = All
const SORT_OPTIONS: { value: SortField; label: string }[] = [
  { value: "daily",        label: "1D %" },
  { value: "wtd",          label: "WTD %" },
  { value: "mtd",          label: "MTD %" },
  { value: "ytd",          label: "YTD %" },
  { value: "one_y",        label: "1Y %" },
  { value: "three_y",      label: "3Y %" },
  { value: "five_y",       label: "5Y %" },
  { value: "marketCap",    label: "Mkt Cap" },
  { value: "volume",       label: "Volume" },
  { value: "rs",           label: "RS Score" },
  { value: "dividendYield", label: "Div Yield" },
  { value: "beta",         label: "Beta" },
];

const PRESETS: { name: PresetName; label: string }[] = [
  { name: "top-gainers",           label: "Top Gainers" },
  { name: "top-losers",            label: "Top Losers" },
  { name: "ytd-leaders",           label: "YTD Leaders" },
  { name: "five-year-compounders", label: "5Y Compounders" },
  { name: "most-active",           label: "Most Active" },
  { name: "near-52w-high",         label: "📈 Near 52W High" },
  { name: "vol-surge",             label: "🔥 Vol Surge" },
  { name: "watchlist",             label: "⭐ Watchlist" },
];

const STATE_DOT: Record<MarketState, string> = {
  open:        "bg-emerald-400",
  "pre-market": "bg-amber-400",
  closed:      "bg-slate-600",
};

const STATE_LABEL: Record<MarketState, string> = {
  open:        "Open",
  "pre-market": "Pre",
  closed:      "Closed",
};

interface Props {
  filters: FilterState;
  onChange: (f: FilterState) => void;
  onPreset: (p: PresetName) => void;
}

export default function FilterBar({ filters, onChange, onPreset }: Props) {
  const sectors = getAllSectors();
  const [marketStates, setMarketStates] = useState<Record<Market, MarketState>>(() =>
    Object.fromEntries(MARKETS.map((m) => [m, getMarketState(m)])) as Record<Market, MarketState>
  );

  // Refresh market states every minute
  useEffect(() => {
    const refresh = () =>
      setMarketStates(
        Object.fromEntries(MARKETS.map((m) => [m, getMarketState(m)])) as Record<Market, MarketState>
      );
    const id = setInterval(refresh, 60_000);
    return () => clearInterval(id);
  }, []);

  const toggleMarket = (m: Market) => {
    const next = filters.markets.includes(m)
      ? filters.markets.filter((x) => x !== m)
      : [...filters.markets, m];
    onChange({ ...filters, markets: next.length ? next : MARKETS });
  };

  const toggleSector = (s: string) => {
    const next = filters.sectors.includes(s)
      ? filters.sectors.filter((x) => x !== s)
      : [...filters.sectors, s];
    onChange({ ...filters, sectors: next });
  };

  return (
    <div className="flex flex-wrap gap-3 items-center py-3 px-1">
      {/* Market toggles with live state badges */}
      <div className="flex gap-1">
        {MARKETS.map((m) => {
          const state = marketStates[m];
          const active = filters.markets.includes(m);
          return (
            <button
              key={m}
              onClick={() => toggleMarket(m)}
              title={`${m}: ${STATE_LABEL[state]}`}
              aria-pressed={active}
              aria-label={`${m}, ${STATE_LABEL[state]}, ${active ? "selected" : "not selected"}`}
              className={`relative px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                active
                  ? "bg-blue-600 border-blue-500 text-white"
                  : "bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500"
              }`}
            >
              {MARKET_FLAG[m]} {m}
              <span
                className={`absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full border border-slate-900 ${STATE_DOT[state]}`}
                aria-hidden="true"
              />
            </button>
          );
        })}
      </div>

      {/* Market state legend */}
      <div className="flex items-center gap-2 text-[10px] text-slate-600">
        <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />Open</span>
        <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />Pre</span>
        <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-slate-600 inline-block" />Closed</span>
      </div>

      {/* Sector dropdown */}
      <select
        value=""
        onChange={(e) => { if (e.target.value) toggleSector(e.target.value); }}
        className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-lg px-2 py-1"
      >
        <option value="">Sector ▼</option>
        {sectors.map((s) => (
          <option key={s} value={s}>
            {filters.sectors.includes(s) ? "✓ " : ""}
            {s}
          </option>
        ))}
      </select>
      {filters.sectors.length > 0 && (
        <button
          onClick={() => onChange({ ...filters, sectors: [] })}
          className="text-xs text-slate-400 hover:text-white"
        >
          Clear sectors ✕
        </button>
      )}

      {/* Sort field */}
      <div className="flex items-center gap-1">
        <span className="text-slate-500 text-xs">Sort:</span>
        <select
          value={filters.sortField}
          onChange={(e) => onChange({ ...filters, sortField: e.target.value as SortField })}
          className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-lg px-2 py-1"
        >
          {SORT_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <button
          onClick={() =>
            onChange({ ...filters, sortDir: filters.sortDir === "desc" ? "asc" : "desc" })
          }
          className="text-slate-400 hover:text-white text-sm px-1"
          title="Toggle sort direction"
        >
          {filters.sortDir === "desc" ? "↓" : "↑"}
        </button>
      </div>

      {/* Top N */}
      <div className="flex items-center gap-1">
        <span className="text-slate-500 text-xs">Show:</span>
        <select
          value={filters.topN ?? 0}
          onChange={(e) => {
            const v = Number(e.target.value);
            onChange({ ...filters, topN: v === 0 ? null : v });
          }}
          className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-lg px-2 py-1"
        >
          {TOP_N_OPTIONS.map((n) => (
            <option key={n} value={n}>{n === 0 ? "All" : `Top ${n}`}</option>
          ))}
        </select>
      </div>

      {/* Min % change */}
      <div className="flex items-center gap-1">
        <span className="text-slate-500 text-xs">Min %:</span>
        <input
          type="number"
          step="0.1"
          placeholder="e.g. 2"
          value={filters.minChangePct ?? ""}
          onChange={(e) => onChange({ ...filters, minChangePct: e.target.value ? Number(e.target.value) : null })}
          className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-lg px-2 py-1 w-16"
        />
      </div>

      {/* Search */}
      <input
        type="search"
        placeholder="Search ticker / name..."
        value={filters.search}
        onChange={(e) => onChange({ ...filters, search: e.target.value })}
        className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-lg px-3 py-1 w-44 ml-auto"
      />

      {/* Presets */}
      <div className="flex gap-1 flex-wrap">
        {PRESETS.map((p) => (
          <button
            key={p.name}
            onClick={() => onPreset(p.name)}
            className="px-2 py-1 text-xs rounded border border-slate-700 bg-slate-800 text-slate-400 hover:text-white hover:border-slate-500 transition-colors"
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}
