"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { StockData, FilterState, SortField, SortDirection, Market, PresetName } from "@/types";
import { fmtPct, fmtMarketCap, pctColor, MARKET_FLAG, cn } from "@/lib/utils";
import FilterBar from "./FilterBar";
import MiniSparkline from "./MiniSparkline";

const REFRESH_INTERVAL = Number(process.env.NEXT_PUBLIC_REFRESH_INTERVAL_MS ?? 720_000);
const VIRTUAL_THRESHOLD = Number(process.env.NEXT_PUBLIC_VIRTUAL_ROW_THRESHOLD ?? 100);

async function fetchBatch(markets: Market[]): Promise<StockData[]> {
  const res = await fetch("/api/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ market: markets.length === 4 ? "All" : markets[0] ?? "All" }),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  const json = await res.json();
  return json.data;
}

const DEFAULT_FILTERS: FilterState = {
  markets: ["US", "India", "UAE", "Saudi"],
  sectors: [],
  sortField: "daily",
  sortDir: "desc",
  topN: null,
  minChangePct: null,
  search: "",
};

function applyFilters(data: StockData[], f: FilterState): StockData[] {
  let rows = data.filter((d) => f.markets.includes(d.market));

  if (f.sectors.length) rows = rows.filter((d) => f.sectors.includes(d.sector));

  if (f.minChangePct !== null) {
    rows = rows.filter((d) => {
      const v = d.performance[f.sortField as keyof typeof d.performance] as number | null;
      return v !== null && v >= (f.minChangePct as number);
    });
  }

  if (f.search.trim()) {
    const q = f.search.trim().toLowerCase();
    rows = rows.filter(
      (d) =>
        d.symbol.toLowerCase().includes(q) ||
        d.name.toLowerCase().includes(q)
    );
  }

  rows.sort((a, b) => {
    const getVal = (d: StockData): number => {
      if (f.sortField === "marketCap") return d.marketCap ?? -Infinity;
      if (f.sortField === "volume") return d.volume ?? -Infinity;
      const p = d.performance[f.sortField as keyof typeof d.performance] as number | null;
      return p ?? -Infinity;
    };
    const diff = getVal(a) - getVal(b);
    return f.sortDir === "desc" ? -diff : diff;
  });

  if (f.topN) rows = rows.slice(0, f.topN);
  return rows;
}

function applyPreset(preset: PresetName): FilterState {
  const base = { ...DEFAULT_FILTERS };
  switch (preset) {
    case "top-gainers":    return { ...base, sortField: "daily",   sortDir: "desc", topN: 25 };
    case "top-losers":     return { ...base, sortField: "daily",   sortDir: "asc",  topN: 25 };
    case "ytd-leaders":    return { ...base, sortField: "ytd",     sortDir: "desc", topN: 25 };
    case "five-year-compounders": return { ...base, sortField: "five_y", sortDir: "desc", topN: 25 };
    case "most-active":    return { ...base, sortField: "volume",  sortDir: "desc", topN: 25 };
    default: return base;
  }
}

function PctCell({ value, sortField }: { value: number | null; sortField: SortField }) {
  void sortField;
  return (
    <span className={cn("font-mono tabular-nums text-sm", pctColor(value))}>
      {fmtPct(value)}
    </span>
  );
}

function SortHeader({
  label, field, current, dir, onSort,
}: {
  label: string; field: SortField; current: SortField; dir: SortDirection;
  onSort: (f: SortField) => void;
}) {
  const active = current === field;
  return (
    <th
      className="px-3 py-2 text-left text-xs font-semibold text-slate-400 cursor-pointer select-none whitespace-nowrap hover:text-white transition-colors"
      onClick={() => onSort(field)}
    >
      {label}
      {active && <span className="ml-1 text-blue-400">{dir === "desc" ? "↓" : "↑"}</span>}
    </th>
  );
}

function SkeletonRow() {
  return (
    <tr className="border-b border-slate-800 animate-pulse">
      {Array.from({ length: 14 }).map((_, i) => (
        <td key={i} className="px-3 py-3">
          <div className="h-3 bg-slate-800 rounded w-full" />
        </td>
      ))}
    </tr>
  );
}

interface Props {
  onSelectStock: (s: StockData) => void;
}

export default function ScreenerTable({ onSelectStock }: Props) {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [searchInput, setSearchInput] = useState("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [countdown, setCountdown] = useState<string>("");
  const nextRefreshRef = useRef<number>(Date.now() + REFRESH_INTERVAL);

  // Debounce search
  useEffect(() => {
    const t = setTimeout(() => setFilters((f) => ({ ...f, search: searchInput })), 300);
    return () => clearTimeout(t);
  }, [searchInput]);

  const { data, isLoading, isRefetching, isError, dataUpdatedAt } = useQuery({
    queryKey: ["screener", filters.markets],
    queryFn: () => fetchBatch(filters.markets),
    refetchInterval: REFRESH_INTERVAL,
    refetchIntervalInBackground: true,
    staleTime: REFRESH_INTERVAL,
    placeholderData: keepPreviousData,
  });

  useEffect(() => {
    if (dataUpdatedAt) {
      setLastUpdated(new Date(dataUpdatedAt));
      nextRefreshRef.current = dataUpdatedAt + REFRESH_INTERVAL;
    }
  }, [dataUpdatedAt]);

  // Countdown timer
  useEffect(() => {
    const tick = () => {
      const remaining = Math.max(0, nextRefreshRef.current - Date.now());
      const m = Math.floor(remaining / 60000);
      const s = Math.floor((remaining % 60000) / 1000);
      setCountdown(`${m}:${s.toString().padStart(2, "0")}`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    return applyFilters(data, filters);
  }, [data, filters]);

  const handleSort = useCallback((field: SortField) => {
    setFilters((f) => ({
      ...f,
      sortField: field,
      sortDir: f.sortField === field && f.sortDir === "desc" ? "asc" : "desc",
    }));
  }, []);

  const exportCSV = useCallback(() => {
    if (!rows.length) return;
    const headers = ["#","Ticker","Company","Market","Price","Currency","Daily%","WTD%","MTD%","YTD%","1Y%","3Y%","5Y%","MarketCap","Volume","Timestamp"];
    const lines = rows.map((r, i) => [
      i + 1, r.symbol, `"${r.name}"`, r.market,
      r.price ?? "", r.currency,
      r.performance.daily ?? "N/A",
      r.performance.wtd ?? "N/A",
      r.performance.mtd ?? "N/A",
      r.performance.ytd ?? "N/A",
      r.performance.one_y ?? "N/A",
      r.performance.three_y ?? "N/A",
      r.performance.five_y ?? "N/A",
      r.marketCap ?? "N/A",
      r.volume ?? "N/A",
      new Date().toISOString(),
    ].join(","));
    const csv = [headers.join(","), ...lines].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `screener_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
  }, [rows]);

  const statusDot = isError
    ? "bg-red-500"
    : isRefetching
    ? "bg-amber-400 animate-pulse"
    : "bg-emerald-400";

  const isVirtual = rows.length > VIRTUAL_THRESHOLD;

  return (
    <div className="flex flex-col gap-3">
      {/* Filter bar + status */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-slate-900 rounded-xl border border-slate-700 px-4">
        <FilterBar
          filters={filters}
          onChange={setFilters}
          onPreset={(p) => setFilters(applyPreset(p))}
        />
        <div className="flex items-center gap-3 text-xs text-slate-400 py-2 ml-auto pl-4 border-l border-slate-700">
          <span className={cn("w-2 h-2 rounded-full", statusDot)} />
          {lastUpdated && (
            <span>Updated {lastUpdated.toLocaleTimeString()}</span>
          )}
          <span>Next in {countdown}</span>
          <button
            onClick={exportCSV}
            className="px-2 py-1 rounded border border-slate-700 hover:border-slate-500 hover:text-white transition-colors ml-1"
          >
            ↓ CSV
          </button>
        </div>
      </div>

      {/* Stale data banner */}
      {isError && (
        <div className="bg-amber-900/40 border border-amber-700 rounded-lg px-4 py-2 text-amber-300 text-sm">
          Data source unavailable — showing last cached values. Retrying...
        </div>
      )}

      {/* Table */}
      <div className="rounded-xl border border-slate-700 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-900 sticky top-0 z-10">
              <tr className="border-b border-slate-700">
                <th className="px-3 py-2 text-left text-xs text-slate-500 w-8">#</th>
                <th className="px-3 py-2 text-left text-xs text-slate-400 whitespace-nowrap">Ticker</th>
                <th className="px-3 py-2 text-left text-xs text-slate-400">Company</th>
                <th className="px-3 py-2 text-left text-xs text-slate-400">Market</th>
                <SortHeader label="Price"  field="price"    current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="1D %"   field="daily"    current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="WTD %"  field="wtd"      current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="MTD %"  field="mtd"      current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="YTD %"  field="ytd"      current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="1Y %"   field="one_y"    current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="3Y %*"  field="three_y"  current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="5Y %*"  field="five_y"   current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="Mkt Cap" field="marketCap" current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <th className="px-3 py-2 text-left text-xs text-slate-400">Chart (1M)</th>
              </tr>
            </thead>
            <tbody>
              {isLoading
                ? Array.from({ length: 10 }).map((_, i) => <SkeletonRow key={i} />)
                : rows.map((stock, idx) => (
                    <tr
                      key={stock.symbol}
                      onClick={() => onSelectStock(stock)}
                      className={cn(
                        "border-b border-slate-800 cursor-pointer transition-colors",
                        "hover:bg-slate-800/60",
                        isRefetching && "opacity-70",
                        stock.isStale && "opacity-60"
                      )}
                    >
                      <td className="px-3 py-2 text-slate-600 tabular-nums">{idx + 1}</td>
                      <td className="px-3 py-2 font-semibold text-white whitespace-nowrap">
                        {stock.symbol}
                        {stock.isStale && (
                          <span className="ml-1 text-xs text-amber-500 font-normal">STALE</span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-slate-300 max-w-[180px] truncate">{stock.name}</td>
                      <td className="px-3 py-2 text-slate-400 whitespace-nowrap">
                        {MARKET_FLAG[stock.market]} {stock.market}
                      </td>
                      <td className="px-3 py-2 text-white font-mono tabular-nums whitespace-nowrap">
                        {stock.price !== null
                          ? stock.price.toLocaleString(undefined, { maximumFractionDigits: 4 })
                          : "N/A"}
                        {" "}
                        <span className="text-slate-600 text-xs">{stock.currency}</span>
                      </td>
                      <td className="px-3 py-2"><PctCell value={stock.performance.daily}   sortField={filters.sortField} /></td>
                      <td className="px-3 py-2"><PctCell value={stock.performance.wtd}     sortField={filters.sortField} /></td>
                      <td className="px-3 py-2"><PctCell value={stock.performance.mtd}     sortField={filters.sortField} /></td>
                      <td className="px-3 py-2"><PctCell value={stock.performance.ytd}     sortField={filters.sortField} /></td>
                      <td className="px-3 py-2"><PctCell value={stock.performance.one_y}   sortField={filters.sortField} /></td>
                      <td className="px-3 py-2"><PctCell value={stock.performance.three_y} sortField={filters.sortField} /></td>
                      <td className="px-3 py-2"><PctCell value={stock.performance.five_y}  sortField={filters.sortField} /></td>
                      <td className="px-3 py-2 text-slate-400 tabular-nums text-xs">
                        {fmtMarketCap(stock.marketCap)}
                      </td>
                      <td className="px-2 py-1">
                        {!isVirtual && <MiniSparkline tvSymbol={stock.tvSymbol} />}
                      </td>
                    </tr>
                  ))}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-2 bg-slate-900 border-t border-slate-800 flex justify-between text-xs text-slate-600">
          <span>
            Showing {rows.length} stocks · *3Y and 5Y are cumulative total returns
          </span>
          <span>Prices delayed up to 15 minutes</span>
        </div>
      </div>
    </div>
  );
}
