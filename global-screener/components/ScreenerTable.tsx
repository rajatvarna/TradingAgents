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
  onlyStarred: false,
  only52wHigh: false,
};

function applyFilters(data: StockData[], f: FilterState, starred: Set<string>): StockData[] {
  let rows = data.filter((d) => f.markets.includes(d.market));

  if (f.sectors.length) rows = rows.filter((d) => f.sectors.includes(d.sector));
  if (f.onlyStarred) rows = rows.filter((d) => starred.has(d.symbol));
  if (f.only52wHigh) rows = rows.filter((d) => d.fiftyTwoWeekHighChangePct !== null && d.fiftyTwoWeekHighChangePct >= -5);

  if (f.minChangePct !== null) {
    rows = rows.filter((d) => {
      const v = getFieldValue(d, f.sortField);
      return v !== null && v >= (f.minChangePct as number);
    });
  }

  if (f.search.trim()) {
    const q = f.search.trim().toLowerCase();
    rows = rows.filter(
      (d) => d.symbol.toLowerCase().includes(q) || d.name.toLowerCase().includes(q)
    );
  }

  rows.sort((a, b) => {
    const av = getFieldValue(a, f.sortField) ?? -Infinity;
    const bv = getFieldValue(b, f.sortField) ?? -Infinity;
    return f.sortDir === "desc" ? bv - av : av - bv;
  });

  if (f.topN) rows = rows.slice(0, f.topN);
  return rows;
}

function getFieldValue(d: StockData, field: SortField): number | null {
  switch (field) {
    case "marketCap": return d.marketCap;
    case "volume":    return d.volume;
    case "price":     return d.price;
    case "fiftyTwoWeekHighChangePct": return d.fiftyTwoWeekHighChangePct;
    default: return d.performance[field as keyof typeof d.performance] as number | null;
  }
}

function applyPreset(preset: PresetName): FilterState {
  const base = { ...DEFAULT_FILTERS };
  switch (preset) {
    case "top-gainers":    return { ...base, sortField: "daily",   sortDir: "desc", topN: 25 };
    case "top-losers":     return { ...base, sortField: "daily",   sortDir: "asc",  topN: 25 };
    case "ytd-leaders":    return { ...base, sortField: "ytd",     sortDir: "desc", topN: 25 };
    case "five-year-compounders": return { ...base, sortField: "five_y", sortDir: "desc", topN: 25 };
    case "most-active":    return { ...base, sortField: "volume",  sortDir: "desc", topN: 25 };
    case "52w-highs":      return { ...base, sortField: "fiftyTwoWeekHighChangePct", sortDir: "desc", only52wHigh: true, topN: 25 };
    case "starred":        return { ...base, onlyStarred: true };
    default: return base;
  }
}

function PctCell({ value }: { value: number | null }) {
  return (
    <span className={cn("font-mono tabular-nums text-xs", pctColor(value))}>
      {fmtPct(value)}
    </span>
  );
}

function SortHeader({
  label, field, current, dir, onSort, title,
}: {
  label: string; field: SortField; current: SortField; dir: SortDirection;
  onSort: (f: SortField) => void; title?: string;
}) {
  const active = current === field;
  return (
    <th
      className="px-2 py-2 text-left text-xs font-semibold text-slate-400 cursor-pointer select-none whitespace-nowrap hover:text-white transition-colors"
      onClick={() => onSort(field)}
      title={title}
    >
      {label}
      {active && <span className="ml-0.5 text-blue-400">{dir === "desc" ? "↓" : "↑"}</span>}
    </th>
  );
}

function SkeletonRow() {
  return (
    <tr className="border-b border-slate-800 animate-pulse">
      {Array.from({ length: 17 }).map((_, i) => (
        <td key={i} className="px-2 py-3">
          <div className="h-3 bg-slate-800 rounded w-full" />
        </td>
      ))}
    </tr>
  );
}

/** Formats volume vs 20D average as a ratio string. */
function fmtVolRatio(volume: number | null, avg: number | null): string {
  if (!volume || !avg || avg === 0) return "—";
  return `${(volume / avg).toFixed(1)}×`;
}

function volRatioColor(volume: number | null, avg: number | null): string {
  if (!volume || !avg || avg === 0) return "text-slate-500";
  const r = volume / avg;
  if (r >= 3) return "text-amber-400 font-bold";
  if (r >= 2) return "text-amber-500";
  if (r >= 1.5) return "text-yellow-600";
  return "text-slate-400";
}

interface Props {
  onSelectStock: (s: StockData) => void;
  onDataLoaded?: (data: StockData[]) => void;
}

export default function ScreenerTable({ onSelectStock, onDataLoaded }: Props) {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [searchInput, setSearchInput] = useState("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [countdown, setCountdown] = useState<string>("");
  const [starred, setStarred] = useState<Set<string>>(new Set());
  const nextRefreshRef = useRef<number>(Date.now() + REFRESH_INTERVAL);

  // Load starred list from localStorage
  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem("starred") ?? "[]") as string[];
      setStarred(new Set(stored));
    } catch { /* ignore */ }
  }, []);

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
    const result = applyFilters(data, filters, starred);
    onDataLoaded?.(result);
    return result;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, filters, starred]);

  const handleSort = useCallback((field: SortField) => {
    setFilters((f) => ({
      ...f,
      sortField: field,
      sortDir: f.sortField === field && f.sortDir === "desc" ? "asc" : "desc",
    }));
  }, []);

  const toggleStar = useCallback((symbol: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setStarred((prev) => {
      const next = new Set(prev);
      if (next.has(symbol)) next.delete(symbol);
      else next.add(symbol);
      localStorage.setItem("starred", JSON.stringify([...next]));
      return next;
    });
  }, []);

  const exportCSV = useCallback(() => {
    if (!rows.length) return;
    const headers = ["#","Ticker","Company","Market","Sector","Price","Currency","1D%","WTD%","MTD%","YTD%","1Y%","3Y%","5Y%","52WH%","MarketCap","Volume","Vol/Avg","Timestamp"];
    const lines = rows.map((r, i) => [
      i + 1, r.symbol, `"${r.name}"`, r.market, r.sector,
      r.price ?? "", r.currency,
      r.performance.daily ?? "N/A",
      r.performance.wtd ?? "N/A",
      r.performance.mtd ?? "N/A",
      r.performance.ytd ?? "N/A",
      r.performance.one_y ?? "N/A",
      r.performance.three_y ?? "N/A",
      r.performance.five_y ?? "N/A",
      r.fiftyTwoWeekHighChangePct ?? "N/A",
      r.marketCap ?? "N/A",
      r.volume ?? "N/A",
      fmtVolRatio(r.volume, r.avgVolume20d),
      new Date().toISOString(),
    ].join(","));
    const csv = [headers.join(","), ...lines].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `screener_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [rows]);

  const statusDot = isError ? "bg-red-500" : isRefetching ? "bg-amber-400 animate-pulse" : "bg-emerald-400";
  const isVirtual = rows.length > VIRTUAL_THRESHOLD;

  return (
    <div className="flex flex-col gap-3">
      {/* Filter bar + status */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-slate-900 rounded-xl border border-slate-700 px-4">
        <FilterBar
          filters={filters}
          onChange={setFilters}
          onPreset={(p) => setFilters(applyPreset(p))}
          searchInput={searchInput}
          onSearchChange={setSearchInput}
        />
        <div className="flex items-center gap-3 text-xs text-slate-400 py-2 pl-4 border-l border-slate-700 shrink-0">
          <span className={cn("w-2 h-2 rounded-full", statusDot)} />
          {lastUpdated && <span>Updated {lastUpdated.toLocaleTimeString()}</span>}
          <span>Next in {countdown}</span>
          <button
            onClick={exportCSV}
            className="px-2 py-1 rounded border border-slate-700 hover:border-slate-500 hover:text-white transition-colors"
          >
            ↓ CSV
          </button>
        </div>
      </div>

      {isError && (
        <div className="bg-amber-900/40 border border-amber-700 rounded-lg px-4 py-2 text-amber-300 text-sm">
          Data source unavailable — showing last cached values. Retrying...
        </div>
      )}

      {/* Desktop table */}
      <div className="hidden md:block rounded-xl border border-slate-700 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-900 sticky top-0 z-10">
              <tr className="border-b border-slate-700">
                <th className="px-2 py-2 text-left text-xs text-slate-600 w-6">☆</th>
                <th className="px-2 py-2 text-left text-xs text-slate-500 w-7">#</th>
                <th className="px-2 py-2 text-left text-xs text-slate-400 whitespace-nowrap">Ticker</th>
                <th className="px-2 py-2 text-left text-xs text-slate-400">Company</th>
                <th className="px-2 py-2 text-left text-xs text-slate-400">Mkt</th>
                <th className="px-2 py-2 text-left text-xs text-slate-400">Sector</th>
                <SortHeader label="Price"    field="price"    current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="1D %"     field="daily"    current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="WTD %"    field="wtd"      current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="MTD %"    field="mtd"      current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="YTD %"    field="ytd"      current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="1Y %"     field="one_y"    current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="3Y %*"    field="three_y"  current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="5Y %*"    field="five_y"   current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <SortHeader label="52W H %"  field="fiftyTwoWeekHighChangePct" current={filters.sortField} dir={filters.sortDir} onSort={handleSort} title="% from 52-week high" />
                <th className="px-2 py-2 text-left text-xs text-slate-400 whitespace-nowrap">Vol/Avg</th>
                <SortHeader label="Mkt Cap"  field="marketCap" current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />
                <th className="px-2 py-2 text-left text-xs text-slate-400">Chart</th>
              </tr>
            </thead>
            <tbody>
              {isLoading
                ? Array.from({ length: 12 }).map((_, i) => <SkeletonRow key={i} />)
                : rows.map((stock, idx) => (
                    <tr
                      key={stock.symbol}
                      onClick={() => onSelectStock(stock)}
                      className={cn(
                        "border-b border-slate-800 cursor-pointer transition-colors",
                        "hover:bg-slate-800/60",
                        isRefetching && "opacity-70"
                      )}
                    >
                      <td className="px-2 py-2">
                        <button
                          onClick={(e) => toggleStar(stock.symbol, e)}
                          className="text-sm leading-none hover:scale-125 transition-transform"
                          title={starred.has(stock.symbol) ? "Unstar" : "Star"}
                        >
                          {starred.has(stock.symbol) ? "⭐" : "☆"}
                        </button>
                      </td>
                      <td className="px-2 py-2 text-slate-600 tabular-nums text-xs">{idx + 1}</td>
                      <td className="px-2 py-2 font-semibold text-white whitespace-nowrap text-xs">
                        {stock.symbol}
                        {stock.isStale && <span className="ml-1 text-xs text-amber-500 font-normal">STALE</span>}
                      </td>
                      <td className="px-2 py-2 text-slate-300 max-w-[160px] truncate text-xs">{stock.name}</td>
                      <td className="px-2 py-2 text-slate-400 whitespace-nowrap text-xs">
                        {MARKET_FLAG[stock.market]}
                      </td>
                      <td className="px-2 py-2 text-slate-500 text-xs whitespace-nowrap">{stock.sector}</td>
                      <td className="px-2 py-2 text-white font-mono tabular-nums whitespace-nowrap text-xs">
                        {stock.price !== null
                          ? stock.price.toLocaleString(undefined, { maximumFractionDigits: 4 })
                          : "N/A"}
                        <span className="text-slate-700 ml-0.5">{stock.currency}</span>
                      </td>
                      <td className="px-2 py-2"><PctCell value={stock.performance.daily} /></td>
                      <td className="px-2 py-2"><PctCell value={stock.performance.wtd} /></td>
                      <td className="px-2 py-2"><PctCell value={stock.performance.mtd} /></td>
                      <td className="px-2 py-2"><PctCell value={stock.performance.ytd} /></td>
                      <td className="px-2 py-2"><PctCell value={stock.performance.one_y} /></td>
                      <td className="px-2 py-2"><PctCell value={stock.performance.three_y} /></td>
                      <td className="px-2 py-2"><PctCell value={stock.performance.five_y} /></td>
                      <td className="px-2 py-2">
                        <span className={cn("font-mono tabular-nums text-xs", pctColor(stock.fiftyTwoWeekHighChangePct))}>
                          {fmtPct(stock.fiftyTwoWeekHighChangePct)}
                        </span>
                      </td>
                      <td className={cn("px-2 py-2 font-mono tabular-nums text-xs", volRatioColor(stock.volume, stock.avgVolume20d))}>
                        {fmtVolRatio(stock.volume, stock.avgVolume20d)}
                      </td>
                      <td className="px-2 py-2 text-slate-400 tabular-nums text-xs">
                        {fmtMarketCap(stock.marketCap)}
                      </td>
                      <td className="px-1 py-1">
                        {!isVirtual && <MiniSparkline tvSymbol={stock.tvSymbol} />}
                      </td>
                    </tr>
                  ))}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-2 bg-slate-900 border-t border-slate-800 flex justify-between text-xs text-slate-600">
          <span>
            {rows.length} stocks · *3Y and 5Y are cumulative · 52W H% = distance from 52-week high
          </span>
          <span>Prices delayed up to 15 minutes</span>
        </div>
      </div>

      {/* Mobile card grid */}
      <div className="md:hidden grid grid-cols-1 sm:grid-cols-2 gap-3">
        {isLoading
          ? Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="rounded-xl border border-slate-700 bg-slate-900 p-4 animate-pulse h-32" />
            ))
          : rows.map((stock) => (
              <div
                key={stock.symbol}
                onClick={() => onSelectStock(stock)}
                className="rounded-xl border border-slate-700 bg-slate-900 p-4 cursor-pointer hover:border-slate-500 transition-colors"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <span className="font-bold text-white text-sm">{stock.symbol}</span>
                    <span className="ml-2 text-slate-500 text-xs">{MARKET_FLAG[stock.market]}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => toggleStar(stock.symbol, e)}
                      className="text-sm"
                    >
                      {starred.has(stock.symbol) ? "⭐" : "☆"}
                    </button>
                    <span className={cn("font-mono font-bold text-sm tabular-nums", pctColor(stock.performance.daily))}>
                      {fmtPct(stock.performance.daily)}
                    </span>
                  </div>
                </div>
                <div className="text-slate-400 text-xs truncate mb-2">{stock.name}</div>
                <div className="flex justify-between text-xs">
                  <span className="text-white font-mono tabular-nums">
                    {stock.price?.toLocaleString(undefined, { maximumFractionDigits: 4 }) ?? "N/A"}
                    <span className="text-slate-600 ml-0.5">{stock.currency}</span>
                  </span>
                  <span className={cn("font-mono tabular-nums", pctColor(stock.performance.ytd))}>
                    YTD {fmtPct(stock.performance.ytd)}
                  </span>
                  <span className="text-slate-500">{fmtMarketCap(stock.marketCap)}</span>
                </div>
              </div>
            ))}
      </div>
    </div>
  );
}
