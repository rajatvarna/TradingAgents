"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { StockData, FilterState, SortField, SortDirection, Market, PresetName } from "@/types";
import { fmtPct, fmtMarketCap, pctColor, MARKET_FLAG, cn } from "@/lib/utils";
import FilterBar from "./FilterBar";
import MiniSparkline from "./MiniSparkline";
import * as Tooltip from "@radix-ui/react-tooltip";

interface FxRates {
  USD: number;
  INR: number;
  AED: number;
  SAR: number;
}

type DisplayCurrency = "local" | "USD";

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

// ─── Column definitions ────────────────────────────────────────────────────
interface ColumnDef {
  id: string;
  label: string;
  defaultVisible: boolean;
}

const COLUMNS: ColumnDef[] = [
  { id: "star",      label: "★",          defaultVisible: true },
  { id: "rank",      label: "#",          defaultVisible: true },
  { id: "ticker",    label: "Ticker",     defaultVisible: true },
  { id: "company",   label: "Company",    defaultVisible: true },
  { id: "market",    label: "Market",     defaultVisible: true },
  { id: "price",     label: "Price",      defaultVisible: true },
  { id: "daily",     label: "1D %",       defaultVisible: true },
  { id: "wtd",       label: "WTD %",      defaultVisible: true },
  { id: "mtd",       label: "MTD %",      defaultVisible: true },
  { id: "ytd",       label: "YTD %",      defaultVisible: true },
  { id: "one_y",     label: "1Y %",       defaultVisible: true },
  { id: "three_y",   label: "3Y %",       defaultVisible: true },
  { id: "five_y",    label: "5Y %",       defaultVisible: true },
  { id: "rs",        label: "RS",         defaultVisible: true },
  { id: "marketCap", label: "Mkt Cap",    defaultVisible: true },
  { id: "52w-range", label: "52W Range",  defaultVisible: false },
  { id: "chart",     label: "Chart (1M)", defaultVisible: true },
];

const STORAGE_KEY = "screener-columns";

function loadVisibleColumns(): Set<string> {
  if (typeof window === "undefined") {
    return new Set(COLUMNS.filter((c) => c.defaultVisible).map((c) => c.id));
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const arr: string[] = JSON.parse(raw);
      return new Set(arr);
    }
  } catch {
    // ignore parse errors
  }
  return new Set(COLUMNS.filter((c) => c.defaultVisible).map((c) => c.id));
}

// ─── Filters / sorting ────────────────────────────────────────────────────

function applyFilters(data: StockData[], f: FilterState, rsMap: Map<string, number>, favorites: Set<string>): StockData[] {
  let rows = data.filter((d) => f.markets.includes(d.market));

  if (f.sectors.length) rows = rows.filter((d) => f.sectors.includes(d.sector));

  if (f.watchlistOnly) rows = rows.filter((d) => favorites.has(d.symbol));

  if (f.volSurge) {
    rows = rows.filter(
      (d) => d.volume !== null && d.avgVolume20d !== null && d.volume > 2 * d.avgVolume20d
    );
  }

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
      if (f.sortField === "volume")    return d.volume ?? -Infinity;
      if (f.sortField === "rs")        return rsMap.get(d.symbol) ?? -Infinity;
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
    case "vol-surge":      return { ...base, sortField: "volume",  sortDir: "desc", topN: 25, volSurge: true };
    case "watchlist":      return { ...base, watchlistOnly: true, topN: null };
    default: return base;
  }
}

// ─── RS score computation ─────────────────────────────────────────────────

function computeRsMap(data: StockData[]): Map<string, number> {
  const withVal = data
    .map((d) => ({ symbol: d.symbol, val: d.performance.one_y }))
    .filter((x): x is { symbol: string; val: number } => x.val !== null);

  // Sort ascending to get ranks
  const sorted = [...withVal].sort((a, b) => a.val - b.val);
  const n = sorted.length;

  const map = new Map<string, number>();
  sorted.forEach((x, i) => {
    map.set(x.symbol, n === 1 ? 100 : Math.round((i / (n - 1)) * 100));
  });
  return map;
}

function rsColor(score: number): string {
  if (score >= 90) return "text-emerald-400 font-bold";
  if (score >= 70) return "text-blue-400";
  if (score >= 50) return "text-amber-400";
  return "text-slate-400";
}

// ─── Sub-components ───────────────────────────────────────────────────────

function PctCell({ value }: { value: number | null }) {
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

function SkeletonRow({ colCount }: { colCount: number }) {
  return (
    <tr className="border-b border-slate-800 animate-pulse">
      {Array.from({ length: colCount }).map((_, i) => (
        <td key={i} className="px-3 py-3">
          <div className="h-3 bg-slate-800 rounded w-full" />
        </td>
      ))}
    </tr>
  );
}

// ─── Column customizer dropdown ───────────────────────────────────────────

function ColumnCustomizer({
  visibleColumns,
  onChange,
}: {
  visibleColumns: Set<string>;
  onChange: (next: Set<string>) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const toggle = (id: string) => {
    const next = new Set(visibleColumns);
    if (next.has(id)) {
      if (next.size <= 1) return; // always keep at least 1 column
      next.delete(id);
    } else {
      next.add(id);
    }
    onChange(next);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(next))); } catch { /* ignore */ }
  };

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="px-2 py-1 rounded border border-slate-700 hover:border-slate-500 hover:text-white transition-colors text-xs text-slate-400"
      >
        Columns ▼
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-2 min-w-[160px]">
          {COLUMNS.map((col) => (
            <label key={col.id} className="flex items-center gap-2 px-2 py-1 cursor-pointer hover:bg-slate-700 rounded text-xs text-slate-300">
              <input
                type="checkbox"
                checked={visibleColumns.has(col.id)}
                onChange={() => toggle(col.id)}
                className="accent-blue-500"
              />
              {col.label}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Sparkline tooltip ───────────────────────────────────────────────────

function SparklineSvg({ closes }: { closes: Array<{ date: string; close: number }> }) {
  const last90 = closes.slice(-90);
  if (last90.length < 2) return <span className="text-slate-500 text-xs">No data</span>;
  const values = last90.map((c) => c.close);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const W = 200; const H = 60;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * W},${H - ((v - min) / range) * H}`).join(" ");
  const rising = values[values.length - 1] >= values[0];
  const color = rising ? "#34d399" : "#f87171";
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

function RowTooltip({ stock, children }: { stock: StockData; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const { data: history } = useQuery<Array<{ date: string; close: number }>>({
    queryKey: ["hist-tooltip", stock.yahooSuffix],
    queryFn: async () => {
      const res = await fetch(`/api/history?symbol=${encodeURIComponent(stock.yahooSuffix)}`);
      return res.json();
    },
    enabled: open,
    staleTime: 3600_000,
  });

  return (
    <Tooltip.Root open={open} onOpenChange={setOpen}>
      <Tooltip.Trigger asChild>{children as React.ReactElement}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          side="right"
          sideOffset={8}
          className="z-50 bg-slate-800 border border-slate-600 rounded-xl p-3 shadow-2xl"
        >
          <div className="text-xs text-slate-400 mb-1 font-semibold">{stock.symbol} · 90-day</div>
          {history ? <SparklineSvg closes={history} /> : <div className="w-[200px] h-[60px] bg-slate-700 animate-pulse rounded" />}
          <Tooltip.Arrow className="fill-slate-800" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

// ─── Main component ───────────────────────────────────────────────────────

interface Props {
  onSelectStock: (s: StockData) => void;
  onDataLoaded?: (stocks: StockData[]) => void;
}

export default function ScreenerTable({ onSelectStock, onDataLoaded }: Props) {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [searchInput, setSearchInput] = useState("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [countdown, setCountdown] = useState<string>("");
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(loadVisibleColumns);
  const [displayCurrency, setDisplayCurrency] = useState<DisplayCurrency>("local");
  const [shareCopied, setShareCopied] = useState(false);
  const [favorites, setFavorites] = useState<Set<string>>(() => {
    if (typeof window === "undefined") return new Set();
    try {
      const raw = localStorage.getItem("favorites");
      return raw ? new Set(JSON.parse(raw) as string[]) : new Set();
    } catch { return new Set(); }
  });
  const nextRefreshRef = useRef<number>(Date.now() + REFRESH_INTERVAL);

  const toggleFavorite = useCallback((symbol: string) => {
    setFavorites((prev) => {
      const next = new Set(prev);
      if (next.has(symbol)) next.delete(symbol);
      else next.add(symbol);
      try { localStorage.setItem("favorites", JSON.stringify(Array.from(next))); } catch { /* ignore */ }
      return next;
    });
  }, []);

  const { data: fxRates } = useQuery<FxRates>({
    queryKey: ["fx"],
    queryFn: async () => {
      const res = await fetch("/api/fx");
      return res.json() as Promise<FxRates>;
    },
    staleTime: 60 * 60 * 1000,
  });

  // Debounce search
  useEffect(() => {
    const t = setTimeout(() => setFilters((f) => ({ ...f, search: searchInput })), 300);
    return () => clearTimeout(t);
  }, [searchInput]);

  // Initialize filters from URL on mount
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const market = params.get("market");
    const sector = params.get("sector");
    const sort = params.get("sort") as SortField | null;
    const dir = params.get("dir") as SortDirection | null;
    const topN = params.get("topN");
    const search = params.get("search");
    const preset = params.get("preset") as PresetName | null;

    if (!market && !sort && !search && !preset) return;

    if (preset) {
      setFilters(applyPreset(preset));
      return;
    }

    setFilters((f) => ({
      ...f,
      markets: market ? (market.split(",") as Market[]) : f.markets,
      sectors: sector ? sector.split(",") : f.sectors,
      sortField: sort ?? f.sortField,
      sortDir: dir ?? f.sortDir,
      topN: topN ? Number(topN) : f.topN,
      search: search ?? f.search,
    }));
  }, []); // run once on mount

  // Sync filters to URL
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams();
    if (filters.markets.length < 4) params.set("market", filters.markets.join(","));
    if (filters.sectors.length) params.set("sector", filters.sectors.join(","));
    if (filters.sortField !== "daily") params.set("sort", filters.sortField);
    if (filters.sortDir !== "desc") params.set("dir", filters.sortDir);
    if (filters.topN) params.set("topN", String(filters.topN));
    if (filters.search) params.set("search", filters.search);
    const qs = params.toString();
    window.history.replaceState(null, "", qs ? `?${qs}` : window.location.pathname);
  }, [filters]);

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

  useEffect(() => {
    if (data && onDataLoaded) {
      onDataLoaded(data);
    }
  }, [data, onDataLoaded]);

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

  // RS score map (computed over full dataset, before filters)
  const rsMap = useMemo(() => {
    if (!data) return new Map<string, number>();
    return computeRsMap(data);
  }, [data]);

  const rows = useMemo(() => {
    if (!data) return [];
    return applyFilters(data, filters, rsMap, favorites);
  }, [data, filters, rsMap, favorites]);

  // Keyboard navigation
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "SELECT") return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIdx((i) => (i === null ? 0 : Math.min(i + 1, rows.length - 1)));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIdx((i) => (i === null ? 0 : Math.max(i - 1, 0)));
      } else if (e.key === "Enter") {
        if (selectedIdx !== null && rows[selectedIdx]) {
          onSelectStock(rows[selectedIdx]);
        }
      } else if (e.key === "Escape") {
        setSelectedIdx(null);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [rows, selectedIdx, onSelectStock]);

  // Clear selection when rows change
  useEffect(() => {
    setSelectedIdx(null);
  }, [rows]);

  const handleSort = useCallback((field: SortField) => {
    setFilters((f) => ({
      ...f,
      sortField: field,
      sortDir: f.sortField === field && f.sortDir === "desc" ? "asc" : "desc",
    }));
  }, []);

  const exportCSV = useCallback(() => {
    if (!rows.length) return;
    const headers = ["#","Ticker","Company","Market","Price","Currency","Daily%","WTD%","MTD%","YTD%","1Y%","3Y%","5Y%","RS","MarketCap","Volume","Timestamp"];
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
      rsMap.get(r.symbol) ?? "N/A",
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
  }, [rows, rsMap]);

  const statusDot = isError
    ? "bg-red-500"
    : isRefetching
    ? "bg-amber-400 animate-pulse"
    : "bg-emerald-400";

  const isVirtual = rows.length > VIRTUAL_THRESHOLD;
  const vis = visibleColumns;
  const visibleColDefs = COLUMNS.filter((c) => vis.has(c.id));

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
          <ColumnCustomizer visibleColumns={vis} onChange={setVisibleColumns} />
          <button
            onClick={() => setDisplayCurrency((c) => (c === "local" ? "USD" : "local"))}
            className={cn(
              "px-2 py-1 rounded border transition-colors text-xs",
              displayCurrency === "USD"
                ? "border-blue-500 text-blue-400"
                : "border-slate-700 text-slate-400 hover:border-slate-500 hover:text-white"
            )}
            title="Toggle price display currency"
          >
            {displayCurrency === "USD" ? "USD" : "Local"}
          </button>
          <button
            onClick={() => {
              navigator.clipboard.writeText(window.location.href).catch(() => {});
              setShareCopied(true);
              setTimeout(() => setShareCopied(false), 2000);
            }}
            className="px-2 py-1 rounded border border-slate-700 hover:border-slate-500 hover:text-white transition-colors text-xs"
            title="Copy shareable link"
          >
            {shareCopied ? "✓ Copied" : "🔗 Share"}
          </button>
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
                {vis.has("star")      && <th className="px-2 py-2 w-8"></th>}
                {vis.has("rank")      && <th className="px-3 py-2 text-left text-xs text-slate-500 w-8">#</th>}
                {vis.has("ticker")    && <th className="px-3 py-2 text-left text-xs text-slate-400 whitespace-nowrap">Ticker</th>}
                {vis.has("company")   && <th className="px-3 py-2 text-left text-xs text-slate-400">Company</th>}
                {vis.has("market")    && <th className="px-3 py-2 text-left text-xs text-slate-400">Market</th>}
                {vis.has("price")     && <SortHeader label="Price"    field="price"     current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("daily")     && <SortHeader label="1D %"     field="daily"     current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("wtd")       && <SortHeader label="WTD %"    field="wtd"       current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("mtd")       && <SortHeader label="MTD %"    field="mtd"       current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("ytd")       && <SortHeader label="YTD %"    field="ytd"       current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("one_y")     && <SortHeader label="1Y %"     field="one_y"     current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("three_y")   && <SortHeader label="3Y %*"    field="three_y"   current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("five_y")    && <SortHeader label="5Y %*"    field="five_y"    current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("rs")        && <SortHeader label="RS"       field="rs"        current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("marketCap") && <SortHeader label="Mkt Cap"  field="marketCap" current={filters.sortField} dir={filters.sortDir} onSort={handleSort} />}
                {vis.has("52w-range") && <th className="px-3 py-2 text-left text-xs text-slate-400 whitespace-nowrap">52W Range</th>}
                {vis.has("chart")     && <th className="px-3 py-2 text-left text-xs text-slate-400">Chart (1M)</th>}
              </tr>
            </thead>
            <tbody>
              {isLoading
                ? Array.from({ length: 10 }).map((_, i) => (
                    <SkeletonRow key={i} colCount={visibleColDefs.length} />
                  ))
                : rows.map((stock, idx) => {
                    const rs = rsMap.get(stock.symbol) ?? null;
                    const isVolSurge =
                      stock.volume !== null &&
                      stock.avgVolume20d !== null &&
                      stock.volume > 2 * stock.avgVolume20d;
                    const isSelected = selectedIdx === idx;

                    return (
                      <RowTooltip key={stock.symbol} stock={stock}>
                      <tr
                        onClick={() => { setSelectedIdx(idx); onSelectStock(stock); }}
                        className={cn(
                          "border-b border-slate-800 cursor-pointer transition-colors",
                          "hover:bg-slate-800/60",
                          isRefetching && "opacity-70",
                          stock.isStale && "opacity-60",
                          isSelected && "bg-blue-900/30 ring-1 ring-blue-500 ring-inset"
                        )}
                      >
                        {vis.has("star") && (
                          <td className="px-2 py-2 w-8">
                            <button
                              onClick={(e) => { e.stopPropagation(); toggleFavorite(stock.symbol); }}
                              className="text-base leading-none hover:scale-125 transition-transform"
                              title={favorites.has(stock.symbol) ? "Remove from watchlist" : "Add to watchlist"}
                            >
                              {favorites.has(stock.symbol) ? "★" : "☆"}
                            </button>
                          </td>
                        )}
                        {vis.has("rank")    && <td className="px-3 py-2 text-slate-600 tabular-nums">{idx + 1}</td>}
                        {vis.has("ticker")  && (
                          <td className="px-3 py-2 font-semibold text-white whitespace-nowrap">
                            {stock.symbol}
                            {isVolSurge && <span className="ml-1" title="Volume surge (>2× avg)">🔥</span>}
                            {stock.isStale && (
                              <span className="ml-1 text-xs text-amber-500 font-normal">STALE</span>
                            )}
                          </td>
                        )}
                        {vis.has("company")   && <td className="px-3 py-2 text-slate-300 max-w-[180px] truncate">{stock.name}</td>}
                        {vis.has("market")    && (
                          <td className="px-3 py-2 text-slate-400 whitespace-nowrap">
                            {MARKET_FLAG[stock.market]} {stock.market}
                          </td>
                        )}
                        {vis.has("price")     && (() => {
                          let displayPrice = stock.price;
                          let displayCurr = stock.currency;
                          if (
                            displayCurrency === "USD" &&
                            stock.price !== null &&
                            fxRates &&
                            stock.currency !== "USD"
                          ) {
                            const rate =
                              stock.currency === "INR" ? fxRates.INR
                              : stock.currency === "AED" ? fxRates.AED
                              : stock.currency === "SAR" ? fxRates.SAR
                              : 1;
                            displayPrice = stock.price / rate;
                            displayCurr = "USD";
                          }
                          return (
                            <td className="px-3 py-2 text-white font-mono tabular-nums whitespace-nowrap">
                              {displayPrice !== null
                                ? (displayCurr === "USD" ? "$" : "") +
                                  displayPrice.toLocaleString(undefined, { maximumFractionDigits: 4 })
                                : "N/A"}
                              {" "}
                              <span className="text-slate-600 text-xs">
                                {displayCurr !== "USD" ? displayCurr : ""}
                              </span>
                            </td>
                          );
                        })()}
                        {vis.has("daily")     && <td className="px-3 py-2"><PctCell value={stock.performance.daily} /></td>}
                        {vis.has("wtd")       && <td className="px-3 py-2"><PctCell value={stock.performance.wtd} /></td>}
                        {vis.has("mtd")       && <td className="px-3 py-2"><PctCell value={stock.performance.mtd} /></td>}
                        {vis.has("ytd")       && <td className="px-3 py-2"><PctCell value={stock.performance.ytd} /></td>}
                        {vis.has("one_y")     && <td className="px-3 py-2"><PctCell value={stock.performance.one_y} /></td>}
                        {vis.has("three_y")   && <td className="px-3 py-2"><PctCell value={stock.performance.three_y} /></td>}
                        {vis.has("five_y")    && <td className="px-3 py-2"><PctCell value={stock.performance.five_y} /></td>}
                        {vis.has("rs")        && (
                          <td className="px-3 py-2 tabular-nums text-sm font-mono">
                            {rs !== null
                              ? <span className={rsColor(rs)}>{rs}</span>
                              : <span className="text-slate-600">—</span>}
                          </td>
                        )}
                        {vis.has("marketCap") && (
                          <td className="px-3 py-2 text-slate-400 tabular-nums text-xs">
                            {fmtMarketCap(stock.marketCap)}
                          </td>
                        )}
                        {vis.has("52w-range") && (() => {
                          const lo = stock.fiftyTwoWeekLow;
                          const hi = stock.fiftyTwoWeekHigh;
                          const p  = stock.price;
                          if (lo === null || hi === null || p === null) {
                            return <td className="px-3 py-2 text-slate-600 text-xs">—</td>;
                          }
                          const range = hi - lo;
                          const pos = range > 0 ? Math.min(100, Math.max(0, ((p - lo) / range) * 100)) : 50;
                          const barColor = pos >= 80 ? "bg-emerald-500" : pos >= 40 ? "bg-amber-500" : "bg-red-500";
                          const pctFromHigh = ((p - hi) / hi * 100).toFixed(1);
                          return (
                            <td className="px-3 py-2">
                              <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden" title={`${pctFromHigh}% from 52W High`}>
                                <div className={cn("h-full rounded-full", barColor)} style={{ width: `${pos}%` }} />
                              </div>
                            </td>
                          );
                        })()}
                        {vis.has("chart")     && (
                          <td className="px-2 py-1">
                            {!isVirtual && <MiniSparkline tvSymbol={stock.tvSymbol} />}
                          </td>
                        )}
                      </tr>
                      </RowTooltip>
                    );
                  })}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-2 bg-slate-900 border-t border-slate-800 flex justify-between text-xs text-slate-600">
          <span>
            Showing {rows.length} stocks · *3Y and 5Y are cumulative total returns
            {selectedIdx !== null && ` · ↑↓ navigate · Enter to chart · Esc to clear`}
          </span>
          <span>Prices delayed up to 15 minutes</span>
        </div>
      </div>
    </div>
  );
}
