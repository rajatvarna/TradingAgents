"use client";
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

interface Props {
  primarySymbol: string;
  primaryYahooSuffix: string;
}

interface HistoryPoint {
  date: string;
  close: number;
}

const COLORS = ["#3b82f6", "#10b981", "#f59e0b"];
const LABELS = ["Primary", "Compare 1", "Compare 2"];

function normalizeToPercent(series: HistoryPoint[], baseDate: string): { date: string; pct: number }[] {
  const baseIdx = series.findIndex((p) => p.date >= baseDate);
  if (baseIdx < 0) return [];
  const baseClose = series[baseIdx].close;
  if (!baseClose) return [];
  return series.slice(baseIdx).map((p) => ({
    date: p.date,
    pct: ((p.close - baseClose) / baseClose) * 100,
  }));
}

function last252(series: HistoryPoint[]): HistoryPoint[] {
  return series.slice(-252);
}

export default function ComparisonChart({ primarySymbol, primaryYahooSuffix }: Props) {
  const [extraSymbols, setExtraSymbols] = useState<string[]>([]);
  const [inputVal, setInputVal] = useState("");

  const symbols = [primaryYahooSuffix, ...extraSymbols];

  const queries = [
    useQuery<HistoryPoint[]>({
      queryKey: ["hist-compare", symbols[0]],
      queryFn: async () => {
        const res = await fetch(`/api/history?symbol=${encodeURIComponent(symbols[0])}`);
        return res.json();
      },
      staleTime: 3600 * 1000,
      enabled: !!symbols[0],
    }),
    useQuery<HistoryPoint[]>({
      queryKey: ["hist-compare", symbols[1] ?? "__none1__"],
      queryFn: async () => {
        const res = await fetch(`/api/history?symbol=${encodeURIComponent(symbols[1])}`);
        return res.json();
      },
      staleTime: 3600 * 1000,
      enabled: !!symbols[1],
    }),
    useQuery<HistoryPoint[]>({
      queryKey: ["hist-compare", symbols[2] ?? "__none2__"],
      queryFn: async () => {
        const res = await fetch(`/api/history?symbol=${encodeURIComponent(symbols[2])}`);
        return res.json();
      },
      staleTime: 3600 * 1000,
      enabled: !!symbols[2],
    }),
  ];

  const addSymbol = () => {
    const s = inputVal.trim().toUpperCase();
    if (!s || extraSymbols.includes(s) || s === primaryYahooSuffix) return;
    if (extraSymbols.length >= 2) return;
    setExtraSymbols((prev) => [...prev, s]);
    setInputVal("");
  };

  const removeExtra = (sym: string) => {
    setExtraSymbols((prev) => prev.filter((s) => s !== sym));
  };

  const chartData = useMemo(() => {
    const activeSeries: { label: string; color: string; data: HistoryPoint[] }[] = [];
    for (let i = 0; i < symbols.length; i++) {
      const d = queries[i].data;
      if (d && d.length > 0) {
        activeSeries.push({ label: i === 0 ? primarySymbol : symbols[i], color: COLORS[i], data: last252(d) });
      }
    }
    if (activeSeries.length === 0) return null;

    const baseDate = activeSeries.reduce((latest, s) => {
      const first = s.data[0]?.date ?? "";
      return first > latest ? first : latest;
    }, activeSeries[0].data[0]?.date ?? "");

    const normalized = activeSeries.map((s) => ({
      label: s.label,
      color: s.color,
      series: normalizeToPercent(s.data, baseDate),
    }));

    const allDates = normalized[0].series.map((p) => p.date);
    const allPcts = normalized.flatMap((n) => n.series.map((p) => p.pct));
    const minPct = Math.min(0, ...allPcts);
    const maxPct = Math.max(0, ...allPcts);
    const range = maxPct - minPct || 1;

    return { normalized, allDates, minPct, maxPct, range };
  }, [queries[0].data, queries[1].data, queries[2].data, primarySymbol]);

  const W = 600;
  const H = 200;
  const PAD_LEFT = 48;
  const PAD_RIGHT = 40;
  const PAD_TOP = 10;
  const PAD_BOTTOM = 20;
  const chartW = W - PAD_LEFT - PAD_RIGHT;
  const chartH = H - PAD_TOP - PAD_BOTTOM;

  const toX = (i: number, total: number) =>
    PAD_LEFT + (i / Math.max(1, total - 1)) * chartW;

  const toY = (pct: number, minPct: number, range: number) =>
    PAD_TOP + chartH - ((pct - minPct) / range) * chartH;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={inputVal}
          onChange={(e) => setInputVal(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addSymbol()}
          placeholder="Add symbol (e.g. MSFT)"
          className="px-2 py-1 bg-slate-800 border border-slate-600 rounded text-sm text-white placeholder-slate-500 w-36 focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={addSymbol}
          disabled={extraSymbols.length >= 2}
          className="px-3 py-1 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded transition-colors"
        >
          Add
        </button>
        {extraSymbols.map((sym) => (
          <span key={sym} className="flex items-center gap-1 bg-slate-700 text-slate-300 text-xs px-2 py-1 rounded">
            {sym}
            <button onClick={() => removeExtra(sym)} className="text-slate-400 hover:text-white ml-1">×</button>
          </span>
        ))}
      </div>

      <div className="flex items-center gap-4 text-xs">
        {symbols.slice(0, symbols.length).map((sym, i) => (
          <span key={sym} className="flex items-center gap-1.5">
            <span className="w-4 h-0.5 inline-block" style={{ backgroundColor: COLORS[i] }} />
            <span className="text-slate-300">{i === 0 ? primarySymbol : sym}</span>
          </span>
        ))}
      </div>

      {!chartData ? (
        <div className="h-[200px] flex items-center justify-center text-slate-500 text-sm">Loading…</div>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 200 }}>
          {[-20, -10, 0, 10, 20].map((pctLine) => {
            if (pctLine < chartData.minPct - 5 || pctLine > chartData.maxPct + 5) return null;
            const y = toY(pctLine, chartData.minPct, chartData.range);
            return (
              <g key={pctLine}>
                <line x1={PAD_LEFT} x2={W - PAD_RIGHT} y1={y} y2={y} stroke="#334155" strokeWidth="0.5" />
                <text x={PAD_LEFT - 4} y={y + 3} textAnchor="end" fontSize="9" fill="#64748b">
                  {pctLine > 0 ? "+" : ""}{pctLine}%
                </text>
              </g>
            );
          })}

          {chartData.normalized.map(({ label, color, series }) => {
            if (series.length < 2) return null;
            const points = series.map((p, i) =>
              `${toX(i, series.length).toFixed(1)},${toY(p.pct, chartData.minPct, chartData.range).toFixed(1)}`
            ).join(" ");
            const lastPt = series[series.length - 1];
            const lx = toX(series.length - 1, series.length);
            const ly = toY(lastPt.pct, chartData.minPct, chartData.range);
            return (
              <g key={label}>
                <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
                <text x={lx + 3} y={ly + 3} fontSize="9" fill={color}>{lastPt.pct >= 0 ? "+" : ""}{lastPt.pct.toFixed(1)}%</text>
              </g>
            );
          })}
        </svg>
      )}
      <p className="text-xs text-slate-600">Normalized returns from earliest common date · Last 252 trading days</p>
    </div>
  );
}
