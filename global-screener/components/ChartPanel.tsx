"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { StockData } from "@/types";
import { fmtPct, fmtMarketCap, pctColor, MARKET_FLAG, cn } from "@/lib/utils";

interface Props {
  stock: StockData | null;
}

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    TradingView: any;
    tvWidget: ReturnType<typeof setTimeout> | null;
  }
}

type Tab = "chart" | "insider" | "fundamentals";

interface EarningsData {
  nextEarningsDate: string | null;
  epsEstimate: number | null;
}

interface InsiderTransaction {
  name: string;
  relation: string;
  transactionDate: string;
  transactionType: string;
  shares: number;
  value: number | null;
}

interface FundamentalsData {
  shortPercentOfFloat: number | null;
  shortRatio: number | null;
  forwardPE: number | null;
  pegRatio: number | null;
  priceToBook: number | null;
  returnOnEquity: number | null;
  debtToEquity: number | null;
}

function fmtNum(v: number | null, decimals = 2): string {
  if (v === null) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
}

function fmtPct2(v: number | null): string {
  if (v === null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function EarningsBadge({ symbol }: { symbol: string }) {
  const { data } = useQuery<EarningsData>({
    queryKey: ["earnings", symbol],
    queryFn: async () => {
      const res = await fetch(`/api/earnings?symbol=${encodeURIComponent(symbol)}`);
      return res.json() as Promise<EarningsData>;
    },
    staleTime: 60 * 60 * 1000,
  });

  if (!data?.nextEarningsDate) {
    return (
      <span className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded-full">
        Earnings: TBD
      </span>
    );
  }

  const date = new Date(data.nextEarningsDate);
  const now = new Date();
  const daysUntil = Math.ceil((date.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
  const label = date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const isClose = daysUntil >= 0 && daysUntil <= 7;

  return (
    <span
      className={cn(
        "text-xs px-2 py-0.5 rounded-full",
        isClose ? "bg-amber-700 text-amber-100" : "bg-slate-700 text-slate-300"
      )}
    >
      Earnings: {label}
    </span>
  );
}

function InsiderTab({ symbol }: { symbol: string }) {
  const { data, isLoading } = useQuery<InsiderTransaction[]>({
    queryKey: ["insider", symbol],
    queryFn: async () => {
      const res = await fetch(`/api/insider?symbol=${encodeURIComponent(symbol)}`);
      return res.json() as Promise<InsiderTransaction[]>;
    },
    staleTime: 60 * 60 * 1000,
  });

  if (isLoading) {
    return (
      <div className="p-4 animate-pulse space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-8 bg-slate-800 rounded" />
        ))}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return <div className="p-8 text-center text-slate-500 text-sm">No data</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-slate-900 border-b border-slate-700">
          <tr>
            {["Name", "Role", "Date", "Type", "Shares", "Value"].map((h) => (
              <th key={h} className="px-3 py-2 text-left text-xs font-semibold text-slate-400">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((tx, i) => {
            const isBuy = tx.transactionType.toLowerCase().includes("purchase") ||
              tx.transactionType.toLowerCase().includes("buy");
            const isSell = tx.transactionType.toLowerCase().includes("sale") ||
              tx.transactionType.toLowerCase().includes("sell");
            return (
              <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/40">
                <td className="px-3 py-2 text-white font-medium">{tx.name}</td>
                <td className="px-3 py-2 text-slate-400 text-xs">{tx.relation}</td>
                <td className="px-3 py-2 text-slate-400 font-mono text-xs">{tx.transactionDate}</td>
                <td className="px-3 py-2">
                  <span
                    className={cn(
                      "text-xs font-semibold px-1.5 py-0.5 rounded",
                      isBuy ? "bg-emerald-900/60 text-emerald-300" :
                      isSell ? "bg-red-900/60 text-red-300" :
                      "bg-slate-700 text-slate-300"
                    )}
                  >
                    {isBuy ? "BUY" : isSell ? "SELL" : tx.transactionType.slice(0, 12)}
                  </span>
                </td>
                <td className="px-3 py-2 text-slate-300 font-mono tabular-nums text-xs">
                  {tx.shares.toLocaleString()}
                </td>
                <td className="px-3 py-2 text-slate-300 font-mono tabular-nums text-xs">
                  {tx.value !== null ? `$${(tx.value / 1000).toFixed(0)}K` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function FundamentalsTab({ symbol }: { symbol: string }) {
  const { data, isLoading } = useQuery<FundamentalsData>({
    queryKey: ["fundamentals", symbol],
    queryFn: async () => {
      const res = await fetch(`/api/fundamentals?symbol=${encodeURIComponent(symbol)}`);
      return res.json() as Promise<FundamentalsData>;
    },
    staleTime: 60 * 60 * 1000,
  });

  if (isLoading) {
    return (
      <div className="p-4 animate-pulse grid grid-cols-2 gap-3">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className="h-12 bg-slate-800 rounded" />
        ))}
      </div>
    );
  }

  if (!data) return <div className="p-8 text-center text-slate-500 text-sm">No data</div>;

  const shortPct = data.shortPercentOfFloat !== null ? data.shortPercentOfFloat * 100 : null;
  const roePct = data.returnOnEquity !== null ? data.returnOnEquity * 100 : null;

  const stats: { label: string; value: string; highlight?: string }[] = [
    {
      label: "Short Float %",
      value: shortPct !== null ? shortPct.toFixed(1) + "%" : "—",
      highlight: shortPct !== null
        ? shortPct > 40 ? "text-red-400"
        : shortPct > 20 ? "text-amber-400"
        : "text-slate-200"
        : "text-slate-500",
    },
    { label: "Days to Cover", value: fmtNum(data.shortRatio, 1) },
    { label: "Forward P/E", value: fmtNum(data.forwardPE, 1) },
    { label: "PEG Ratio", value: fmtNum(data.pegRatio, 2) },
    { label: "P/B", value: fmtNum(data.priceToBook, 2) },
    { label: "ROE %", value: roePct !== null ? roePct.toFixed(1) + "%" : "—" },
    { label: "Debt/Equity", value: fmtNum(data.debtToEquity, 2) },
  ];

  return (
    <div className="p-4 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
      {stats.map((s) => (
        <div key={s.label} className="bg-slate-800/60 rounded-lg px-3 py-2">
          <div className="text-xs text-slate-500 mb-1">{s.label}</div>
          <div className={cn("text-sm font-semibold font-mono", s.highlight ?? "text-slate-200")}>
            {s.value}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Full-width TradingView advanced chart panel, updated when a row is selected. */
export default function ChartPanel({ stock }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scriptRef = useRef<HTMLScriptElement | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("chart");

  useEffect(() => {
    if (!stock || activeTab !== "chart") return;

    const initWidget = () => {
      if (!containerRef.current) return;
      containerRef.current.innerHTML = "";
      const w = new window.TradingView.widget({
        symbol: stock.tvSymbol,
        interval: "D",
        container: containerRef.current,
        width: "100%",
        height: 520,
        theme: "dark",
        style: "1",
        locale: "en",
        toolbar_bg: "#0f172a",
        enable_publishing: false,
        withdateranges: true,
        allow_symbol_change: true,
        save_image: true,
        timezone: "exchange",
        hide_top_toolbar: false,
        studies: ["RSI@tv-basicstudies", "MACD@tv-basicstudies"],
      });
      window.tvWidget = w;
    };

    if (window.TradingView?.widget) {
      initWidget();
    } else if (!scriptRef.current) {
      const s = document.createElement("script");
      s.src = "https://s3.tradingview.com/tv.js";
      s.async = true;
      s.onload = initWidget;
      document.head.appendChild(s);
      scriptRef.current = s;
    } else {
      scriptRef.current.addEventListener("load", initWidget);
    }
  }, [stock?.tvSymbol, activeTab]);

  // Reset tab when stock changes
  useEffect(() => {
    setActiveTab("chart");
  }, [stock?.symbol]);

  if (!stock) {
    return (
      <div className="flex items-center justify-center h-[560px] rounded-xl border border-slate-700 bg-slate-900 text-slate-500 text-sm">
        Click any row to open the full chart
      </div>
    );
  }

  const tabs: { id: Tab; label: string }[] = [
    { id: "chart", label: "Chart" },
    { id: "insider", label: "Insider Transactions" },
    { id: "fundamentals", label: "Fundamentals" },
  ];

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 overflow-hidden">
      {/* Stock header */}
      <div className="flex flex-wrap items-center gap-4 px-4 py-3 border-b border-slate-700">
        <div>
          <span className="text-lg font-bold text-white mr-2">{stock.symbol}</span>
          <span className="text-slate-400 text-sm">{stock.name}</span>
        </div>
        <span className="text-slate-500 text-sm">
          {MARKET_FLAG[stock.market]} {stock.market}
        </span>
        <span className="text-slate-500 text-sm">{stock.sector}</span>
        {stock.price !== null && (
          <span className="text-white font-mono text-lg">
            {stock.price.toLocaleString(undefined, { maximumFractionDigits: 4 })}{" "}
            <span className="text-slate-400 text-sm">{stock.currency}</span>
          </span>
        )}
        <span className={`font-mono font-semibold ${pctColor(stock.performance.daily)}`}>
          {fmtPct(stock.performance.daily)} 1D
        </span>
        <span className={`font-mono text-sm ${pctColor(stock.performance.ytd)}`}>
          {fmtPct(stock.performance.ytd)} YTD
        </span>
        <EarningsBadge symbol={stock.symbol} />
        <span className="text-slate-500 text-sm ml-auto">
          MCap: {fmtMarketCap(stock.marketCap)}
        </span>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-slate-700 bg-slate-900/60">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px",
              activeTab === tab.id
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-slate-500 hover:text-slate-300"
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "chart" && (
        <>
          <div ref={containerRef} id="tv-chart-container" className="w-full h-[520px]" />
          <p className="text-xs text-slate-600 px-4 py-2">
            Prices delayed 15 minutes · Powered by TradingView
          </p>
        </>
      )}
      {activeTab === "insider" && (
        <div className="min-h-[200px]">
          <InsiderTab symbol={stock.symbol} />
        </div>
      )}
      {activeTab === "fundamentals" && (
        <div className="min-h-[200px]">
          <FundamentalsTab symbol={stock.symbol} />
        </div>
      )}
    </div>
  );
}
