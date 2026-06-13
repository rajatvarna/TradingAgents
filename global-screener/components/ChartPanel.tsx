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

type Tab = "chart" | "insider" | "fundamentals" | "news";

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
  strongBuy: number | null;
  buy: number | null;
  hold: number | null;
  sell: number | null;
  strongSell: number | null;
}

interface NewsArticle {
  title: string;
  link: string;
  publisher: string;
  publishedAt: string | null;
}

function fmtNum(v: number | null, decimals = 2): string {
  if (v === null) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
}

function fmtPct2(v: number | null): string {
  if (v === null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function EarningsBadge({ yahooSuffix }: { yahooSuffix: string }) {
  const { data } = useQuery<EarningsData>({
    queryKey: ["earnings", yahooSuffix],
    queryFn: async () => {
      const res = await fetch(`/api/earnings?symbol=${encodeURIComponent(yahooSuffix)}`);
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

function InsiderTab({ yahooSuffix }: { yahooSuffix: string }) {
  const { data, isLoading } = useQuery<InsiderTransaction[]>({
    queryKey: ["insider", yahooSuffix],
    queryFn: async () => {
      const res = await fetch(`/api/insider?symbol=${encodeURIComponent(yahooSuffix)}`);
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

function FundamentalsTab({ yahooSuffix }: { yahooSuffix: string }) {
  const { data, isLoading } = useQuery<FundamentalsData>({
    queryKey: ["fundamentals", yahooSuffix],
    queryFn: async () => {
      const res = await fetch(`/api/fundamentals?symbol=${encodeURIComponent(yahooSuffix)}`);
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

  const sb = data.strongBuy ?? 0;
  const b  = data.buy ?? 0;
  const h  = data.hold ?? 0;
  const s  = data.sell ?? 0;
  const ss = data.strongSell ?? 0;
  const total = sb + b + h + s + ss;
  const bullish = sb + b;
  const bearish = s + ss;
  const consensus = total === 0 ? null : bullish > bearish * 2 ? "Bullish" : bearish > bullish * 2 ? "Bearish" : "Neutral";

  return (
    <div className="p-4 space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {stats.map((stat) => (
          <div key={stat.label} className="bg-slate-800/60 rounded-lg px-3 py-2">
            <div className="text-xs text-slate-500 mb-1">{stat.label}</div>
            <div className={cn("text-sm font-semibold font-mono", stat.highlight ?? "text-slate-200")}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>
      {total > 0 && (
        <div className="bg-slate-800/40 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-slate-400 font-semibold">Analyst Consensus</span>
            {consensus && (
              <span className={cn("text-xs font-bold px-2 py-0.5 rounded-full",
                consensus === "Bullish" ? "bg-emerald-900/60 text-emerald-300" :
                consensus === "Bearish" ? "bg-red-900/60 text-red-300" :
                "bg-slate-700 text-slate-300"
              )}>{consensus}</span>
            )}
          </div>
          <div className="flex rounded-full overflow-hidden h-4 text-xs">
            {sb > 0 && <div style={{ width: `${(sb/total)*100}%` }} className="bg-emerald-600 flex items-center justify-center text-white font-bold" title={`Strong Buy: ${sb}`}>{sb}</div>}
            {b  > 0 && <div style={{ width: `${(b/total)*100}%`  }} className="bg-emerald-400 flex items-center justify-center text-white" title={`Buy: ${b}`}>{b}</div>}
            {h  > 0 && <div style={{ width: `${(h/total)*100}%`  }} className="bg-yellow-400 flex items-center justify-center text-slate-900" title={`Hold: ${h}`}>{h}</div>}
            {s  > 0 && <div style={{ width: `${(s/total)*100}%`  }} className="bg-red-400 flex items-center justify-center text-white" title={`Sell: ${s}`}>{s}</div>}
            {ss > 0 && <div style={{ width: `${(ss/total)*100}%` }} className="bg-red-600 flex items-center justify-center text-white font-bold" title={`Strong Sell: ${ss}`}>{ss}</div>}
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>Strong Buy</span><span>{total} analysts</span><span>Strong Sell</span>
          </div>
        </div>
      )}
    </div>
  );
}

function NewsTab({ yahooSuffix }: { yahooSuffix: string }) {
  const { data, isLoading } = useQuery<NewsArticle[]>({
    queryKey: ["news", yahooSuffix],
    queryFn: async () => {
      const res = await fetch(`/api/news?symbol=${encodeURIComponent(yahooSuffix)}`);
      return res.json() as Promise<NewsArticle[]>;
    },
    staleTime: 15 * 60 * 1000,
  });

  if (isLoading) {
    return (
      <div className="p-4 animate-pulse space-y-3">
        {[1,2,3,4,5].map((i) => <div key={i} className="h-10 bg-slate-800 rounded" />)}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return <div className="p-8 text-center text-slate-500 text-sm">No news found</div>;
  }

  function relativeTime(ts: string | null): string {
    if (!ts) return "";
    const date = new Date(ts);
    const diff = Math.floor((Date.now() - date.getTime()) / 1000);
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    if (diff < 172800) return "Yesterday";
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  }

  return (
    <div className="divide-y divide-slate-800">
      {data.map((article, i) => (
        <a
          key={i}
          href={article.link}
          target="_blank"
          rel="noopener noreferrer"
          className="flex flex-col gap-1 px-4 py-3 hover:bg-slate-800/50 transition-colors"
        >
          <span className="text-sm text-white leading-snug hover:text-blue-300 transition-colors">
            {article.title}
          </span>
          <span className="text-xs text-slate-500">
            {article.publisher}{article.publishedAt ? ` · ${relativeTime(article.publishedAt)}` : ""}
          </span>
        </a>
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
    { id: "news", label: "News" },
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
        <EarningsBadge yahooSuffix={stock.yahooSuffix} />
        {stock.fiftyTwoWeekLow !== null && stock.fiftyTwoWeekHigh !== null && stock.price !== null && (() => {
          const range = stock.fiftyTwoWeekHigh - stock.fiftyTwoWeekLow;
          const pos = range > 0 ? Math.min(100, Math.max(0, ((stock.price - stock.fiftyTwoWeekLow) / range) * 100)) : 50;
          const pctFromHigh = ((stock.price - stock.fiftyTwoWeekHigh) / stock.fiftyTwoWeekHigh) * 100;
          const barColor = pos >= 80 ? "bg-emerald-500" : pos >= 40 ? "bg-amber-500" : "bg-red-500";
          return (
            <div className="flex items-center gap-1.5 text-xs text-slate-500" title={`${pctFromHigh.toFixed(1)}% from 52W High`}>
              <span className="hidden sm:inline">{stock.fiftyTwoWeekLow.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
              <div className="w-20 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div className={cn("h-full rounded-full", barColor)} style={{ width: `${pos}%` }} />
              </div>
              <span className="hidden sm:inline">{stock.fiftyTwoWeekHigh.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
              <span className="text-slate-600">52W</span>
            </div>
          );
        })()}
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
      {activeTab === "news" && (
        <div className="min-h-[200px] max-h-[520px] overflow-y-auto">
          <NewsTab yahooSuffix={stock.yahooSuffix} />
        </div>
      )}
      {activeTab === "insider" && (
        <div className="min-h-[200px]">
          <InsiderTab yahooSuffix={stock.yahooSuffix} />
        </div>
      )}
      {activeTab === "fundamentals" && (
        <div className="min-h-[200px]">
          <FundamentalsTab yahooSuffix={stock.yahooSuffix} />
        </div>
      )}
    </div>
  );
}
