"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { StockData, Fundamentals, SentimentData } from "@/types";
import { fmtPct, pctColor, cn } from "@/lib/utils";

interface Props {
  stock: StockData;
}

async function loadFundamentals(yahooSuffix: string): Promise<Fundamentals> {
  const res = await fetch(`/api/fundamentals?symbol=${encodeURIComponent(yahooSuffix)}`);
  if (!res.ok) throw new Error("Failed");
  return res.json();
}

async function loadSentiment(symbol: string, name: string): Promise<SentimentData> {
  const res = await fetch(`/api/sentiment?symbol=${encodeURIComponent(symbol)}&name=${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error("Failed");
  return res.json();
}

function MetricRow({ label, value, format = "text", suffix = "" }: {
  label: string;
  value: number | string | null;
  format?: "pct" | "text" | "x" | "currency";
  suffix?: string;
}) {
  if (value === null || value === undefined) return null;
  let display: string;
  if (format === "pct" && typeof value === "number") {
    display = fmtPct(value * 100);
  } else if (format === "x" && typeof value === "number") {
    display = `${value.toFixed(2)}×`;
  } else if (format === "currency" && typeof value === "number") {
    display = `$${value.toFixed(2)}`;
  } else {
    display = String(value) + suffix;
  }
  const colorClass = format === "pct" && typeof value === "number" ? pctColor(value * 100) : "text-slate-200";
  return (
    <div className="flex justify-between items-center py-1 border-b border-slate-800 last:border-0">
      <span className="text-slate-500 text-xs">{label}</span>
      <span className={cn("text-xs font-mono tabular-nums font-medium", colorClass)}>{display}</span>
    </div>
  );
}

function SentimentBadge({ label }: { label: SentimentData["label"] }) {
  const config = {
    Bullish:  { bg: "bg-emerald-900/60 border-emerald-700", text: "text-emerald-300", icon: "📈" },
    Bearish:  { bg: "bg-red-900/60 border-red-700",         text: "text-red-300",     icon: "📉" },
    Neutral:  { bg: "bg-slate-800 border-slate-600",        text: "text-slate-300",   icon: "➖" },
  };
  if (!label) return null;
  const c = config[label];
  return (
    <span className={cn("inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-semibold", c.bg, c.text)}>
      {c.icon} {label}
    </span>
  );
}

function ScoreBar({ score }: { score: number | null }) {
  if (score === null) return null;
  const pct = Math.round((score + 1) * 50); // -1→0%, 0→50%, +1→100%
  return (
    <div className="flex items-center gap-2 mt-1">
      <span className="text-xs text-red-400">Bear</span>
      <div className="flex-1 h-2 rounded-full bg-slate-700 overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${pct}%`,
            background: pct > 55 ? "#22c55e" : pct < 45 ? "#ef4444" : "#94a3b8",
          }}
        />
      </div>
      <span className="text-xs text-emerald-400">Bull</span>
    </div>
  );
}

/** Fundamentals + Social Sentiment detail panel for a selected stock. */
export default function StockDetail({ stock }: Props) {
  const [tab, setTab] = useState<"fundamentals" | "sentiment">("fundamentals");

  // Reset tab when stock changes
  useEffect(() => { setTab("fundamentals"); }, [stock.symbol]);

  const { data: fundamentals, isLoading: fundLoading } = useQuery({
    queryKey: ["fundamentals", stock.symbol],
    queryFn: () => loadFundamentals(stock.symbol),
    staleTime: 30 * 60 * 1000,
    retry: 1,
  });

  const { data: sentiment, isLoading: sentLoading } = useQuery({
    queryKey: ["sentiment", stock.symbol],
    queryFn: () => loadSentiment(stock.symbol, stock.name),
    staleTime: 15 * 60 * 1000,
    retry: 1,
  });

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-slate-700">
        {(["fundamentals", "sentiment"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn(
              "px-4 py-2.5 text-xs font-semibold uppercase tracking-wide transition-colors",
              tab === t
                ? "border-b-2 border-blue-500 text-blue-400 bg-slate-800/50"
                : "text-slate-500 hover:text-slate-300"
            )}
          >
            {t === "fundamentals" ? "📊 Fundamentals" : "💬 Sentiment"}
          </button>
        ))}
      </div>

      <div className="p-4">
        {/* Fundamentals tab */}
        {tab === "fundamentals" && (
          fundLoading ? (
            <div className="space-y-2 animate-pulse">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="h-6 bg-slate-800 rounded" />
              ))}
            </div>
          ) : !fundamentals ? (
            <p className="text-slate-500 text-sm">Fundamentals unavailable for {stock.symbol}</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8">
              {/* Valuation */}
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Valuation</p>
                <MetricRow label="P/E (TTM)"          value={fundamentals.pe}       format="x" />
                <MetricRow label="Forward P/E"        value={fundamentals.forwardPe} format="x" />
                <MetricRow label="P/B Ratio"          value={fundamentals.pbRatio}  format="x" />
                <MetricRow label="P/S Ratio"          value={fundamentals.psRatio}  format="x" />
                <MetricRow label="EV/EBITDA"          value={fundamentals.evEbitda} format="x" />
                <MetricRow label="EPS (TTM)"          value={fundamentals.eps}      format="currency" />
                <MetricRow label="EPS (Forward)"      value={fundamentals.epsForward} format="currency" />
                <MetricRow label="Dividend Yield"     value={fundamentals.dividendYield} format="pct" />
                <MetricRow label="Beta"               value={fundamentals.beta}     suffix="×" />
                {fundamentals.earningsDate && (
                  <div className="mt-3 p-2 rounded-lg bg-slate-800 border border-slate-700">
                    <p className="text-xs text-slate-400">Next Earnings</p>
                    <p className="text-sm font-semibold text-white mt-0.5">
                      {new Date(fundamentals.earningsDate).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                    </p>
                  </div>
                )}
              </div>
              {/* Profitability & Growth */}
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Profitability & Growth</p>
                <MetricRow label="Profit Margin"      value={fundamentals.profitMargin}      format="pct" />
                <MetricRow label="Return on Equity"   value={fundamentals.returnOnEquity}    format="pct" />
                <MetricRow label="Return on Assets"   value={fundamentals.returnOnAssets}    format="pct" />
                <MetricRow label="Revenue Growth YoY" value={fundamentals.revenueGrowthYoy}  format="pct" />
                <MetricRow label="Earnings Growth YoY" value={fundamentals.earningsGrowthYoy} format="pct" />
                <MetricRow label="Debt / Equity"      value={fundamentals.debtToEquity}      format="x" />
                {/* Analyst consensus */}
                {fundamentals.analystRating && (
                  <div className="mt-3 p-3 rounded-lg bg-slate-800 border border-slate-700">
                    <p className="text-xs text-slate-400 mb-1">
                      Analyst Consensus
                      {fundamentals.analystCount && (
                        <span className="text-slate-600 ml-1">({fundamentals.analystCount} analysts)</span>
                      )}
                    </p>
                    <p className={cn(
                      "text-sm font-bold",
                      fundamentals.analystRating.includes("Buy") ? "text-emerald-400" :
                      fundamentals.analystRating.includes("Sell") ? "text-red-400" : "text-slate-300"
                    )}>
                      {fundamentals.analystRating}
                    </p>
                    {fundamentals.targetPrice && (
                      <p className="text-xs text-slate-400 mt-1">
                        Price Target: <span className="text-white font-mono">${fundamentals.targetPrice.toFixed(2)}</span>
                        {stock.price && (
                          <span className={cn("ml-2 font-mono", pctColor(((fundamentals.targetPrice - stock.price) / stock.price) * 100))}>
                            {fmtPct(((fundamentals.targetPrice - stock.price) / stock.price) * 100)} upside
                          </span>
                        )}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
          )
        )}

        {/* Sentiment tab */}
        {tab === "sentiment" && (
          sentLoading ? (
            <div className="space-y-3 animate-pulse">
              <div className="h-8 bg-slate-800 rounded" />
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="h-12 bg-slate-800 rounded" />
              ))}
            </div>
          ) : !sentiment ? (
            <p className="text-slate-500 text-sm">Sentiment data unavailable</p>
          ) : (
            <div className="space-y-4">
              {/* Score summary */}
              <div className="flex items-center gap-4 p-3 rounded-lg bg-slate-800 border border-slate-700">
                <div>
                  <div className="flex items-center gap-2">
                    <SentimentBadge label={sentiment.label} />
                    <span className="text-xs text-slate-500">
                      {sentiment.mentionCount} posts this week
                    </span>
                  </div>
                  <ScoreBar score={sentiment.score} />
                </div>
                {sentiment.score !== null && (
                  <div className="ml-auto text-right">
                    <span className={cn("text-2xl font-bold font-mono tabular-nums", pctColor(sentiment.score * 100))}>
                      {sentiment.score >= 0 ? "+" : ""}{(sentiment.score * 100).toFixed(0)}
                    </span>
                    <p className="text-xs text-slate-500">sentiment score</p>
                  </div>
                )}
              </div>

              {/* Posts */}
              {sentiment.posts.length === 0 ? (
                <p className="text-slate-500 text-sm text-center py-4">
                  No recent social mentions found for {stock.symbol}
                </p>
              ) : (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                    Recent Mentions
                  </p>
                  {sentiment.posts.map((post, i) => (
                    <a
                      key={i}
                      href={post.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block p-3 rounded-lg bg-slate-800/50 border border-slate-700 hover:border-slate-500 transition-colors"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-xs text-slate-200 leading-snug flex-1">{post.title}</p>
                        {post.source === "reddit" && post.score > 0 && (
                          <span className="text-xs text-orange-400 font-mono whitespace-nowrap ml-2">
                            ↑{post.score.toLocaleString()}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        {post.source === "reddit" ? (
                          <span className="text-xs text-orange-500/80">
                            🟠 {post.subreddit}
                          </span>
                        ) : (
                          <span className="text-xs text-blue-500/80">📰 News</span>
                        )}
                        <span className="text-xs text-slate-600">
                          {new Date(post.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                    </a>
                  ))}
                </div>
              )}
              <p className="text-xs text-slate-600 text-center">
                Sources: Reddit (r/wallstreetbets, r/stocks, r/investing) · Yahoo Finance News
              </p>
            </div>
          )
        )}
      </div>
    </div>
  );
}
