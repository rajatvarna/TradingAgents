"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { StockData } from "@/types";

const TickerTape        = dynamic(() => import("@/components/TickerTape"),        { ssr: false });
const MarketIndexCards  = dynamic(() => import("@/components/MarketIndexCards"),   { ssr: false });
const ScreenerTable     = dynamic(() => import("@/components/ScreenerTable"),      { ssr: false });
const ChartPanel        = dynamic(() => import("@/components/ChartPanel"),         { ssr: false });
const FearGreed         = dynamic(() => import("@/components/FearGreed"),          { ssr: false });
const PriceAlerts       = dynamic(() => import("@/components/PriceAlerts"),        { ssr: false });
const PortfolioTracker  = dynamic(() => import("@/components/PortfolioTracker"),   { ssr: false });
const EconomicCalendar  = dynamic(() => import("@/components/EconomicCalendar"),   { ssr: false });
const SectorHeatmap     = dynamic(() => import("@/components/SectorHeatmap"),      { ssr: false });

type ViewMode = "table" | "heatmap";

export default function Home() {
  const [selectedStock, setSelectedStock] = useState<StockData | null>(null);
  const [screenerStocks, setScreenerStocks] = useState<StockData[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>("table");

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-950 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold text-white tracking-tight">
            🌐 GlobalScreener
          </span>
          <span className="text-xs text-slate-500 hidden sm:block">
            US · India · UAE · Saudi Arabia
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span className="hidden md:block">Prices delayed 15 min · Free data</span>
          <PriceAlerts />
          <a
            href="https://www.tradingview.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-500 hover:text-slate-300"
          >
            Powered by TradingView
          </a>
        </div>
      </header>

      {/* Ticker tape */}
      <div className="border-b border-slate-800">
        <TickerTape />
      </div>

      {/* Main content */}
      <main className="flex-1 px-4 md:px-6 py-4 space-y-4 max-w-[1800px] mx-auto w-full">
        {/* Market index overview cards */}
        <MarketIndexCards />

        {/* VIX Fear & Greed gauge */}
        <FearGreed />

        {/* View mode toggle */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode("table")}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              viewMode === "table"
                ? "bg-blue-600 border-blue-500 text-white"
                : "bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500 hover:text-white"
            }`}
          >
            📋 Table
          </button>
          <button
            onClick={() => setViewMode("heatmap")}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              viewMode === "heatmap"
                ? "bg-blue-600 border-blue-500 text-white"
                : "bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500 hover:text-white"
            }`}
          >
            🗺 Heatmap
          </button>
        </div>

        {/* Screener table (always mounted to keep data loaded) */}
        <div className={viewMode === "table" ? "" : "hidden"}>
          <ScreenerTable onSelectStock={setSelectedStock} onDataLoaded={setScreenerStocks} />
        </div>

        {/* Sector heatmap */}
        {viewMode === "heatmap" && (
          <SectorHeatmap stocks={screenerStocks} onSelectStock={setSelectedStock} />
        )}

        {/* Chart panel */}
        <ChartPanel stock={selectedStock} />

        {/* Portfolio tracker */}
        <PortfolioTracker stocks={screenerStocks} />

        {/* Economic calendar */}
        <EconomicCalendar />
      </main>

      <footer className="px-6 py-3 border-t border-slate-800 text-xs text-slate-600 flex justify-between">
        <span>GlobalScreener — built with Next.js, TanStack Query &amp; TradingView</span>
        <span>Data: Yahoo Finance (free, delayed)</span>
      </footer>
    </div>
  );
}
