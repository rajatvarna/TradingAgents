"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { StockData } from "@/types";

const TickerTape = dynamic(() => import("@/components/TickerTape"), { ssr: false });
const MarketIndexCards = dynamic(() => import("@/components/MarketIndexCards"), { ssr: false });
const ScreenerTable = dynamic(() => import("@/components/ScreenerTable"), { ssr: false });
const ChartPanel = dynamic(() => import("@/components/ChartPanel"), { ssr: false });
const MarketStatusBar = dynamic(() => import("@/components/MarketStatusBar"), { ssr: false });
const SectorHeatmap = dynamic(() => import("@/components/SectorHeatmap"), { ssr: false });
const ThemeToggle = dynamic(() => import("@/components/ThemeToggle"), { ssr: false });

export default function Home() {
  const [selectedStock, setSelectedStock] = useState<StockData | null>(null);
  const [stockData, setStockData] = useState<StockData[]>([]);

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
          <a
            href="https://www.tradingview.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-500 hover:text-slate-300"
          >
            Powered by TradingView
          </a>
          <ThemeToggle />
        </div>
      </header>

      {/* Ticker tape */}
      <div className="border-b border-slate-800">
        <TickerTape />
      </div>

      {/* Market status bar */}
      <div className="border-b border-slate-800 px-4 md:px-6">
        <MarketStatusBar />
      </div>

      {/* Main content */}
      <main className="flex-1 px-4 md:px-6 py-4 space-y-4 max-w-[1800px] mx-auto w-full">
        {/* Market index overview cards */}
        <MarketIndexCards />

        {/* Screener table */}
        <ScreenerTable onSelectStock={setSelectedStock} onDataLoaded={setStockData} />

        {/* Sector heatmap */}
        <SectorHeatmap data={stockData} />

        {/* Chart panel */}
        <ChartPanel stock={selectedStock} />
      </main>

      <footer className="px-6 py-3 border-t border-slate-800 text-xs text-slate-600 flex justify-between">
        <span>GlobalScreener — built with Next.js, TanStack Query &amp; TradingView</span>
        <span>Data: Yahoo Finance (free, delayed)</span>
      </footer>
    </div>
  );
}
