"use client";

import { useEffect, useRef } from "react";
import { StockData } from "@/types";
import { fmtPct, fmtMarketCap, pctColor, MARKET_FLAG } from "@/lib/utils";

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

/** Full-width TradingView advanced chart panel, updated when a row is selected. */
export default function ChartPanel({ stock }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scriptRef = useRef<HTMLScriptElement | null>(null);

  useEffect(() => {
    if (!stock) return;

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
  }, [stock?.tvSymbol]);

  if (!stock) {
    return (
      <div className="flex items-center justify-center h-[560px] rounded-xl border border-slate-700 bg-slate-900 text-slate-500 text-sm">
        Click any row to open the full chart
      </div>
    );
  }

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
        <span className="text-slate-500 text-sm ml-auto">
          MCap: {fmtMarketCap(stock.marketCap)}
        </span>
      </div>

      <div ref={containerRef} id="tv-chart-container" className="w-full h-[520px]" />

      <p className="text-xs text-slate-600 px-4 py-2">
        Prices delayed 15 minutes · Powered by TradingView
      </p>
    </div>
  );
}
