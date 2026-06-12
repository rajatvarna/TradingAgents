"use client";

import { useEffect, useRef } from "react";
import { isMarketOpen } from "@/lib/marketHours";

interface MarketCard {
  market: string;
  flag: string;
  symbols: Array<{ s: string; d: string }>;
}

const CARDS: MarketCard[] = [
  { market: "US", flag: "🇺🇸", symbols: [
    { s: "SP:SPX", d: "S&P 500" }, { s: "DJ:DJI", d: "Dow Jones" }, { s: "NASDAQ:IXIC", d: "NASDAQ" },
  ]},
  { market: "India", flag: "🇮🇳", symbols: [
    { s: "BSE:SENSEX", d: "SENSEX" }, { s: "NSE:NIFTY", d: "NIFTY 50" }, { s: "NSE:NIFTY_MIDCAP_100", d: "Midcap 100" },
  ]},
  { market: "UAE", flag: "🇦🇪", symbols: [
    { s: "DFM:DFMGI", d: "DFM Index" }, { s: "ADX:ADXGI", d: "ADX Index" },
  ]},
  { market: "Saudi", flag: "🇸🇦", symbols: [
    { s: "TADAWUL:TASI", d: "TASI" },
  ]},
];

function MarketCard({ card }: { card: MarketCard }) {
  const ref = useRef<HTMLDivElement>(null);
  const open = isMarketOpen(card.market);

  useEffect(() => {
    if (!ref.current || ref.current.childElementCount > 0) return;
    const widget = document.createElement("div");
    widget.className = "tradingview-widget-container__widget";
    ref.current.appendChild(widget);

    const script = document.createElement("script");
    script.src =
      "https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js";
    script.async = true;
    script.innerHTML = JSON.stringify({
      colorTheme: "dark",
      dateRange: "1D",
      showChart: true,
      locale: "en",
      width: "100%",
      height: 180,
      isTransparent: true,
      showSymbolLogo: false,
      showFloatingTooltip: false,
      tabs: [
        {
          title: `${card.flag} ${card.market}`,
          symbols: card.symbols.map((s) => ({ s: s.s, d: s.d })),
          originalTitle: card.market,
        },
      ],
    });
    ref.current.appendChild(script);
  }, [card]);

  return (
    <div className="relative rounded-xl border border-slate-700 bg-slate-900 p-3 overflow-hidden min-h-[200px]">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-semibold text-slate-200">
          {card.flag} {card.market}
        </span>
        <span
          className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            open ? "bg-emerald-900 text-emerald-300" : "bg-slate-700 text-slate-400"
          }`}
        >
          {open ? "LIVE" : "CLOSED"}
        </span>
      </div>
      <div ref={ref} className="tradingview-widget-container w-full" />
    </div>
  );
}

/** Four market overview cards with embedded TradingView widgets. */
export default function MarketIndexCards() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {CARDS.map((c) => (
        <MarketCard key={c.market} card={c} />
      ))}
    </div>
  );
}
