"use client";

import { useEffect, useRef } from "react";

/** TradingView Ticker Tape widget showing all 4 market indices. */
export default function TickerTape() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || ref.current.childElementCount > 0) return;

    const container = document.createElement("div");
    container.className = "tradingview-widget-container__widget";
    ref.current.appendChild(container);

    const script = document.createElement("script");
    script.src =
      "https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js";
    script.async = true;
    script.innerHTML = JSON.stringify({
      symbols: [
        { proName: "SP:SPX",       title: "S&P 500" },
        { proName: "NASDAQ:IXIC",  title: "NASDAQ" },
        { proName: "BSE:SENSEX",   title: "SENSEX" },
        { proName: "NSE:NIFTY",    title: "NIFTY 50" },
        { proName: "DFM:DFMGI",    title: "DFM Index" },
        { proName: "ADX:ADXGI",    title: "ADX Index" },
        { proName: "TADAWUL:TASI", title: "TASI" },
      ],
      showSymbolLogo: true,
      colorTheme: "dark",
      isTransparent: false,
      displayMode: "adaptive",
      locale: "en",
    });
    ref.current.appendChild(script);
  }, []);

  return (
    <div className="tradingview-widget-container w-full overflow-hidden" ref={ref} />
  );
}
