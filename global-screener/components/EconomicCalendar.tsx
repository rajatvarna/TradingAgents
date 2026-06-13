"use client";

import { useEffect, useRef, useState } from "react";

export default function EconomicCalendar() {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const injectedRef = useRef(false);

  useEffect(() => {
    if (!open || injectedRef.current || !containerRef.current) return;
    injectedRef.current = true;

    const container = containerRef.current;
    container.innerHTML = "";

    // Outer wrapper required by TradingView embed
    const wrapper = document.createElement("div");
    wrapper.className = "tradingview-widget-container";
    wrapper.style.height = "400px";
    wrapper.style.width = "100%";

    const inner = document.createElement("div");
    inner.className = "tradingview-widget-container__widget";
    wrapper.appendChild(inner);

    const script = document.createElement("script");
    script.type = "text/javascript";
    script.src =
      "https://s3.tradingview.com/external-embedding/embed-widget-events.js";
    script.async = true;
    script.innerHTML = JSON.stringify({
      colorTheme: "dark",
      isTransparent: true,
      width: "100%",
      height: "400",
      locale: "en",
      importanceFilter: "-1,0,1",
      countryFilter: "us,in,ae,sa",
    });

    wrapper.appendChild(script);
    container.appendChild(wrapper);
  }, [open]);

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-4 py-3 border-b border-slate-700 hover:bg-slate-800/40 transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="font-semibold text-white text-sm">
          Economic Calendar
        </span>
        <span className="text-slate-400 text-xs">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div ref={containerRef} className="w-full" style={{ minHeight: "400px" }} />
      )}
    </div>
  );
}
