"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  tvSymbol: string;
}

/** Lazy-loaded TradingView mini symbol overview. Only mounts when visible in viewport. */
export default function MiniSparkline({ tvSymbol }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); observer.disconnect(); } },
      { rootMargin: "200px" }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!visible || mounted || !ref.current) return;
    setMounted(true);

    const container = document.createElement("div");
    container.className = "tradingview-widget-container__widget";
    ref.current.appendChild(container);

    const script = document.createElement("script");
    script.src =
      "https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js";
    script.async = true;
    script.innerHTML = JSON.stringify({
      symbol: tvSymbol,
      width: 180,
      height: 70,
      locale: "en",
      dateRange: "1M",
      colorTheme: "dark",
      isTransparent: true,
      autosize: false,
      largeChartUrl: "",
      noTimeScale: true,
    });
    ref.current.appendChild(script);
  }, [visible, tvSymbol, mounted]);

  return (
    <div
      ref={ref}
      className="tradingview-widget-container w-[180px] h-[70px]"
    />
  );
}
