"use client";

import { useEffect, useRef, useState } from "react";

interface Alert {
  id: string;
  symbol: string;
  price: number;
  direction: "above" | "below";
}

const STORAGE_KEY = "price-alerts";

function loadAlerts(): Alert[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Alert[]) : [];
  } catch {
    return [];
  }
}

function saveAlerts(alerts: Alert[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(alerts));
  } catch { /* ignore */ }
}

function uid(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function PriceAlerts() {
  const [open, setOpen] = useState(false);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [symInput, setSymInput] = useState("");
  const [priceInput, setPriceInput] = useState("");
  const [dirInput, setDirInput] = useState<"above" | "below">("above");
  const modalRef = useRef<HTMLDivElement>(null);
  const checkedRef = useRef<Set<string>>(new Set());

  // Load on mount + request notification permission
  useEffect(() => {
    setAlerts(loadAlerts());
    if (typeof Notification !== "undefined" && Notification.permission === "default") {
      Notification.requestPermission().catch(() => {/* ignore */});
    }
  }, []);

  // Close modal on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Poll /api/batch every 60s and check alerts
  useEffect(() => {
    if (!alerts.length) return;

    let cancelled = false;

    const check = async () => {
      try {
        const res = await fetch("/api/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ market: "All" }),
        });
        if (!res.ok || cancelled) return;
        const json = await res.json();
        const priceMap = new Map<string, number>();
        for (const stock of json.data ?? []) {
          if (stock.price !== null) priceMap.set(stock.symbol, stock.price);
        }

        const triggered: string[] = [];
        for (const alert of alerts) {
          const key = `${alert.id}`;
          if (checkedRef.current.has(key)) continue;
          const current = priceMap.get(alert.symbol.toUpperCase());
          if (current === undefined) continue;
          const hit =
            (alert.direction === "above" && current >= alert.price) ||
            (alert.direction === "below" && current <= alert.price);
          if (hit) {
            triggered.push(key);
            if (
              typeof Notification !== "undefined" &&
              Notification.permission === "granted"
            ) {
              new Notification(`🔔 ${alert.symbol} price alert`, {
                body: `${alert.symbol} is now ${alert.direction} ${alert.price} (current: ${current.toFixed(2)})`,
              });
            }
          }
        }

        if (triggered.length && !cancelled) {
          triggered.forEach((k) => checkedRef.current.add(k));
          setAlerts((prev) => {
            const next = prev.filter((a) => !triggered.includes(a.id));
            saveAlerts(next);
            return next;
          });
        }
      } catch { /* ignore */ }
    };

    check();
    const id = setInterval(check, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alerts.length]);

  const addAlert = () => {
    const sym = symInput.trim().toUpperCase();
    const price = parseFloat(priceInput);
    if (!sym || isNaN(price) || price <= 0) return;
    const alert: Alert = { id: uid(), symbol: sym, price, direction: dirInput };
    const next = [...alerts, alert];
    setAlerts(next);
    saveAlerts(next);
    setSymInput("");
    setPriceInput("");
  };

  const removeAlert = (id: string) => {
    const next = alerts.filter((a) => a.id !== id);
    setAlerts(next);
    saveAlerts(next);
  };

  return (
    <>
      {/* Bell button */}
      <button
        onClick={() => setOpen(true)}
        className="relative px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800 text-slate-300 hover:border-slate-500 hover:text-white transition-colors text-sm"
        title="Price Alerts"
      >
        🔔
        {alerts.length > 0 && (
          <span className="absolute -top-1 -right-1 bg-blue-500 text-white text-[10px] rounded-full w-4 h-4 flex items-center justify-center font-bold">
            {alerts.length}
          </span>
        )}
      </button>

      {/* Modal */}
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div
            ref={modalRef}
            className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-md mx-4 p-5"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-white font-semibold text-lg">🔔 Price Alerts</h2>
              <button
                onClick={() => setOpen(false)}
                className="text-slate-500 hover:text-white text-xl leading-none"
              >
                ×
              </button>
            </div>

            {/* Add form */}
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                placeholder="Symbol (e.g. AAPL)"
                value={symInput}
                onChange={(e) => setSymInput(e.target.value)}
                className="flex-1 bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-1.5 placeholder-slate-600"
              />
              <input
                type="number"
                placeholder="Price"
                value={priceInput}
                onChange={(e) => setPriceInput(e.target.value)}
                className="w-24 bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-1.5 placeholder-slate-600"
              />
              <select
                value={dirInput}
                onChange={(e) => setDirInput(e.target.value as "above" | "below")}
                className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-2 py-1.5"
              >
                <option value="above">Above</option>
                <option value="below">Below</option>
              </select>
              <button
                onClick={addAlert}
                className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs rounded-lg font-medium transition-colors"
              >
                Add
              </button>
            </div>

            {/* Alert list */}
            {alerts.length === 0 ? (
              <p className="text-slate-500 text-sm text-center py-4">No active alerts</p>
            ) : (
              <ul className="space-y-2 max-h-64 overflow-y-auto">
                {alerts.map((a) => (
                  <li
                    key={a.id}
                    className="flex items-center justify-between bg-slate-800 rounded-lg px-3 py-2 text-sm"
                  >
                    <span className="text-white font-semibold">{a.symbol}</span>
                    <span className="text-slate-400 mx-2">
                      {a.direction === "above" ? "↑" : "↓"} {a.price.toLocaleString()}
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${a.direction === "above" ? "bg-emerald-900 text-emerald-300" : "bg-red-900 text-red-300"}`}>
                      {a.direction}
                    </span>
                    <button
                      onClick={() => removeAlert(a.id)}
                      className="ml-3 text-slate-600 hover:text-red-400 transition-colors"
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <p className="text-[10px] text-slate-600 mt-3">
              Alerts check every 60 s. Browser notifications must be allowed.
            </p>
          </div>
        </div>
      )}
    </>
  );
}
