"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface Alert { id: string; symbol: string; targetPrice: number; direction: "above" | "below"; triggered: boolean; }

interface Props { latestPrices: Map<string, number>; }

const STORAGE_KEY = "price_alerts";

function loadAlerts(): Alert[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]"); } catch { return []; }
}
function saveAlerts(alerts: Alert[]) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(alerts)); } catch {}
}

export default function PriceAlerts({ latestPrices }: Props) {
  const [open, setOpen] = useState(false);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [symbol, setSymbol] = useState("");
  const [price, setPrice] = useState("");
  const [direction, setDirection] = useState<"above" | "below">("above");

  useEffect(() => { setAlerts(loadAlerts()); }, []);

  // Check alerts on price updates
  useEffect(() => {
    if (!alerts.length || !latestPrices.size) return;
    let changed = false;
    const updated = alerts.map((a) => {
      if (a.triggered) return a;
      const current = latestPrices.get(a.symbol);
      if (current === undefined) return a;
      const hit = a.direction === "above" ? current >= a.targetPrice : current <= a.targetPrice;
      if (!hit) return a;
      changed = true;
      // Fire browser notification
      if ("Notification" in window && Notification.permission === "granted") {
        new Notification(`Price Alert: ${a.symbol}`, {
          body: `${a.symbol} is now ${current.toFixed(2)} (${a.direction} ${a.targetPrice})`,
          icon: "/favicon.ico",
        });
      }
      return { ...a, triggered: true };
    });
    if (changed) { setAlerts(updated); saveAlerts(updated); }
  }, [alerts, latestPrices]);

  function addAlert() {
    const p = parseFloat(price);
    if (!symbol.trim() || isNaN(p)) return;
    const a: Alert = { id: Date.now().toString(), symbol: symbol.trim().toUpperCase(), targetPrice: p, direction, triggered: false };
    const next = [...alerts, a];
    setAlerts(next);
    saveAlerts(next);
    setSymbol(""); setPrice("");
  }

  function removeAlert(id: string) {
    const next = alerts.filter((a) => a.id !== id);
    setAlerts(next);
    saveAlerts(next);
  }

  function requestPermission() {
    if ("Notification" in window) Notification.requestPermission();
  }

  const activeCount = alerts.filter((a) => !a.triggered).length;

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className={cn(
          "relative px-2 py-1 rounded border text-xs transition-colors",
          activeCount > 0 ? "border-amber-600 text-amber-400 hover:border-amber-400" : "border-slate-700 text-slate-400 hover:text-white hover:border-slate-500"
        )}
      >
        🔔 Alerts
        {activeCount > 0 && (
          <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-amber-500 text-black text-[10px] font-bold flex items-center justify-center">
            {activeCount}
          </span>
        )}
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setOpen(false)}>
          <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-md mx-4 p-5" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-white font-bold text-sm">🔔 Price Alerts</h2>
              <button onClick={() => setOpen(false)} className="text-slate-500 hover:text-white">✕</button>
            </div>

            {"Notification" in window && Notification.permission === "default" && (
              <div className="mb-3 p-2 rounded-lg bg-amber-900/40 border border-amber-700 text-amber-300 text-xs flex items-center justify-between">
                <span>Enable browser notifications to receive alerts</span>
                <button onClick={requestPermission} className="ml-2 px-2 py-1 rounded bg-amber-600 text-white text-xs">Enable</button>
              </div>
            )}

            {/* Add alert form */}
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                placeholder="Symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="flex-1 bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded px-2 py-1.5 uppercase"
              />
              <select
                value={direction}
                onChange={(e) => setDirection(e.target.value as "above" | "below")}
                className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded px-2 py-1.5"
              >
                <option value="above">Above</option>
                <option value="below">Below</option>
              </select>
              <input
                type="number"
                placeholder="Price"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                className="w-24 bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded px-2 py-1.5"
              />
              <button
                onClick={addAlert}
                className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs rounded font-semibold"
              >
                Add
              </button>
            </div>

            {/* Alert list */}
            {alerts.length === 0 ? (
              <p className="text-slate-500 text-xs text-center py-4">No alerts set. Add one above.</p>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {alerts.map((a) => (
                  <div key={a.id} className={cn("flex items-center justify-between p-2 rounded-lg border text-xs", a.triggered ? "border-slate-800 bg-slate-800/30 opacity-50" : "border-slate-700 bg-slate-800")}>
                    <div>
                      <span className="font-semibold text-white">{a.symbol}</span>
                      <span className="text-slate-400 ml-2">{a.direction} <span className="font-mono text-white">{a.targetPrice}</span></span>
                      {a.triggered && <span className="ml-2 text-emerald-400">✓ Triggered</span>}
                    </div>
                    <button onClick={() => removeAlert(a.id)} className="text-slate-600 hover:text-red-400 ml-2">✕</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
