"use client";

import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { StockData } from "@/types";
import { cn } from "@/lib/utils";

interface Position {
  symbol: string;
  shares: number;
  costBasis: number;
  costCurrency: string;
}

interface Props {
  stocks: StockData[];
}

const STORAGE_KEY = "portfolio";

function loadPortfolio(): Position[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as Position[];
  } catch {
    // ignore
  }
  return [];
}

function savePortfolio(positions: Position[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(positions));
  } catch {
    // ignore
  }
}

interface FxRates {
  USD: number;
  INR: number;
  AED: number;
  SAR: number;
}

function getCurrencyRate(currency: string, fxRates: FxRates): number {
  if (currency === "USD") return 1;
  if (currency === "INR") return 1 / fxRates.INR;
  if (currency === "AED") return 1 / fxRates.AED;
  if (currency === "SAR") return 1 / fxRates.SAR;
  return 1;
}

function fmtUSD(value: number): string {
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
}

function fmtPctLocal(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

export default function PortfolioTracker({ stocks }: Props) {
  const [open, setOpen] = useState(true);
  const [positions, setPositions] = useState<Position[]>([]);
  const [symbol, setSymbol] = useState("");
  const [shares, setShares] = useState("");
  const [cost, setCost] = useState("");
  const [costCurrency, setCostCurrency] = useState("USD");
  const [error, setError] = useState("");

  // Load from localStorage on mount
  useEffect(() => {
    setPositions(loadPortfolio());
  }, []);

  const { data: fxRates } = useQuery<FxRates>({
    queryKey: ["fx"],
    queryFn: async () => {
      const res = await fetch("/api/fx");
      return res.json() as Promise<FxRates>;
    },
    staleTime: 60 * 60 * 1000,
  });

  const getStockPrice = useCallback(
    (sym: string): { price: number | null; currency: string } => {
      const stock = stocks.find((s) => s.symbol.toUpperCase() === sym.toUpperCase());
      if (!stock || stock.price === null) return { price: null, currency: "USD" };
      return { price: stock.price, currency: stock.currency };
    },
    [stocks]
  );

  const addPosition = () => {
    const sym = symbol.trim().toUpperCase();
    if (!sym) { setError("Symbol required"); return; }
    const sh = parseFloat(shares);
    const cb = parseFloat(cost);
    if (isNaN(sh) || sh <= 0) { setError("Enter valid shares"); return; }
    if (isNaN(cb) || cb < 0) { setError("Enter valid cost basis"); return; }

    setError("");
    const next: Position[] = [...positions, { symbol: sym, shares: sh, costBasis: cb, costCurrency }];
    setPositions(next);
    savePortfolio(next);
    setSymbol("");
    setShares("");
    setCost("");
  };

  const removePosition = (idx: number) => {
    const next = positions.filter((_, i) => i !== idx);
    setPositions(next);
    savePortfolio(next);
  };

  const rates = fxRates ?? { USD: 1, INR: 83.5, AED: 3.67, SAR: 3.75 };

  // Compute rows — all monetary values in USD
  const rows = positions.map((pos) => {
    const { price, currency } = getStockPrice(pos.symbol);
    const currentRateToUSD = getCurrencyRate(currency, rates);
    const currentPriceUSD = price !== null ? price * currentRateToUSD : null;
    // Convert cost basis from its entry currency to USD
    const costRateToUSD = getCurrencyRate(pos.costCurrency ?? "USD", rates);
    const totalCost = pos.costBasis * pos.shares * costRateToUSD;
    const totalValue = currentPriceUSD !== null ? currentPriceUSD * pos.shares : null;
    const pnlDollar = totalValue !== null ? totalValue - totalCost : null;
    const pnlPct = totalCost > 0 && pnlDollar !== null ? (pnlDollar / totalCost) * 100 : null;
    return { pos, totalCost, totalValue, pnlDollar, pnlPct, currentPriceUSD };
  });

  const grandCost = rows.reduce((s, r) => s + r.totalCost, 0);
  const grandValue = rows.reduce((s, r) => s + (r.totalValue ?? r.totalCost), 0);
  const grandPnl = grandValue - grandCost;
  const grandWeight = grandValue > 0 ? grandValue : 1;

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 overflow-hidden">
      {/* Header */}
      <button
        className="w-full flex items-center justify-between px-4 py-3 border-b border-slate-700 hover:bg-slate-800/40 transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="font-semibold text-white text-sm">
          Portfolio Tracker
          {rows.length > 0 && (
            <span className="ml-2 text-slate-400 font-normal text-xs">
              {rows.length} position{rows.length !== 1 ? "s" : ""} · {fmtUSD(grandValue)} total value
            </span>
          )}
        </span>
        <span className="text-slate-400 text-xs">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="p-4 space-y-4">
          {/* Add position form */}
          <div className="flex flex-wrap gap-2 items-end">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-500">Symbol</label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="AAPL"
                className="bg-slate-800 border border-slate-600 text-white text-sm rounded px-3 py-1.5 w-28 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-500">Shares</label>
              <input
                type="number"
                value={shares}
                onChange={(e) => setShares(e.target.value)}
                placeholder="100"
                min="0"
                className="bg-slate-800 border border-slate-600 text-white text-sm rounded px-3 py-1.5 w-24 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-500">Cost/Share</label>
              <div className="flex gap-1">
                <input
                  type="number"
                  value={cost}
                  onChange={(e) => setCost(e.target.value)}
                  placeholder="150.00"
                  min="0"
                  step="0.01"
                  className="bg-slate-800 border border-slate-600 text-white text-sm rounded px-3 py-1.5 w-28 focus:outline-none focus:border-blue-500"
                />
                <select
                  value={costCurrency}
                  onChange={(e) => setCostCurrency(e.target.value)}
                  className="bg-slate-800 border border-slate-600 text-slate-300 text-xs rounded px-2 py-1.5"
                >
                  <option value="USD">USD</option>
                  <option value="INR">INR</option>
                  <option value="AED">AED</option>
                  <option value="SAR">SAR</option>
                </select>
              </div>
            </div>
            <button
              onClick={addPosition}
              className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded font-medium transition-colors"
            >
              Add
            </button>
            {error && <span className="text-red-400 text-xs self-end pb-1">{error}</span>}
          </div>

          {/* Holdings table */}
          {rows.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-800/60 border-b border-slate-700">
                  <tr>
                    {["Symbol", "Shares", "Cost (USD equiv.)", "Current", "P&L $", "P&L %", "Weight%", ""].map((h) => (
                      <th key={h} className="px-3 py-2 text-left text-xs font-semibold text-slate-400 whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => {
                    const weight = (row.totalValue ?? row.totalCost) / grandWeight * 100;
                    const pnlColor = row.pnlDollar === null
                      ? "text-slate-400"
                      : row.pnlDollar >= 0
                      ? "text-emerald-400"
                      : "text-red-400";

                    return (
                      <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/40">
                        <td className="px-3 py-2 font-semibold text-white">{row.pos.symbol}</td>
                        <td className="px-3 py-2 text-slate-300 tabular-nums font-mono">
                          {row.pos.shares.toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-slate-300 tabular-nums font-mono">
                          {fmtUSD(row.totalCost)}
                          {row.pos.costCurrency && row.pos.costCurrency !== "USD" && (
                            <span className="ml-1 text-[10px] text-slate-600">{row.pos.costCurrency}</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-slate-300 tabular-nums font-mono">
                          {row.currentPriceUSD !== null ? fmtUSD(row.currentPriceUSD) : "—"}
                        </td>
                        <td className={cn("px-3 py-2 tabular-nums font-mono", pnlColor)}>
                          {row.pnlDollar !== null ? fmtUSD(row.pnlDollar) : "—"}
                        </td>
                        <td className={cn("px-3 py-2 tabular-nums font-mono", pnlColor)}>
                          {row.pnlPct !== null ? fmtPctLocal(row.pnlPct) : "—"}
                        </td>
                        <td className="px-3 py-2 text-slate-400 tabular-nums text-xs">
                          {weight.toFixed(1)}%
                        </td>
                        <td className="px-3 py-2">
                          <button
                            onClick={() => removePosition(i)}
                            className="text-slate-600 hover:text-red-400 transition-colors text-xs px-1"
                            aria-label="Remove position"
                          >
                            ✕
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
                {/* Footer totals */}
                <tfoot className="border-t border-slate-600 bg-slate-800/30">
                  <tr>
                    <td className="px-3 py-2 text-xs font-semibold text-slate-400" colSpan={2}>Total</td>
                    <td className="px-3 py-2 text-xs font-mono text-slate-300">{fmtUSD(grandCost)}</td>
                    <td className="px-3 py-2 text-xs font-mono text-slate-300">{fmtUSD(grandValue)}</td>
                    <td className={cn("px-3 py-2 text-xs font-mono", grandPnl >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {fmtUSD(grandPnl)}
                    </td>
                    <td className={cn("px-3 py-2 text-xs font-mono", grandPnl >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {grandCost > 0 ? fmtPctLocal((grandPnl / grandCost) * 100) : "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-slate-500">100%</td>
                    <td />
                  </tr>
                </tfoot>
              </table>
            </div>
          ) : (
            <div className="text-center text-slate-500 text-sm py-6">
              No positions yet — add a symbol above
            </div>
          )}
        </div>
      )}
    </div>
  );
}
