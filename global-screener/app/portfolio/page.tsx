"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { PortfolioFill, PortfolioPosition, PortfolioStatus, getPortfolioStatus } from "@/lib/engine";
import { cn } from "@/lib/utils";

const POLL_INTERVAL_MS = 15000;

function StatTile({ label, value, tone }: { label: string; value: string; tone?: "warn" | "ok" }) {
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-4">
      <div className="text-xs text-slate-500">{label}</div>
      <div className={cn("text-2xl font-bold mt-1", tone === "warn" ? "text-red-400" : "text-white")}>
        {value}
      </div>
    </div>
  );
}

function PositionRow({ p }: { p: PortfolioPosition }) {
  return (
    <tr className="border-b border-slate-800">
      <td className="px-3 py-2 font-semibold text-white">{p.symbol}</td>
      <td className="px-3 py-2 text-slate-300 tabular-nums">{p.quantity}</td>
      <td className="px-3 py-2 text-slate-400 tabular-nums">{p.entry !== null ? `$${p.entry.toFixed(2)}` : "—"}</td>
      <td className="px-3 py-2 text-slate-400 tabular-nums">{p.stop !== null ? `$${p.stop.toFixed(2)}` : "—"}</td>
    </tr>
  );
}

function FillRow({ f }: { f: PortfolioFill }) {
  return (
    <tr className="border-b border-slate-800">
      <td className="px-3 py-2 font-semibold text-white">{f.symbol}</td>
      <td className={cn("px-3 py-2 font-medium", f.side === "BUY" ? "text-emerald-400" : "text-red-400")}>{f.side}</td>
      <td className="px-3 py-2 text-slate-300 tabular-nums">{f.quantity}</td>
      <td className="px-3 py-2 text-slate-400 tabular-nums">${f.price.toFixed(2)}</td>
      <td className="px-3 py-2 text-slate-500 text-xs">{new Date(f.filled_at).toLocaleString()}</td>
    </tr>
  );
}

export default function PortfolioPage() {
  const [status, setStatus] = useState<PortfolioStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notConfigured, setNotConfigured] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const refresh = async () => {
      try {
        const s = await getPortfolioStatus();
        if (!cancelled) {
          setStatus(s);
          setError(null);
          setNotConfigured(false);
        }
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Failed to load portfolio";
        if (message.toLowerCase().includes("not configured")) {
          setNotConfigured(true);
        } else {
          setError(message);
        }
      }
    };

    refresh();
    const id = setInterval(refresh, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const anomalyCount = status
    ? Object.values(status.anomalies_7d).reduce((sum, a) => sum + a.count, 0)
    : 0;

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      <header className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-950 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-xl font-bold text-white tracking-tight">
            🌐 GlobalScreener
          </Link>
          <span className="text-xs text-slate-500 hidden sm:block">Portfolio (journal view)</span>
        </div>
        <nav className="flex items-center gap-4 text-sm">
          <Link href="/" className="text-slate-400 hover:text-white transition-colors">Screener</Link>
          <Link href="/analyze" className="text-slate-400 hover:text-white transition-colors">Analyze</Link>
          <Link href="/reports" className="text-slate-400 hover:text-white transition-colors">Reports</Link>
          <Link href="/runs" className="text-slate-400 hover:text-white transition-colors">Runs</Link>
          <Link href="/portfolio" className="text-white font-semibold">Portfolio</Link>
        </nav>
      </header>

      <main className="flex-1 px-4 md:px-6 py-6 max-w-4xl mx-auto w-full space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Portfolio</h1>
          <p className="text-sm text-slate-400">
            Read-only view over the ops live-trading journal — replayed positions and cash, not live broker
            state. Order placement is never exposed here.
          </p>
        </div>

        {notConfigured && (
          <div className="text-sm text-slate-500 border border-dashed border-slate-800 rounded-xl p-6 text-center">
            Portfolio view is not configured. Set <code className="text-slate-300">OPS_JOURNAL_PATH</code> on
            the engine API host to the ops daemon&apos;s journal file to enable this page.
          </div>
        )}

        {error && (
          <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-red-300 text-sm">
            {error}
          </div>
        )}

        {status && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatTile label="Cash" value={`$${status.cash.cash.toLocaleString(undefined, { maximumFractionDigits: 2 })}`} />
              <StatTile label="Open positions" value={String(status.positions.items.length)} />
              <StatTile label="Broker mode" value={status.service.broker_mode} />
              <StatTile
                label="Anomalies (7d)"
                value={String(anomalyCount)}
                tone={anomalyCount > 0 ? "warn" : "ok"}
              />
            </div>

            {(status.halts.daily_halt_today || status.halts.kill_switch_this_week) && (
              <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-red-300 text-sm">
                {status.halts.daily_halt_today && <div>Daily halt is active today.</div>}
                {status.halts.kill_switch_this_week && <div>Kill switch triggered this week.</div>}
              </div>
            )}

            <section className="space-y-2">
              <h2 className="text-sm font-semibold text-slate-300">Positions</h2>
              <div className="rounded-xl border border-slate-700 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900">
                    <tr className="border-b border-slate-700">
                      <th className="px-3 py-2 text-left text-xs text-slate-400">Symbol</th>
                      <th className="px-3 py-2 text-left text-xs text-slate-400">Quantity</th>
                      <th className="px-3 py-2 text-left text-xs text-slate-400">Entry</th>
                      <th className="px-3 py-2 text-left text-xs text-slate-400">Stop</th>
                    </tr>
                  </thead>
                  <tbody>
                    {status.positions.items.length === 0 && (
                      <tr><td colSpan={4} className="px-3 py-4 text-center text-slate-500 text-sm">No open positions.</td></tr>
                    )}
                    {status.positions.items.map((p) => <PositionRow key={p.symbol} p={p} />)}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="space-y-2">
              <h2 className="text-sm font-semibold text-slate-300">Fills Today</h2>
              <div className="rounded-xl border border-slate-700 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900">
                    <tr className="border-b border-slate-700">
                      <th className="px-3 py-2 text-left text-xs text-slate-400">Symbol</th>
                      <th className="px-3 py-2 text-left text-xs text-slate-400">Side</th>
                      <th className="px-3 py-2 text-left text-xs text-slate-400">Quantity</th>
                      <th className="px-3 py-2 text-left text-xs text-slate-400">Price</th>
                      <th className="px-3 py-2 text-left text-xs text-slate-400">Filled At</th>
                    </tr>
                  </thead>
                  <tbody>
                    {status.fills.today.length === 0 && (
                      <tr><td colSpan={5} className="px-3 py-4 text-center text-slate-500 text-sm">No fills today.</td></tr>
                    )}
                    {status.fills.today.map((f, i) => <FillRow key={i} f={f} />)}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
