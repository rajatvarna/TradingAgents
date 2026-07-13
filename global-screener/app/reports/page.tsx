"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ReportSummary, listReports } from "@/lib/engine";
import { cn } from "@/lib/utils";

function ratingColor(rating: string): string {
  const r = rating.toLowerCase();
  if (r.includes("buy") || r.includes("bull")) return "text-emerald-400";
  if (r.includes("sell") || r.includes("bear")) return "text-red-400";
  return "text-slate-300";
}

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listReports()
      .then(setReports)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load reports"));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      <header className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-950 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-xl font-bold text-white tracking-tight">
            🌐 GlobalScreener
          </Link>
          <span className="text-xs text-slate-500 hidden sm:block">Report Archive</span>
        </div>
        <nav className="flex items-center gap-4 text-sm">
          <Link href="/" className="text-slate-400 hover:text-white transition-colors">Screener</Link>
          <Link href="/analyze" className="text-slate-400 hover:text-white transition-colors">Analyze</Link>
          <Link href="/reports" className="text-white font-semibold">Reports</Link>
          <Link href="/runs" className="text-slate-400 hover:text-white transition-colors">Runs</Link>
          <Link href="/portfolio" className="text-slate-400 hover:text-white transition-colors">Portfolio</Link>
        </nav>
      </header>

      <main className="flex-1 px-4 md:px-6 py-6 max-w-4xl mx-auto w-full space-y-4">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Analysis Reports</h1>
          <p className="text-sm text-slate-400">
            Persisted per-ticker analyses from the TradingAgents engine, most recent first.
          </p>
        </div>

        {error && (
          <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-red-300 text-sm">
            {error}
          </div>
        )}

        {!error && reports === null && (
          <div className="text-sm text-slate-500">Loading reports…</div>
        )}

        {reports !== null && reports.length === 0 && !error && (
          <div className="text-sm text-slate-500 border border-dashed border-slate-800 rounded-xl p-6 text-center">
            No reports yet. Run an analysis from the{" "}
            <Link href="/analyze" className="text-blue-400 hover:underline">Analyze</Link> page.
          </div>
        )}

        {reports && reports.length > 0 && (
          <div className="rounded-xl border border-slate-700 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-900">
                <tr className="border-b border-slate-700">
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Ticker</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Date</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Rating</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Action</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((r) => (
                  <tr key={`${r.ticker}-${r.date}`} className="border-b border-slate-800 hover:bg-slate-800/60">
                    <td className="px-3 py-2">
                      <Link
                        href={`/reports/${encodeURIComponent(r.ticker)}/${encodeURIComponent(r.date)}`}
                        className="font-semibold text-white hover:text-blue-400 transition-colors"
                      >
                        {r.ticker}
                      </Link>
                    </td>
                    <td className="px-3 py-2 text-slate-400">{r.date}</td>
                    <td className={cn("px-3 py-2 font-medium", ratingColor(r.rating))}>{r.rating}</td>
                    <td className="px-3 py-2 text-slate-300">{r.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}
