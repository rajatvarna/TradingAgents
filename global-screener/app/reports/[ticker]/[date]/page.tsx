"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ReportDetail, getReport } from "@/lib/engine";

const SECTION_LABELS: Record<string, string> = {
  final_decision: "Final Decision",
  trader_plan: "Trader Plan",
  research_plan: "Research Plan",
  market: "Market",
  sentiment: "Sentiment",
  news: "News",
  fundamentals: "Fundamentals",
};

function SummaryStat({ label, value }: { label: string; value: string | null }) {
  return (
    <div>
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="text-sm text-slate-200 font-medium">{value ?? "—"}</dd>
    </div>
  );
}

export default function ReportDetailPage() {
  const params = useParams<{ ticker: string; date: string }>();
  const ticker = decodeURIComponent(String(params.ticker ?? ""));
  const date = decodeURIComponent(String(params.date ?? ""));

  const [report, setReport] = useState<ReportDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ticker || !date) return;
    setReport(null);
    setError(null);
    getReport(ticker, date)
      .then(setReport)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load report"));
  }, [ticker, date]);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      <header className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-950 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-xl font-bold text-white tracking-tight">
            🌐 GlobalScreener
          </Link>
        </div>
        <nav className="flex items-center gap-4 text-sm">
          <Link href="/" className="text-slate-400 hover:text-white transition-colors">Screener</Link>
          <Link href="/analyze" className="text-slate-400 hover:text-white transition-colors">Analyze</Link>
          <Link href="/reports" className="text-white font-semibold">Reports</Link>
        </nav>
      </header>

      <main className="flex-1 px-4 md:px-6 py-6 max-w-3xl mx-auto w-full space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">{ticker}</h1>
            <p className="text-sm text-slate-500">{date}</p>
          </div>
          <Link
            href={`/analyze?ticker=${encodeURIComponent(ticker)}`}
            className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-colors"
          >
            Re-analyze →
          </Link>
        </div>

        {error && (
          <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-red-300 text-sm">
            {error}
          </div>
        )}

        {!error && !report && (
          <div className="text-sm text-slate-500">Loading report…</div>
        )}

        {report && (
          <>
            <div className="bg-slate-900 border border-slate-700 rounded-xl p-4">
              <dl className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <SummaryStat label="Rating" value={report.rating} />
                <SummaryStat label="Action" value={report.action} />
                <SummaryStat label="Price Target" value={report.price_target} />
                <SummaryStat label="Entry Price" value={report.entry_price} />
                <SummaryStat label="Stop Loss" value={report.stop_loss} />
                <SummaryStat label="Time Horizon" value={report.time_horizon} />
              </dl>
            </div>

            <div className="space-y-3">
              {Object.entries(report.sections)
                .filter(([, content]) => content.trim().length > 0)
                .map(([key, content]) => (
                  <details key={key} className="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden" open={key === "final_decision"}>
                    <summary className="px-4 py-3 cursor-pointer text-sm font-semibold text-white hover:bg-slate-800/60">
                      {SECTION_LABELS[key] ?? key}
                    </summary>
                    <div className="px-4 pb-4 text-sm text-slate-300 whitespace-pre-wrap border-t border-slate-800 pt-3">
                      {content}
                    </div>
                  </details>
                ))}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
