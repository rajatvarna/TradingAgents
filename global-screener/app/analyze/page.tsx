"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  EngineProvider,
  RequestStatus,
  SUPPORTED_PROVIDERS,
  isTerminalStatus,
  getStatus,
  submitAnalysis,
} from "@/lib/engine";
import { cn } from "@/lib/utils";

const POLL_INTERVAL_MS = 3000;

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    pending: "bg-slate-700 text-slate-300",
    running: "bg-amber-900/50 text-amber-300 border border-amber-700",
    completed: "bg-emerald-900/50 text-emerald-300 border border-emerald-700",
    failed: "bg-red-900/50 text-red-300 border border-red-700",
    canceled: "bg-slate-800 text-slate-400 border border-slate-700",
  };
  return (
    <span className={cn("px-2 py-0.5 rounded-full text-xs font-semibold uppercase", styles[status] ?? styles.pending)}>
      {status}
    </span>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense fallback={null}>
      <AnalyzeForm />
    </Suspense>
  );
}

function AnalyzeForm() {
  const searchParams = useSearchParams();

  const [ticker, setTicker] = useState(searchParams.get("ticker") ?? "");
  const [date, setDate] = useState(todayIso());
  const [provider, setProvider] = useState<EngineProvider>("ollama");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [status, setStatus] = useState<RequestStatus | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const startPolling = (id: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    const poll = async () => {
      try {
        const s = await getStatus(id);
        setStatus(s);
        if (isTerminalStatus(s.status) && pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch status");
        if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      }
    };
    poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleanTicker = ticker.trim().toUpperCase();
    if (!cleanTicker) {
      setError("Ticker is required.");
      return;
    }
    setError(null);
    setSubmitting(true);
    setStatus(null);
    try {
      const res = await submitAnalysis({ ticker: cleanTicker, date, llm_provider: provider });
      setRequestId(res.request_id);
      startPolling(res.request_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit analysis");
    } finally {
      setSubmitting(false);
    }
  };

  const isRunning = status ? !isTerminalStatus(status.status) : submitting;

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      <header className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-950 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-xl font-bold text-white tracking-tight">
            🌐 GlobalScreener
          </Link>
          <span className="text-xs text-slate-500 hidden sm:block">AI Analysis</span>
        </div>
        <nav className="flex items-center gap-4 text-sm">
          <Link href="/" className="text-slate-400 hover:text-white transition-colors">Screener</Link>
          <Link href="/analyze" className="text-white font-semibold">Analyze</Link>
          <Link href="/reports" className="text-slate-400 hover:text-white transition-colors">Reports</Link>
          <Link href="/runs" className="text-slate-400 hover:text-white transition-colors">Runs</Link>
        </nav>
      </header>

      <main className="flex-1 px-4 md:px-6 py-6 max-w-3xl mx-auto w-full space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">AI Stock Analysis</h1>
          <p className="text-sm text-slate-400">
            Submit a ticker to the TradingAgents multi-agent engine and watch the analysis run.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="bg-slate-900 border border-slate-700 rounded-xl p-4 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label className="block text-xs text-slate-400 mb-1" htmlFor="ticker">Ticker</label>
              <input
                id="ticker"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                placeholder="e.g. NVDA"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder:text-slate-600 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1" htmlFor="date">Date</label>
              <input
                id="date"
                type="date"
                value={date}
                max={todayIso()}
                onChange={(e) => setDate(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1" htmlFor="provider">Provider</label>
              <select
                id="provider"
                value={provider}
                onChange={(e) => setProvider(e.target.value as EngineProvider)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              >
                {SUPPORTED_PROVIDERS.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={isRunning}
            className={cn(
              "px-4 py-2 rounded-lg text-sm font-semibold transition-colors",
              isRunning
                ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-500 text-white"
            )}
          >
            {isRunning ? "Running…" : "Run Analysis"}
          </button>

          {error && (
            <div className="bg-red-900/40 border border-red-700 rounded-lg px-3 py-2 text-red-300 text-sm">
              {error}
            </div>
          )}
        </form>

        {status && (
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold text-white">{status.ticker}</span>
                <span className="text-xs text-slate-500">{status.analysis_date}</span>
              </div>
              <StatusBadge status={status.status} />
            </div>

            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-slate-400">
              <div>
                <dt className="text-slate-500">Request</dt>
                <dd className="text-slate-300 font-mono truncate">{requestId}</dd>
              </div>
              <div>
                <dt className="text-slate-500">Provider</dt>
                <dd className="text-slate-300">{status.llm_provider ?? "—"}</dd>
              </div>
              <div>
                <dt className="text-slate-500">Submitted</dt>
                <dd className="text-slate-300">{new Date(status.submitted_at).toLocaleTimeString()}</dd>
              </div>
              <div>
                <dt className="text-slate-500">Tokens</dt>
                <dd className="text-slate-300">{status.total_tokens ?? "—"}</dd>
              </div>
            </dl>

            {status.status === "failed" && status.error_message && (
              <div className="bg-red-900/40 border border-red-700 rounded-lg px-3 py-2 text-red-300 text-xs whitespace-pre-wrap">
                {status.error_message}
              </div>
            )}

            {status.status === "completed" && (
              <div className="space-y-3">
                {status.recommendation && (
                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                    <div className="text-xs text-slate-500 mb-1">Recommendation</div>
                    <div className="text-sm text-slate-200 whitespace-pre-wrap">{status.recommendation}</div>
                  </div>
                )}
                <Link
                  href={`/reports/${encodeURIComponent(status.ticker)}/${encodeURIComponent(status.analysis_date)}`}
                  className="inline-block px-3 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white text-xs font-semibold transition-colors"
                >
                  View full report →
                </Link>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
