"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ProviderUsage,
  RequestStatus,
  TodayLlmUsage,
  getClosedRequests,
  getOpenRequests,
  getTodayLlmUsage,
} from "@/lib/engine";
import { cn } from "@/lib/utils";

const POLL_INTERVAL_MS = 5000;

function StatusPill({ status }: { status: string }) {
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

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-4">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-2xl font-bold text-white mt-1">{value}</div>
    </div>
  );
}

function OpenRequestRow({ r }: { r: RequestStatus }) {
  return (
    <tr className="border-b border-slate-800">
      <td className="px-3 py-2 font-semibold text-white">{r.ticker}</td>
      <td className="px-3 py-2 text-slate-400">{r.analysis_date}</td>
      <td className="px-3 py-2"><StatusPill status={r.status} /></td>
      <td className="px-3 py-2 text-slate-400">{r.llm_provider ?? "—"}</td>
      <td className="px-3 py-2 text-slate-500 text-xs">{new Date(r.submitted_at).toLocaleTimeString()}</td>
    </tr>
  );
}

function ClosedRequestRow({ r }: { r: RequestStatus }) {
  return (
    <tr className="border-b border-slate-800 hover:bg-slate-800/60">
      <td className="px-3 py-2">
        <Link
          href={`/reports/${encodeURIComponent(r.ticker)}/${encodeURIComponent(r.analysis_date)}`}
          className="font-semibold text-white hover:text-blue-400 transition-colors"
        >
          {r.ticker}
        </Link>
      </td>
      <td className="px-3 py-2 text-slate-400">{r.analysis_date}</td>
      <td className="px-3 py-2"><StatusPill status={r.status} /></td>
      <td className="px-3 py-2 text-slate-300 max-w-[280px] truncate" title={r.recommendation ?? undefined}>
        {r.recommendation ?? r.error_message ?? "—"}
      </td>
      <td className="px-3 py-2 text-slate-400 tabular-nums text-xs">
        {r.estimated_cost_usd !== null ? `$${r.estimated_cost_usd.toFixed(4)}` : "—"}
      </td>
      <td className="px-3 py-2 text-slate-500 text-xs">{r.total_tokens ?? "—"}</td>
    </tr>
  );
}

function ProviderUsageRow({ p }: { p: ProviderUsage }) {
  return (
    <tr className="border-b border-slate-800">
      <td className="px-3 py-2 font-semibold text-white">{p.llm_provider}</td>
      <td className="px-3 py-2 text-slate-300 tabular-nums">{p.llm_calls}</td>
      <td className="px-3 py-2 text-slate-400 tabular-nums text-xs">{p.tokens_in.toLocaleString()}</td>
      <td className="px-3 py-2 text-slate-400 tabular-nums text-xs">{p.tokens_out.toLocaleString()}</td>
      <td className="px-3 py-2 text-slate-400 tabular-nums text-xs">{p.total_tokens.toLocaleString()}</td>
    </tr>
  );
}

export default function RunsPage() {
  const [open, setOpen] = useState<RequestStatus[] | null>(null);
  const [closed, setClosed] = useState<RequestStatus[] | null>(null);
  const [usage, setUsage] = useState<TodayLlmUsage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    const refresh = async () => {
      try {
        const [openRes, closedRes, usageRes] = await Promise.all([
          getOpenRequests(),
          getClosedRequests(),
          getTodayLlmUsage(),
        ]);
        if (cancelled) return;
        setOpen(openRes.requests);
        setClosed(closedRes.requests);
        setUsage(usageRes);
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load run data");
      } finally {
        // Recursive setTimeout (not setInterval) so the next poll only
        // starts after this one finishes — avoids overlapping requests and
        // out-of-order responses racing each other if the API is slow.
        if (!cancelled) timeoutId = setTimeout(refresh, POLL_INTERVAL_MS);
      }
    };

    refresh();
    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, []);

  const totalCostToday = closed
    ?.filter((r) => r.completed_at && new Date(r.completed_at).toDateString() === new Date().toDateString())
    .reduce((sum, r) => sum + (r.estimated_cost_usd ?? 0), 0);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      <header className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-950 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-xl font-bold text-white tracking-tight">
            🌐 GlobalScreener
          </Link>
          <span className="text-xs text-slate-500 hidden sm:block">Runs & Usage</span>
        </div>
        <nav className="flex items-center gap-4 text-sm">
          <Link href="/" className="text-slate-400 hover:text-white transition-colors">Screener</Link>
          <Link href="/analyze" className="text-slate-400 hover:text-white transition-colors">Analyze</Link>
          <Link href="/reports" className="text-slate-400 hover:text-white transition-colors">Reports</Link>
          <Link href="/runs" className="text-white font-semibold">Runs</Link>
          <Link href="/portfolio" className="text-slate-400 hover:text-white transition-colors">Portfolio</Link>
        </nav>
      </header>

      <main className="flex-1 px-4 md:px-6 py-6 max-w-5xl mx-auto w-full space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Runs & Usage</h1>
          <p className="text-sm text-slate-400">
            Open and recently completed analysis requests, and today&apos;s LLM usage across providers.
          </p>
        </div>

        {error && (
          <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-red-300 text-sm">
            {error}
          </div>
        )}

        {usage && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatTile label="LLM calls today" value={String(usage.total_llm_calls)} />
            <StatTile label="Providers active" value={String(usage.providers.length)} />
            <StatTile label="Open requests" value={String(open?.length ?? 0)} />
            <StatTile
              label="Est. cost today"
              value={totalCostToday !== undefined ? `$${totalCostToday.toFixed(4)}` : "—"}
            />
          </div>
        )}

        <section className="space-y-2">
          <h2 className="text-sm font-semibold text-slate-300">Open Requests</h2>
          <div className="rounded-xl border border-slate-700 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-900">
                <tr className="border-b border-slate-700">
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Ticker</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Date</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Status</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Provider</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Submitted</th>
                </tr>
              </thead>
              <tbody>
                {open === null && (
                  <tr><td colSpan={5} className="px-3 py-4 text-center text-slate-500 text-sm">Loading…</td></tr>
                )}
                {open !== null && open.length === 0 && (
                  <tr><td colSpan={5} className="px-3 py-4 text-center text-slate-500 text-sm">No open requests.</td></tr>
                )}
                {open?.map((r) => <OpenRequestRow key={r.request_id} r={r} />)}
              </tbody>
            </table>
          </div>
        </section>

        <section className="space-y-2">
          <h2 className="text-sm font-semibold text-slate-300">Recently Completed</h2>
          <div className="rounded-xl border border-slate-700 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-900">
                <tr className="border-b border-slate-700">
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Ticker</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Date</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Status</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Recommendation</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Cost</th>
                  <th className="px-3 py-2 text-left text-xs text-slate-400">Tokens</th>
                </tr>
              </thead>
              <tbody>
                {closed === null && (
                  <tr><td colSpan={6} className="px-3 py-4 text-center text-slate-500 text-sm">Loading…</td></tr>
                )}
                {closed !== null && closed.length === 0 && (
                  <tr><td colSpan={6} className="px-3 py-4 text-center text-slate-500 text-sm">No completed requests yet.</td></tr>
                )}
                {closed?.slice(0, 25).map((r) => <ClosedRequestRow key={r.request_id} r={r} />)}
              </tbody>
            </table>
          </div>
        </section>

        {usage && usage.providers.length > 0 && (
          <section className="space-y-2">
            <h2 className="text-sm font-semibold text-slate-300">Today&apos;s LLM Usage by Provider</h2>
            <div className="rounded-xl border border-slate-700 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-slate-900">
                  <tr className="border-b border-slate-700">
                    <th className="px-3 py-2 text-left text-xs text-slate-400">Provider</th>
                    <th className="px-3 py-2 text-left text-xs text-slate-400">Calls</th>
                    <th className="px-3 py-2 text-left text-xs text-slate-400">Tokens In</th>
                    <th className="px-3 py-2 text-left text-xs text-slate-400">Tokens Out</th>
                    <th className="px-3 py-2 text-left text-xs text-slate-400">Total Tokens</th>
                  </tr>
                </thead>
                <tbody>
                  {usage.providers.map((p) => <ProviderUsageRow key={p.llm_provider} p={p} />)}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
