// Client helpers for the TradingAgents engine API, reached through the
// server-side proxy at app/api/engine/[...path]/route.ts (never called
// directly from the browser — see docs/DEPLOYMENT_INTEGRATION_PLAN.md Phase 2).

export type SubmitResponse = {
  request_id: string;
  ticker: string;
  analysis_date: string;
  llm_provider: string;
  status: string;
  submitted_at: string;
};

export type AgentRecommendation = {
  label: string;
  weight: number;
  recommendation: string;
};

export type RequestStatus = {
  request_id: string;
  ticker: string;
  analysis_date: string;
  llm_provider: string | null;
  deep_model: string | null;
  quick_model: string | null;
  status: string;
  submitted_at: string;
  started_at: string | null;
  completed_at: string | null;
  recommendation: string | null;
  llm_calls: number | null;
  tool_calls: number | null;
  tokens_in: number | null;
  tokens_out: number | null;
  total_tokens: number | null;
  estimated_cost_usd: number | null;
  agent_recommendations: Record<string, AgentRecommendation> | null;
  analysis_url: string | null;
  debug_log_url: string | null;
  error_message: string | null;
};

export type ReportSummary = {
  ticker: string;
  date: string;
  rating: string;
  action: string;
};

export type ReportDetail = {
  ticker: string;
  date: string;
  rating: string | null;
  price_target: string | null;
  time_horizon: string | null;
  action: string | null;
  entry_price: string | null;
  stop_loss: string | null;
  sections: Record<string, string>;
  meta: Record<string, unknown>;
};

export type ReportOutcome = {
  ticker: string;
  date: string;
  rating: string;
  score: number;
  ret_20d: number | null;
  ret_60d: number | null;
  correct_20d: boolean | null;
  correct_60d: boolean | null;
};

export type PortfolioPosition = {
  symbol: string;
  quantity: number;
  entry: number | null;
  stop: number | null;
};

export type PortfolioFill = {
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  filled_at: string;
};

export type PortfolioStatus = {
  service: {
    journal_path: string;
    broker_mode: string;
    last_started: { at: string } | null;
    last_stopping: { at: string } | null;
  };
  positions: { source: string; items: PortfolioPosition[] };
  cash: { cash: number };
  halts: { daily_halt_today: boolean; kill_switch_this_week: boolean };
  fills: { today_count: number; today: PortfolioFill[]; last: PortfolioFill | null };
  live_gate: {
    flip_marker_present: boolean;
    live_buy_fills: number;
    cap: number;
    gate_count: number;
    remaining: number;
  };
  anomalies_7d: Record<string, { count: number; last_at: string | null }>;
};

export type RequestListResponse = {
  total: number;
  requests: RequestStatus[];
};

export type ProviderUsage = {
  llm_provider: string;
  llm_calls: number;
  tokens_in: number;
  tokens_out: number;
  total_tokens: number;
};

export type TodayLlmUsage = {
  date_utc: string;
  total_llm_calls: number;
  providers: ProviderUsage[];
  roles: unknown[];
};

const TERMINAL_STATUSES = new Set(["completed", "failed", "canceled"]);

export function isTerminalStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

const SUPPORTED_PROVIDERS = ["ollama", "google", "openrouter"] as const;
export type EngineProvider = (typeof SUPPORTED_PROVIDERS)[number];
export { SUPPORTED_PROVIDERS };

async function parseOrThrow<T>(res: Response): Promise<T> {
  const text = await res.text();
  let data: unknown;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!res.ok) {
    const detail =
      data && typeof data === "object" && data !== null && ("detail" in data || "error" in data)
        ? String((data as { detail?: string; error?: string }).detail ?? (data as { error?: string }).error)
        : `Engine request failed (${res.status})`;
    throw new Error(detail);
  }
  return data as T;
}

export async function submitAnalysis(body: {
  ticker: string;
  date?: string;
  llm_provider?: EngineProvider;
}): Promise<SubmitResponse> {
  const res = await fetch("/api/engine/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseOrThrow<SubmitResponse>(res);
}

export async function getStatus(requestId: string): Promise<RequestStatus> {
  const res = await fetch(`/api/engine/status/${encodeURIComponent(requestId)}`, { cache: "no-store" });
  return parseOrThrow<RequestStatus>(res);
}

export async function listReports(): Promise<ReportSummary[]> {
  const res = await fetch("/api/engine/reports", { cache: "no-store" });
  return parseOrThrow<ReportSummary[]>(res);
}

export async function getReport(ticker: string, date: string): Promise<ReportDetail> {
  const res = await fetch(
    `/api/engine/reports/${encodeURIComponent(ticker)}/${encodeURIComponent(date)}`,
    { cache: "no-store" }
  );
  return parseOrThrow<ReportDetail>(res);
}

export async function getReportOutcome(ticker: string, date: string): Promise<ReportOutcome> {
  const res = await fetch(
    `/api/engine/reports/${encodeURIComponent(ticker)}/${encodeURIComponent(date)}/outcome`,
    { cache: "no-store" }
  );
  return parseOrThrow<ReportOutcome>(res);
}

export async function getPortfolioStatus(): Promise<PortfolioStatus> {
  const res = await fetch("/api/engine/portfolio", { cache: "no-store" });
  return parseOrThrow<PortfolioStatus>(res);
}

export async function getOpenRequests(): Promise<RequestListResponse> {
  const res = await fetch("/api/engine/requests/open", { cache: "no-store" });
  return parseOrThrow<RequestListResponse>(res);
}

export async function getClosedRequests(): Promise<RequestListResponse> {
  const res = await fetch("/api/engine/requests/closed", { cache: "no-store" });
  return parseOrThrow<RequestListResponse>(res);
}

export async function getTodayLlmUsage(): Promise<TodayLlmUsage> {
  const res = await fetch("/api/engine/metrics/llm-calls/today", { cache: "no-store" });
  return parseOrThrow<TodayLlmUsage>(res);
}
