// Server-side proxy to the TradingAgents engine API (api/main.py, deployed
// off-Vercel — see VERCEL_DEPLOYMENT.md). Keeps ENGINE_API_TOKEN out of the
// browser and narrows the publicly reachable surface to a small allow-list,
// so this proxy can never be used to reach secret-touching engine endpoints
// like /env or /vault/refresh. See docs/DEPLOYMENT_INTEGRATION_PLAN.md Phase 2.
import { NextRequest, NextResponse } from "next/server";
import { getEngineConfig } from "@/lib/engineConfig";

// Only these top-level engine path segments are reachable through this proxy.
const ALLOWED_PREFIXES = new Set(["analyze", "status", "reports", "requests", "metrics"]);

function authHeaders(token: string | null): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function buildTarget(path: string[], search: string):
  | { target: string; token: string | null }
  | { error: NextResponse } {
  const result = getEngineConfig();
  if (!result.ok) {
    return { error: NextResponse.json({ error: result.error }, { status: 503 }) };
  }
  const prefix = path[0];
  if (!prefix || !ALLOWED_PREFIXES.has(prefix)) {
    return { error: NextResponse.json({ error: "Not found" }, { status: 404 }) };
  }
  return {
    target: `${result.config.baseUrl}/${path.map(encodeURIComponent).join("/")}${search}`,
    token: result.config.token,
  };
}

async function proxy(target: string, init: RequestInit): Promise<NextResponse> {
  try {
    const upstream = await fetch(target, init);
    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch (err) {
    return NextResponse.json({ error: `Engine request failed: ${String(err)}` }, { status: 502 });
  }
}

export async function GET(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const built = buildTarget(path, req.nextUrl.search);
  if ("error" in built) return built.error;

  return proxy(built.target, { headers: authHeaders(built.token), cache: "no-store" });
}

export async function POST(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const built = buildTarget(path, req.nextUrl.search);
  if ("error" in built) return built.error;

  const bodyText = await req.text();
  return proxy(built.target, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(built.token) },
    body: bodyText,
    cache: "no-store",
  });
}
