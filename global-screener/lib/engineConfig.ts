// Validated parsing of the engine-API deployment env vars
// (docs/DEPLOYMENT_INTEGRATION_PLAN.md Phase 1/2), so a malformed
// ENGINE_API_URL fails with an actionable message instead of a cryptic
// fetch error at request time.

export type EngineConfig = {
  baseUrl: string;
  token: string | null;
};

export type EngineConfigResult =
  | { ok: true; config: EngineConfig }
  | { ok: false; error: string };

export function getEngineConfig(): EngineConfigResult {
  const raw = process.env.ENGINE_API_URL?.trim();
  if (!raw) {
    return { ok: false, error: "Engine API is not configured (ENGINE_API_URL is unset)." };
  }

  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    return {
      ok: false,
      error: `ENGINE_API_URL is not a valid URL: ${JSON.stringify(raw)} (expected e.g. http://localhost:9000)`,
    };
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return {
      ok: false,
      error: `ENGINE_API_URL must use http:// or https:// (got ${JSON.stringify(parsed.protocol)})`,
    };
  }

  const baseUrl = raw.replace(/\/+$/, "");
  const token = process.env.ENGINE_API_TOKEN?.trim() || null;
  return { ok: true, config: { baseUrl, token } };
}
