let _client: import("@upstash/redis").Redis | null = null;

function getClient(): import("@upstash/redis").Redis | null {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;

  if (!url || !token) return null;

  if (!_client) {
    // Lazy import to avoid build-time errors when env vars are absent
    const { Redis } = require("@upstash/redis") as typeof import("@upstash/redis");
    _client = new Redis({ url, token });
  }

  return _client;
}

export async function cacheGet<T>(key: string): Promise<T | null> {
  const client = getClient();
  if (!client) return null;

  try {
    const value = await client.get<T>(key);
    return value ?? null;
  } catch {
    return null;
  }
}

export async function cacheSet<T>(
  key: string,
  value: T,
  ttlSeconds = 3600
): Promise<void> {
  const client = getClient();
  if (!client) return;

  try {
    await client.set(key, value, { ex: ttlSeconds });
  } catch {
    // Graceful fallback — ignore Redis errors
  }
}
