import { Redis } from "@upstash/redis";

let redis: Redis | null = null;

/** Returns a singleton Upstash Redis client, or null when env vars are absent. */
export function getRedis(): Redis | null {
  if (redis) return redis;
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return null;
  redis = new Redis({ url, token });
  return redis;
}

const CACHE_TTL = 600; // 10 minutes

export async function cacheGet<T>(key: string): Promise<T | null> {
  const r = getRedis();
  if (!r) return null;
  try {
    return await r.get<T>(key);
  } catch {
    return null;
  }
}

export async function cacheSet(key: string, value: unknown): Promise<void> {
  const r = getRedis();
  if (!r) return;
  try {
    await r.set(key, value, { ex: CACHE_TTL });
  } catch {
    // non-fatal: continue without caching
  }
}
