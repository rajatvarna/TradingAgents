import { NextRequest, NextResponse } from "next/server";
import { fetchFundamentals } from "@/lib/fundamentals";
import { cacheGet, cacheSet } from "@/lib/redis";

/** Cache fundamentals for 30 minutes — they change infrequently. */
const TTL_KEY = "fund";

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  if (!symbol) return NextResponse.json({ error: "symbol required" }, { status: 400 });

  const key = `${TTL_KEY}:${symbol}`;
  const cached = await cacheGet(key);
  if (cached) return NextResponse.json(cached);

  try {
    const data = await fetchFundamentals(symbol);
    await cacheSet(key, data);
    return NextResponse.json(data, {
      headers: { "Cache-Control": "public, s-maxage=1800" },
    });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
