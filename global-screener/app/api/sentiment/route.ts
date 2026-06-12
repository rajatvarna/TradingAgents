import { NextRequest, NextResponse } from "next/server";
import { fetchSentiment } from "@/lib/sentiment";
import { cacheGet, cacheSet } from "@/lib/redis";

/** Cache sentiment for 15 minutes — social posts refresh frequently. */
const TTL_KEY = "sent";

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  const name   = req.nextUrl.searchParams.get("name") ?? symbol ?? "";
  if (!symbol) return NextResponse.json({ error: "symbol required" }, { status: 400 });

  const key = `${TTL_KEY}:${symbol}`;
  const cached = await cacheGet(key);
  if (cached) return NextResponse.json(cached);

  try {
    const data = await fetchSentiment(symbol, name);
    await cacheSet(key, data);
    return NextResponse.json(data, {
      headers: { "Cache-Control": "public, s-maxage=900" },
    });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
