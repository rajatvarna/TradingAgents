import { NextResponse } from "next/server";
import { cacheGet, cacheSet } from "@/lib/redis";

export async function GET() {
  const cached = await cacheGet("vix");
  if (cached) return NextResponse.json(cached);

  try {
    const url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=%5EVIX&fields=regularMarketPrice,regularMarketChangePercent";
    const res = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; GlobalScreener/1.0)" },
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    const q = json?.quoteResponse?.result?.[0];
    const data = {
      price: q?.regularMarketPrice ?? null,
      changePct: q?.regularMarketChangePercent ?? null,
      fetchedAt: new Date().toISOString(),
    };
    await cacheSet("vix", data);
    return NextResponse.json(data, { headers: { "Cache-Control": "public, s-maxage=300" } });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
