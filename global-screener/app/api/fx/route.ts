import { NextResponse } from "next/server";

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";
const SYMBOLS = ["USDINR=X", "USDAED=X", "USDSAR=X"];

export async function GET() {
  try {
    const params = new URLSearchParams({
      symbols: SYMBOLS.join(","),
      fields: "regularMarketPrice",
    });
    const url = `https://query2.finance.yahoo.com/v7/finance/quote?${params.toString()}`;
    const res = await fetch(url, {
      headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
      next: { revalidate: 3600 },
    });

    const defaults = { USD: 1, INR: 83.5, AED: 3.67, SAR: 3.75 };
    if (!res.ok) return NextResponse.json(defaults, { headers: { "Cache-Control": "public, s-maxage=3600" } });

    const json = await res.json();
    const results: Array<{ symbol: string; regularMarketPrice: number }> =
      json?.quoteResponse?.result ?? [];

    const rateMap = new Map(results.map((r) => [r.symbol, r.regularMarketPrice]));

    return NextResponse.json(
      {
        USD: 1,
        INR: rateMap.get("USDINR=X") ?? defaults.INR,
        AED: rateMap.get("USDAED=X") ?? defaults.AED,
        SAR: rateMap.get("USDSAR=X") ?? defaults.SAR,
      },
      { headers: { "Cache-Control": "public, s-maxage=3600" } }
    );
  } catch {
    return NextResponse.json({ USD: 1, INR: 83.5, AED: 3.67, SAR: 3.75 });
  }
}
