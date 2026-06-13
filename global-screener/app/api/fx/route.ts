import { NextResponse } from "next/server";

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";

async function fetchRate(symbol: string): Promise<number | null> {
  const params = new URLSearchParams({
    symbols: symbol,
    fields: "regularMarketPrice",
  });
  const url = `https://query2.finance.yahoo.com/v7/finance/quote?${params.toString()}`;
  const res = await fetch(url, {
    headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
    next: { revalidate: 3600 },
  });
  if (!res.ok) return null;
  const json = await res.json();
  const price = json?.quoteResponse?.result?.[0]?.regularMarketPrice;
  return typeof price === "number" ? price : null;
}

export async function GET() {
  try {
    const [inr, aed, sar] = await Promise.all([
      fetchRate("USDINR=X"),
      fetchRate("USDAED=X"),
      fetchRate("USDSAR=X"),
    ]);

    return NextResponse.json(
      {
        USD: 1,
        INR: inr ?? 83.5,
        AED: aed ?? 3.67,
        SAR: sar ?? 3.75,
      },
      { headers: { "Cache-Control": "public, s-maxage=3600" } }
    );
  } catch {
    return NextResponse.json({ USD: 1, INR: 83.5, AED: 3.67, SAR: 3.75 });
  }
}
