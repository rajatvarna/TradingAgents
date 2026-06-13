import { NextResponse } from "next/server";

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";

function vixLabel(vix: number): string {
  if (vix < 15) return "Calm";
  if (vix < 20) return "Neutral";
  if (vix < 28) return "Cautious";
  if (vix < 35) return "Fearful";
  return "Extreme Fear";
}

export async function GET() {
  try {
    const url =
      "https://query2.finance.yahoo.com/v7/finance/quote?symbols=%5EVIX&fields=regularMarketPrice";

    const res = await fetch(url, {
      headers: {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
      },
      next: { revalidate: 300 },
    });

    if (!res.ok) {
      return NextResponse.json(
        { error: `Yahoo Finance error ${res.status}` },
        { status: 502 }
      );
    }

    const json = await res.json();
    const quote = json?.quoteResponse?.result?.[0];
    const vix: number = quote?.regularMarketPrice ?? 0;

    return NextResponse.json(
      { vix, label: vixLabel(vix), timestamp: new Date().toISOString() },
      {
        headers: {
          "Cache-Control": "public, s-maxage=300, stale-while-revalidate=60",
        },
      }
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
