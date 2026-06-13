import { NextRequest, NextResponse } from "next/server";

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  if (!symbol) {
    return NextResponse.json({ error: "symbol required" }, { status: 400 });
  }

  try {
    const url = `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(symbol)}?modules=defaultKeyStatistics,financialData,recommendationTrend`;
    const res = await fetch(url, {
      headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
      next: { revalidate: 3600 },
    });

    if (!res.ok) {
      return NextResponse.json(
        {
          shortPercentOfFloat: null,
          shortRatio: null,
          forwardPE: null,
          pegRatio: null,
          priceToBook: null,
          returnOnEquity: null,
          debtToEquity: null,
          analystRatings: null,
        },
        { headers: { "Cache-Control": "public, s-maxage=3600" } }
      );
    }

    const json = await res.json();
    const ks = json?.quoteSummary?.result?.[0]?.defaultKeyStatistics ?? {};
    const fd = json?.quoteSummary?.result?.[0]?.financialData ?? {};
    const rt = json?.quoteSummary?.result?.[0]?.recommendationTrend?.trend ?? [];

    function raw(obj: Record<string, unknown>, key: string): number | null {
      const v = (obj[key] as Record<string, unknown> | undefined)?.raw;
      return typeof v === "number" ? v : null;
    }

    const latestTrend = (rt as Record<string, number>[])[0] ?? null;
    const analystRatings = latestTrend
      ? {
          strongBuy: latestTrend.strongBuy ?? 0,
          buy: latestTrend.buy ?? 0,
          hold: latestTrend.hold ?? 0,
          underperform: latestTrend.sell ?? 0,
          sell: latestTrend.strongSell ?? 0,
        }
      : null;

    return NextResponse.json(
      {
        shortPercentOfFloat: raw(ks, "shortPercentOfFloat"),
        shortRatio: raw(ks, "shortRatio"),
        forwardPE: raw(ks, "forwardPE"),
        pegRatio: raw(ks, "pegRatio"),
        priceToBook: raw(ks, "priceToBook"),
        returnOnEquity: raw(fd, "returnOnEquity"),
        debtToEquity: raw(fd, "debtToEquity"),
        analystRatings,
      },
      { headers: { "Cache-Control": "public, s-maxage=3600" } }
    );
  } catch {
    return NextResponse.json({
      shortPercentOfFloat: null,
      shortRatio: null,
      forwardPE: null,
      pegRatio: null,
      priceToBook: null,
      returnOnEquity: null,
      debtToEquity: null,
    });
  }
}
