import { NextRequest, NextResponse } from "next/server";

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  if (!symbol) {
    return NextResponse.json({ error: "symbol required" }, { status: 400 });
  }

  try {
    const url = `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(symbol)}?modules=defaultKeyStatistics,financialData`;
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
        },
        { headers: { "Cache-Control": "public, s-maxage=3600" } }
      );
    }

    const json = await res.json();
    const ks = json?.quoteSummary?.result?.[0]?.defaultKeyStatistics ?? {};
    const fd = json?.quoteSummary?.result?.[0]?.financialData ?? {};

    function raw(obj: Record<string, unknown>, key: string): number | null {
      const v = (obj[key] as Record<string, unknown> | undefined)?.raw;
      return typeof v === "number" ? v : null;
    }

    return NextResponse.json(
      {
        shortPercentOfFloat: raw(ks, "shortPercentOfFloat"),
        shortRatio: raw(ks, "shortRatio"),
        forwardPE: raw(ks, "forwardPE"),
        pegRatio: raw(ks, "pegRatio"),
        priceToBook: raw(ks, "priceToBook"),
        returnOnEquity: raw(fd, "returnOnEquity"),
        debtToEquity: raw(fd, "debtToEquity"),
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
