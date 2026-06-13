import { NextRequest, NextResponse } from "next/server";

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  if (!symbol) {
    return NextResponse.json({ error: "symbol required" }, { status: 400 });
  }

  try {
    const url = `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(symbol)}?modules=calendarEvents`;
    const res = await fetch(url, {
      headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
      next: { revalidate: 3600 },
    });

    if (!res.ok) {
      return NextResponse.json({ nextEarningsDate: null, epsEstimate: null });
    }

    const json = await res.json();
    const cal = json?.quoteSummary?.result?.[0]?.calendarEvents;

    let nextEarningsDate: string | null = null;
    let epsEstimate: number | null = null;

    if (cal) {
      const rawDates: Array<number | { raw: number }> = cal?.earnings?.earningsDate ?? [];
      if (rawDates.length > 0) {
        const entry = rawDates[0];
        const ts = typeof entry === "object" && entry !== null ? entry.raw : entry;
        if (typeof ts === "number" && !isNaN(ts)) {
          nextEarningsDate = new Date(ts * 1000).toISOString().split("T")[0];
        }
      }
      const eps = cal?.earnings?.earningsAverage?.raw ?? null;
      epsEstimate = typeof eps === "number" ? eps : null;
    }

    return NextResponse.json(
      { nextEarningsDate, epsEstimate },
      { headers: { "Cache-Control": "public, s-maxage=3600" } }
    );
  } catch {
    return NextResponse.json({ nextEarningsDate: null, epsEstimate: null });
  }
}
