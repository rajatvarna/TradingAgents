import { NextRequest, NextResponse } from "next/server";

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  if (!symbol) {
    return NextResponse.json({ error: "symbol required" }, { status: 400 });
  }

  try {
    const url = `https://query2.finance.yahoo.com/v2/finance/news?tickers=${encodeURIComponent(symbol)}&count=10`;
    const res = await fetch(url, {
      headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
      next: { revalidate: 900 },
    });

    if (!res.ok) {
      return NextResponse.json([], { headers: { "Cache-Control": "public, s-maxage=900" } });
    }

    const json = await res.json();
    const items: Array<Record<string, unknown>> = json?.items?.result ?? [];

    const articles = items.map((item) => ({
      title: (item.title as string) ?? "",
      link: (item.link as string) ?? "",
      publisher: (item.publisher as string) ?? "",
      publishedAt: typeof item.providerPublishTime === "number"
        ? new Date((item.providerPublishTime as number) * 1000).toISOString()
        : null,
    }));

    return NextResponse.json(articles, {
      headers: { "Cache-Control": "public, s-maxage=900" },
    });
  } catch {
    return NextResponse.json([], {
      headers: { "Cache-Control": "public, s-maxage=900" },
    });
  }
}
