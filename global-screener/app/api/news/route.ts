import { NextRequest, NextResponse } from "next/server";

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";

export interface NewsItem {
  uuid: string;
  title: string;
  publisher: string;
  link: string;
  providerPublishTime: number;
  thumbnail: string | null;
}

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  if (!symbol) return NextResponse.json({ error: "symbol required" }, { status: 400 });

  try {
    const url = `https://query2.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(symbol)}&newsCount=10&enableFuzzyQuery=false`;
    const res = await fetch(url, {
      headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
      next: { revalidate: 900 },
    });

    if (!res.ok) return NextResponse.json([], { headers: { "Cache-Control": "public, s-maxage=900" } });

    const json = await res.json();
    const items: NewsItem[] = (json?.news ?? []).map((n: Record<string, unknown>) => ({
      uuid: (n.uuid as string) ?? "",
      title: (n.title as string) ?? "",
      publisher: (n.publisher as string) ?? "",
      link: (n.link as string) ?? "",
      providerPublishTime: (n.providerPublishTime as number) ?? 0,
      thumbnail:
        (n.thumbnail as { resolutions?: { url: string }[] } | undefined)
          ?.resolutions?.[0]?.url ?? null,
    }));

    return NextResponse.json(items, { headers: { "Cache-Control": "public, s-maxage=900" } });
  } catch {
    return NextResponse.json([]);
  }
}
