import { NextRequest, NextResponse } from "next/server";

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";

export interface InsiderTransaction {
  name: string;
  relation: string;
  transactionDate: string;
  transactionType: string;
  shares: number;
  value: number | null;
}

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  if (!symbol) {
    return NextResponse.json({ error: "symbol required" }, { status: 400 });
  }

  try {
    const url = `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(symbol)}?modules=insiderTransactions`;
    const res = await fetch(url, {
      headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
      next: { revalidate: 3600 },
    });

    if (!res.ok) {
      return NextResponse.json([], { headers: { "Cache-Control": "public, s-maxage=3600" } });
    }

    const json = await res.json();
    const raw: Record<string, unknown>[] =
      json?.quoteSummary?.result?.[0]?.insiderTransactions?.transactions ?? [];

    const transactions: InsiderTransaction[] = raw.slice(0, 5).map((t) => {
      const dateRaw = (t.startDate as Record<string, unknown>)?.raw;
      const sharesRaw = (t.shares as Record<string, unknown>)?.raw;
      const valueRaw = (t.value as Record<string, unknown>)?.raw;
      return {
        name: String(t.filerName ?? ""),
        relation: String(t.filerRelation ?? ""),
        transactionDate:
          typeof dateRaw === "number"
            ? new Date(dateRaw * 1000).toISOString().split("T")[0]
            : "",
        transactionType: String(t.transactionText ?? ""),
        shares: typeof sharesRaw === "number" ? sharesRaw : 0,
        value: typeof valueRaw === "number" ? valueRaw : null,
      };
    });

    return NextResponse.json(transactions, {
      headers: { "Cache-Control": "public, s-maxage=3600" },
    });
  } catch {
    return NextResponse.json([]);
  }
}
