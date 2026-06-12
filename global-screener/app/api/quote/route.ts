import { NextRequest, NextResponse } from "next/server";
import { fetchBatchQuotes } from "@/lib/yahoo";

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  if (!symbol) return NextResponse.json({ error: "symbol required" }, { status: 400 });

  try {
    const quotes = await fetchBatchQuotes([symbol]);
    return NextResponse.json(quotes[0] ?? null);
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
