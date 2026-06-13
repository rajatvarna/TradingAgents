import { NextRequest, NextResponse } from "next/server";
import { fetchBatchQuotes, fetchHistory } from "@/lib/yahoo";
import { computePerformance } from "@/lib/performance";
import { cacheGet, cacheSet } from "@/lib/redis";
import { getTickers } from "@/lib/watchlist";
import { StockData, Market, BatchResponse } from "@/types";

const REFRESH_MS = 600_000; // 10 min

/** Splits an array into chunks of at most `size` elements. */
function chunk<T>(arr: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < arr.length; i += size) chunks.push(arr.slice(i, i + size));
  return chunks;
}

/** Fetches and assembles StockData for a single ticker, using Redis cache when available. */
async function resolveStock(
  yahooSuffix: string,
  symbol: string,
  name: string,
  market: Market,
  sector: string,
  tvSymbol: string,
  livePrice?: number,
  currency?: string,
  marketCap?: number,
  volume?: number,
  avgVolume?: number,
  fiftyTwoWeekLow?: number,
  fiftyTwoWeekHigh?: number,
  dividendYield?: number,
  exDividendDate?: number,
  beta?: number
): Promise<StockData> {
  const cacheKey = `hist:${yahooSuffix}`;
  let closes = await cacheGet<Array<{ date: string; close: number }>>(cacheKey);

  if (!closes) {
    try {
      closes = await fetchHistory(yahooSuffix);
      await cacheSet(cacheKey, closes);
    } catch {
      closes = [];
    }
  }

  const price = livePrice ?? (closes?.at(-1)?.close ?? null);

  return {
    symbol,
    yahooSuffix,
    name,
    market,
    sector,
    tvSymbol,
    price,
    currency: currency ?? "USD",
    marketCap: marketCap ?? null,
    volume: volume ?? null,
    avgVolume20d: avgVolume ?? null,
    fiftyTwoWeekLow: fiftyTwoWeekLow ?? null,
    fiftyTwoWeekHigh: fiftyTwoWeekHigh ?? null,
    dividendYield: dividendYield ?? null,
    exDividendDate: exDividendDate ?? null,
    beta: beta ?? null,
    performance: price ? computePerformance(closes ?? [], price) : {
      daily: null, wtd: null, mtd: null, ytd: null, one_y: null, three_y: null, five_y: null,
    },
    isStale: !livePrice,
  };
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const marketFilter: Market | "All" = body.market ?? "All";

    const allTickers = getTickers(marketFilter);

    // Batch fetch live quotes (10 per request)
    const quoteMap = new Map<string, { price: number; currency: string; marketCap: number; volume: number; avgVolume: number; fiftyTwoWeekLow: number; fiftyTwoWeekHigh: number; dividendYield: number; exDividendDate: number; beta: number }>();
    const suffixes = allTickers.map((t) => t.yahooSuffix);

    for (const batch of chunk(suffixes, 10)) {
      try {
        const quotes = await fetchBatchQuotes(batch);
        for (const q of quotes) {
          quoteMap.set(q.symbol, {
            price: q.regularMarketPrice,
            currency: q.currency,
            marketCap: q.marketCap,
            volume: q.regularMarketVolume,
            avgVolume: q.averageDailyVolume3Month,
            fiftyTwoWeekLow: q.fiftyTwoWeekLow,
            fiftyTwoWeekHigh: q.fiftyTwoWeekHigh,
            dividendYield: q.trailingAnnualDividendYield,
            exDividendDate: q.exDividendDate,
            beta: q.beta,
          });
        }
      } catch {
        // continue with partial data
      }
    }

    // Resolve each stock with history + performance (parallelised in groups of 5)
    const results: StockData[] = [];
    for (const batch of chunk(allTickers, 5)) {
      const batchResults = await Promise.all(
        batch.map((t) => {
          const q = quoteMap.get(t.yahooSuffix);
          return resolveStock(
            t.yahooSuffix, t.symbol, t.name, t.market, t.sector, t.tvSymbol,
            q?.price, q?.currency, q?.marketCap, q?.volume, q?.avgVolume,
            q?.fiftyTwoWeekLow, q?.fiftyTwoWeekHigh,
            q?.dividendYield, q?.exDividendDate, q?.beta
          );
        })
      );
      results.push(...batchResults);
    }

    const now = new Date();
    const response: BatchResponse = {
      data: results,
      cachedAt: now.toISOString(),
      nextRefresh: new Date(now.getTime() + REFRESH_MS).toISOString(),
    };

    return NextResponse.json(response, {
      headers: { "Cache-Control": "public, s-maxage=600, stale-while-revalidate=60" },
    });
  } catch (err) {
    console.error("[/api/batch]", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
