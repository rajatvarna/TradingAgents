export interface QuoteResult {
  symbol: string;
  regularMarketPrice: number;
  currency: string;
  marketCap: number;
  regularMarketVolume: number;
  averageDailyVolume3Month: number;
  fiftyTwoWeekLow: number;
  fiftyTwoWeekHigh: number;
}

const USER_AGENT = "Mozilla/5.0 (compatible; GlobalScreener/1.0)";

export async function fetchBatchQuotes(symbols: string[]): Promise<QuoteResult[]> {
  if (!symbols.length) return [];

  const params = new URLSearchParams({
    symbols: symbols.join(","),
    fields: "regularMarketPrice,currency,marketCap,regularMarketVolume,averageDailyVolume3Month,fiftyTwoWeekLow,fiftyTwoWeekHigh",
    crumb: "",
  });

  const url = `https://query2.finance.yahoo.com/v7/finance/quote?${params.toString()}`;

  const res = await fetch(url, {
    headers: {
      "User-Agent": USER_AGENT,
      "Accept": "application/json",
    },
    next: { revalidate: 300 },
  });

  if (!res.ok) {
    throw new Error(`Yahoo Finance quote error ${res.status}: ${res.statusText}`);
  }

  const json = await res.json();
  const quotes = json?.quoteResponse?.result ?? [];

  return quotes.map((q: Record<string, unknown>) => ({
    symbol: q.symbol as string,
    regularMarketPrice: (q.regularMarketPrice as number) ?? 0,
    currency: (q.currency as string) ?? "USD",
    marketCap: (q.marketCap as number) ?? 0,
    regularMarketVolume: (q.regularMarketVolume as number) ?? 0,
    averageDailyVolume3Month: (q.averageDailyVolume3Month as number) ?? 0,
    fiftyTwoWeekLow: (q.fiftyTwoWeekLow as number) ?? 0,
    fiftyTwoWeekHigh: (q.fiftyTwoWeekHigh as number) ?? 0,
  }));
}

export async function fetchHistory(
  symbol: string
): Promise<Array<{ date: string; close: number }>> {
  const now = Math.floor(Date.now() / 1000);
  const fiveYearsAgo = now - 5 * 365 * 24 * 60 * 60;

  const params = new URLSearchParams({
    period1: String(fiveYearsAgo),
    period2: String(now),
    interval: "1d",
    events: "history",
    includeAdjustedClose: "true",
  });

  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?${params.toString()}`;

  const res = await fetch(url, {
    headers: {
      "User-Agent": USER_AGENT,
      "Accept": "application/json",
    },
    next: { revalidate: 3600 },
  });

  if (!res.ok) {
    throw new Error(`Yahoo Finance history error ${res.status} for ${symbol}`);
  }

  const json = await res.json();
  const result = json?.chart?.result?.[0];
  if (!result) return [];

  const timestamps: number[] = result.timestamp ?? [];
  const closes: number[] = result.indicators?.adjclose?.[0]?.adjclose ?? result.indicators?.quote?.[0]?.close ?? [];

  const history: Array<{ date: string; close: number }> = [];
  for (let i = 0; i < timestamps.length; i++) {
    const close = closes[i];
    if (close === null || close === undefined || isNaN(close)) continue;
    const date = new Date(timestamps[i] * 1000).toISOString().split("T")[0];
    history.push({ date, close });
  }

  return history;
}
