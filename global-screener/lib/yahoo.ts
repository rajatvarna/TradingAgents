import { DailyClose } from "./performance";

const BASE = "https://query1.finance.yahoo.com";
const HEADERS = { "User-Agent": "Mozilla/5.0 (compatible; GlobalScreener/1.0)" };

async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Fetch with exponential backoff on 429 rate-limit responses. */
async function fetchWithBackoff(url: string, retries = 3): Promise<Response> {
  for (let attempt = 0; attempt < retries; attempt++) {
    const res = await fetch(url, { headers: HEADERS, cache: "no-store" });
    if (res.ok) return res;
    if (res.status === 429) {
      await sleep(Math.pow(2, attempt) * 1000 + Math.random() * 500);
      continue;
    }
    throw new Error(`HTTP ${res.status} for ${url}`);
  }
  throw new Error("Max retries exceeded");
}

export interface YahooQuote {
  symbol: string;
  regularMarketPrice: number;
  currency: string;
  marketCap: number;
  regularMarketVolume: number;
  averageDailyVolume3Month: number;
  longName: string;
  shortName: string;
  fiftyTwoWeekHigh: number;
  fiftyTwoWeekLow: number;
}

/** Fetches up to 10 live quotes in one request. */
export async function fetchBatchQuotes(symbols: string[]): Promise<YahooQuote[]> {
  const url = `${BASE}/v7/finance/quote?symbols=${symbols.join(",")}&fields=regularMarketPrice,currency,marketCap,regularMarketVolume,averageDailyVolume3Month,longName,shortName,fiftyTwoWeekHigh,fiftyTwoWeekLow`;
  const res = await fetchWithBackoff(url);
  const json = await res.json();
  return json?.quoteResponse?.result ?? [];
}

/** Fetches 5 years of daily adjusted closes for a single symbol. */
export async function fetchHistory(symbol: string): Promise<DailyClose[]> {
  const url = `${BASE}/v8/finance/chart/${symbol}?interval=1d&range=5y&includeAdjustedClose=true`;
  const res = await fetchWithBackoff(url);
  const json = await res.json();

  const chart = json?.chart?.result?.[0];
  if (!chart) return [];

  const timestamps: number[] = chart.timestamp ?? [];
  const adjClose: number[] = chart.indicators?.adjclose?.[0]?.adjclose ?? chart.indicators?.quote?.[0]?.close ?? [];

  return timestamps
    .map((ts, i) => ({
      date: new Date(ts * 1000).toISOString().split("T")[0],
      close: adjClose[i],
    }))
    .filter((d) => d.close != null && !isNaN(d.close));
}
