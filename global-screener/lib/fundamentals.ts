import { Fundamentals } from "@/types";

const BASE = "https://query1.finance.yahoo.com";
const HEADERS = { "User-Agent": "Mozilla/5.0 (compatible; GlobalScreener/1.0)" };

/**
 * Fetches key fundamental metrics from Yahoo Finance quoteSummary.
 * Uses modules: defaultKeyStatistics, financialData, summaryDetail, recommendationTrend.
 */
export async function fetchFundamentals(yahooSuffix: string): Promise<Fundamentals> {
  const modules = [
    "defaultKeyStatistics",
    "financialData",
    "summaryDetail",
    "recommendationTrend",
  ].join(",");

  const url = `${BASE}/v10/finance/quoteSummary/${yahooSuffix}?modules=${modules}`;
  const res = await fetch(url, { headers: HEADERS, cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const json = await res.json();

  const result = json?.quoteSummary?.result?.[0];
  if (!result) throw new Error("No data");

  const ks  = result.defaultKeyStatistics ?? {};
  const fd  = result.financialData ?? {};
  const sd  = result.summaryDetail ?? {};
  const rt  = result.recommendationTrend?.trend?.[0] ?? {};

  // Analyst consensus from recommendationTrend
  const strongBuy = rt.strongBuy ?? 0;
  const buy       = rt.buy ?? 0;
  const hold      = rt.hold ?? 0;
  const sell      = rt.sell ?? 0;
  const strongSell = rt.strongSell ?? 0;
  const total = strongBuy + buy + hold + sell + strongSell;
  let analystRating: string | null = null;
  if (total > 0) {
    const score = (strongBuy * 2 + buy * 1 + hold * 0 + sell * -1 + strongSell * -2) / total;
    analystRating = score > 0.5 ? "Strong Buy" : score > 0.1 ? "Buy" : score > -0.1 ? "Hold" : score > -0.5 ? "Sell" : "Strong Sell";
  }

  const v = (obj: Record<string, unknown>, key: string): number | null => {
    const raw = (obj as Record<string, { raw?: number }>)[key];
    return typeof raw?.raw === "number" ? raw.raw : null;
  };

  return {
    pe:                 v(sd,  "trailingPE"),
    forwardPe:          v(sd,  "forwardPE"),
    eps:                v(ks,  "trailingEps"),
    epsForward:         v(ks,  "forwardEps"),
    pbRatio:            v(ks,  "priceToBook"),
    psRatio:            v(ks,  "priceToSalesTrailing12Months"),
    evEbitda:           v(ks,  "enterpriseToEbitda"),
    debtToEquity:       v(fd,  "debtToEquity"),
    returnOnEquity:     v(fd,  "returnOnEquity"),
    returnOnAssets:     v(fd,  "returnOnAssets"),
    profitMargin:       v(fd,  "profitMargins"),
    revenueGrowthYoy:   v(fd,  "revenueGrowth"),
    earningsGrowthYoy:  v(fd,  "earningsGrowth"),
    dividendYield:      v(sd,  "dividendYield"),
    beta:               v(sd,  "beta"),
    analystRating,
    targetPrice:        v(fd,  "targetMeanPrice"),
    analystCount:       total > 0 ? total : null,
  };
}
