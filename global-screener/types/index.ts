export type Market = "US" | "India" | "UAE" | "Saudi";

/** Raw ticker entry as stored in watchlist.json (no market field). */
export interface RawTickerEntry {
  symbol: string;
  name: string;
  yahooSuffix: string;
  tvSymbol: string;
  sector: string;
}

/** Ticker with its market resolved — returned by getTickers(). */
export interface TickerMeta extends RawTickerEntry {
  market: Market;
}

export interface MarketUniverse {
  description: string;
  tickers: RawTickerEntry[];
}

export interface Watchlist {
  US: MarketUniverse;
  India: MarketUniverse;
  UAE: MarketUniverse;
  Saudi: MarketUniverse;
}

export interface PerformanceMetrics {
  daily: number | null;
  wtd: number | null;
  mtd: number | null;
  ytd: number | null;
  one_y: number | null;
  three_y: number | null;
  five_y: number | null;
}

export interface StockData {
  symbol: string;
  yahooSuffix: string;
  name: string;
  market: Market;
  sector: string;
  tvSymbol: string;
  price: number | null;
  currency: string;
  marketCap: number | null;
  volume: number | null;
  avgVolume20d: number | null;
  fiftyTwoWeekLow: number | null;
  fiftyTwoWeekHigh: number | null;
  performance: PerformanceMetrics;
  dividendYield: number | null;
  exDividendDate: number | null;
  beta: number | null;
  isStale?: boolean;
  error?: string;
}

export interface BatchResponse {
  data: StockData[];
  cachedAt: string;
  nextRefresh: string;
}

export interface MarketSchedule {
  open: string;
  close: string;
  timezone: string;
  tradingDays: number[];
}

export type SortField =
  | "daily"
  | "wtd"
  | "mtd"
  | "ytd"
  | "one_y"
  | "three_y"
  | "five_y"
  | "price"
  | "marketCap"
  | "volume"
  | "rs"
  | "dividendYield"
  | "beta";

export type SortDirection = "asc" | "desc";

export interface FilterState {
  markets: Market[];
  sectors: string[];
  sortField: SortField;
  sortDir: SortDirection;
  topN: number | null;
  minChangePct: number | null;
  minMarketCap: number | null;
  maxMarketCap: number | null;
  search: string;
  volSurge: boolean;
  watchlistOnly: boolean;
  near52wHigh: boolean;
}

/** Types shared between ChartPanel and any future consumers */
export interface EarningsData {
  nextEarningsDate: string | null;
  epsEstimate: number | null;
}

export interface InsiderTransaction {
  name: string;
  relation: string;
  transactionDate: string;
  transactionType: string;
  shares: number;
  value: number | null;
}

export interface AnalystRatings {
  strongBuy: number;
  buy: number;
  hold: number;
  underperform: number;
  sell: number;
}

export interface FundamentalsData {
  shortPercentOfFloat: number | null;
  shortRatio: number | null;
  forwardPE: number | null;
  pegRatio: number | null;
  priceToBook: number | null;
  returnOnEquity: number | null;
  debtToEquity: number | null;
  analystRatings: AnalystRatings | null;
  beta: number | null;
  dividendYield: number | null;
  dividendRate: number | null;
  exDividendDate: number | null;
}

export interface NewsItem {
  uuid: string;
  title: string;
  publisher: string;
  link: string;
  providerPublishTime: number;
  thumbnail: string | null;
}

export type PresetName =
  | "top-gainers"
  | "top-losers"
  | "ytd-leaders"
  | "five-year-compounders"
  | "most-active"
  | "52w-highs"
  | "near-52w-high"
  | "vol-surge"
  | "watchlist";

export type MarketState = "open" | "pre-market" | "closed";
