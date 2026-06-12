export type Market = "US" | "India" | "UAE" | "Saudi";

export interface TickerMeta {
  symbol: string;
  name: string;
  yahooSuffix: string;
  tvSymbol: string;
  sector: string;
}

export interface MarketUniverse {
  description: string;
  tickers: TickerMeta[];
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

export interface Fundamentals {
  pe: number | null;
  forwardPe: number | null;
  eps: number | null;
  epsForward: number | null;
  pbRatio: number | null;
  psRatio: number | null;
  evEbitda: number | null;
  debtToEquity: number | null;
  returnOnEquity: number | null;
  returnOnAssets: number | null;
  profitMargin: number | null;
  revenueGrowthYoy: number | null;
  earningsGrowthYoy: number | null;
  dividendYield: number | null;
  beta: number | null;
  analystRating: string | null;
  targetPrice: number | null;
  analystCount: number | null;
  earningsDate?: string | null;
}

export interface SentimentPost {
  title: string;
  url: string;
  source: "reddit" | "news";
  score: number;
  timestamp: string;
  subreddit?: string;
}

export interface SentimentData {
  score: number | null;        // -1 to +1
  label: "Bullish" | "Bearish" | "Neutral" | null;
  mentionCount: number;
  posts: SentimentPost[];
  fetchedAt: string;
}

export interface StockData {
  symbol: string;
  name: string;
  market: Market;
  sector: string;
  tvSymbol: string;
  price: number | null;
  currency: string;
  marketCap: number | null;
  volume: number | null;
  avgVolume20d: number | null;
  fiftyTwoWeekHigh: number | null;
  fiftyTwoWeekLow: number | null;
  fiftyTwoWeekHighChangePct: number | null;
  performance: PerformanceMetrics;
  fundamentals?: Fundamentals;
  sentiment?: SentimentData;
  isStale?: boolean;
  starred?: boolean;
  error?: string;
  rsScore?: number | null;
  earningsDate?: string | null;
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
  | "fiftyTwoWeekHighChangePct"
  | "rsScore";

export type SortDirection = "asc" | "desc";

export interface FilterState {
  markets: Market[];
  sectors: string[];
  sortField: SortField;
  sortDir: SortDirection;
  topN: number | null;
  minChangePct: number | null;
  search: string;
  onlyStarred: boolean;
  only52wHigh: boolean;
}

export type PresetName =
  | "top-gainers"
  | "top-losers"
  | "ytd-leaders"
  | "five-year-compounders"
  | "most-active"
  | "52w-highs"
  | "starred"
  | "volume-surge";
