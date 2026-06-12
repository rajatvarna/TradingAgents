# GlobalScreener

A production-grade, auto-refreshing stock screener dashboard covering **US, India, UAE, and Saudi Arabia** markets.

## Features

- **130+ stocks** across 4 markets (S&P 500 leaders, Nifty 50, DFM/ADX, TASI)
- **7 performance metrics**: 1D, WTD, MTD, YTD, 1Y, 3Y, 5Y (cumulative)
- **TradingView integration**: live ticker tape, market index cards, inline sparklines, advanced chart panel
- **Auto-refresh** every 12 minutes with visual countdown
- **Redis caching** (Upstash free tier) — 10-minute TTL prevents API hammering
- **Filter & sort** by market, sector, timeframe, min % change; 5 preset screener views
- **CSV export** of visible filtered table
- **Dark mode** by default; responsive mobile card view

## Quick Start

```bash
cd global-screener
npm install
cp .env.example .env.local
# (optional) fill in UPSTASH_REDIS_REST_URL / TOKEN for caching
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `UPSTASH_REDIS_REST_URL` | Optional | Upstash Redis REST URL (free at upstash.com) |
| `UPSTASH_REDIS_REST_TOKEN` | Optional | Upstash Redis token |
| `TWELVE_DATA_API_KEY` | Optional | Fallback data source (twelvedata.com, 800/day free) |
| `NEXT_PUBLIC_REFRESH_INTERVAL_MS` | Optional | Polling interval in ms (default: 720000) |
| `NEXT_PUBLIC_VIRTUAL_ROW_THRESHOLD` | Optional | Row count before sparklines are suppressed |

## Deploy to Vercel

1. Push this directory to a GitHub repository
2. Import to [vercel.com/new](https://vercel.com/new)
3. Set the **Root Directory** to `global-screener`
4. Add environment variables in the Vercel dashboard
5. Deploy

## Extending the Watchlist

Edit `data/watchlist.json`. Each ticker needs:

```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "yahooSuffix": "AAPL",
  "tvSymbol": "NASDAQ:AAPL",
  "sector": "Technology"
}
```

Yahoo suffix guide: US = bare symbol, India NSE = `.NS`, UAE ADX = `.AD`, UAE DFM = `.DU`, Saudi = `.SR`.

## Data Sources

- **Primary**: Yahoo Finance (free, no key required, 15-min delayed)
- **Fallback**: Twelve Data (free tier, 800 calls/day)
- **Charts**: TradingView official embed widgets (free)

## Architecture

See the `lib/`, `components/`, and `app/api/` directories.

---

## Original Next.js Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
