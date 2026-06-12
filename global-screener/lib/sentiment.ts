import { SentimentData, SentimentPost } from "@/types";

/**
 * Fetches sentiment signals for a stock ticker from:
 * 1. Reddit search via JSON API (no key needed, rate-limited)
 * 2. Yahoo Finance news feed as a fallback signal source
 *
 * Sentiment score is derived from upvote ratios and keyword analysis on titles.
 */
export async function fetchSentiment(
  symbol: string,
  companyName: string
): Promise<SentimentData> {
  const posts: SentimentPost[] = [];

  // --- Reddit public JSON search ---
  try {
    const query = encodeURIComponent(`${symbol} stock`);
    const subreddits = ["wallstreetbets", "stocks", "investing", "stockmarket"];
    const redditUrl = `https://www.reddit.com/search.json?q=${query}&sort=hot&limit=15&t=week`;

    const redditRes = await fetch(redditUrl, {
      headers: { "User-Agent": "GlobalScreener/1.0 (educational)" },
      cache: "no-store",
    });

    if (redditRes.ok) {
      const redditJson = await redditRes.json();
      const children = redditJson?.data?.children ?? [];

      for (const child of children.slice(0, 10)) {
        const post = child.data;
        const title: string = post.title ?? "";
        const sub: string = post.subreddit ?? "";

        // Only include relevant finance subreddits
        if (!subreddits.some((s) => sub.toLowerCase().includes(s.replace("r/", "")))) continue;

        posts.push({
          title,
          url: `https://reddit.com${post.permalink}`,
          source: "reddit",
          score: post.score ?? 0,
          timestamp: new Date((post.created_utc ?? 0) * 1000).toISOString(),
          subreddit: `r/${sub}`,
        });
      }
    }
  } catch {
    // Reddit unavailable — continue
  }

  // --- Yahoo Finance news RSS ---
  try {
    const newsUrl = `https://feeds.finance.yahoo.com/rss/2.0/headline?s=${symbol}&region=US&lang=en-US`;
    const newsRes = await fetch(newsUrl, {
      headers: { "User-Agent": "GlobalScreener/1.0" },
      cache: "no-store",
    });

    if (newsRes.ok) {
      const xml = await newsRes.text();
      // Simple regex extraction — no DOM parser available in Edge runtime
      const itemRegex = /<item>([\s\S]*?)<\/item>/g;
      const titleRegex = /<title><!\[CDATA\[(.*?)\]\]><\/title>|<title>(.*?)<\/title>/;
      const linkRegex = /<link>(.*?)<\/link>/;
      const pubDateRegex = /<pubDate>(.*?)<\/pubDate>/;

      let match: RegExpExecArray | null;
      let count = 0;
      while ((match = itemRegex.exec(xml)) !== null && count < 8) {
        const item = match[1];
        const titleMatch = titleRegex.exec(item);
        const linkMatch = linkRegex.exec(item);
        const dateMatch = pubDateRegex.exec(item);

        const title = titleMatch?.[1] ?? titleMatch?.[2] ?? "";
        const link = linkMatch?.[1] ?? "";
        const date = dateMatch?.[1] ?? "";

        if (title && link) {
          posts.push({
            title: title.trim(),
            url: link.trim(),
            source: "news",
            score: 0,
            timestamp: date ? new Date(date).toISOString() : new Date().toISOString(),
          });
          count++;
        }
      }
    }
  } catch {
    // News feed unavailable
  }

  // --- Sentiment scoring ---
  const bullishWords = ["surge", "soar", "rally", "beat", "record", "growth", "buy", "bullish", "strong", "upgrade", "outperform", "profit", "gains", "rise", "up", "positive", "exceed"];
  const bearishWords = ["crash", "drop", "fall", "miss", "loss", "bearish", "sell", "downgrade", "underperform", "decline", "weak", "disappoint", "cut", "down", "negative", "layoff", "debt"];

  let bullCount = 0;
  let bearCount = 0;

  for (const post of posts) {
    const text = post.title.toLowerCase();
    const b = bullishWords.filter((w) => text.includes(w)).length;
    const br = bearishWords.filter((w) => text.includes(w)).length;
    bullCount += b;
    bearCount += br;
  }

  const total = bullCount + bearCount;
  let score: number | null = null;
  let label: SentimentData["label"] = null;

  if (total > 0) {
    score = Math.round(((bullCount - bearCount) / total) * 100) / 100;
    label = score > 0.15 ? "Bullish" : score < -0.15 ? "Bearish" : "Neutral";
  } else if (posts.length > 0) {
    score = 0;
    label = "Neutral";
  }

  return {
    score,
    label,
    mentionCount: posts.length,
    posts: posts.slice(0, 10),
    fetchedAt: new Date().toISOString(),
  };
}
