"""
Scoring weights for the Monster Stock / TraderLion framework.

Each entry is (criterion_key, weight). Higher weight = more influence on composite.
Weights reflect the relative importance described in the Boik / TraderLion methodology.
"""

WEIGHTS: dict = {
    "eps_growth":               1.5,
    "eps_acceleration":         2.0,   # trend over 8Q is primary driver
    "revenue_growth":           1.3,
    "revenue_acceleration":     1.8,   # revenue-only stories need acceleration
    "annual_eps_trend":         1.0,
    "roe":                      0.8,
    "margin_trend":             0.8,
    "forward_estimate":         0.7,
    "fund_count_growth":        1.5,
    "fund_count_acceleration":  1.2,
    "flagship_fund":            1.0,
    "institutional_quality":    0.8,
    "ma_grade":                 2.0,   # non-negotiable: only own A/B stocks
    "volume_quality":           1.2,
    "base_pattern":             1.5,
    "breakout_quality":         1.2,
    "rs_score":                 1.8,   # RS rising before breakout is essential
    "sell_signal":              1.5,   # risk-side gate
    "extension_risk":           1.0,
    "group_rank":               1.5,   # group environment gates 50% of move
    "group_confirmation":       1.5,
    "market_health":            1.5,   # market phase gates everything
}
