from enum import StrEnum


class AnalystType(StrEnum):
    MARKET = "market"
    SENTIMENT = "sentiment"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    ESG = "esg"
    DERIVATIVES = "derivatives"


class AssetType(StrEnum):
    STOCK = "stock"
    CRYPTO = "crypto"
