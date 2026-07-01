from enum import Enum


class AnalystType(str, Enum):
    MARKET = "market"
    SOCIAL = "social"
    SENTIMENT = "sentiment"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    ESG = "esg"
    DERIVATIVES = "derivatives"


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
