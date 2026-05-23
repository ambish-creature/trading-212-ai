import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("TRADING212_API_KEY")
API_SECRET = os.getenv("TRADING212_API_SECRET")

# Example configuration values
BASE_URL_LIVE = "https://live.trading212.com/api/v0"
BASE_URL_DEMO = "https://demo.trading212.com/api/v0"
ENVIRONMENT = "DEMO"  # Switch to LIVE when ready

def get_base_url():
    return BASE_URL_DEMO if ENVIRONMENT == "DEMO" else BASE_URL_LIVE

# ---------------------------------------------------------------------------
# Portfolio Sizing (GBP)
# ---------------------------------------------------------------------------

# Total starting fund in GBP
STARTING_FUND_GBP = 5000.0

# Fraction of current portfolio value that must ALWAYS remain as free cash reserve.
# This protects against sudden crashes and keeps emergency buying power available.
# e.g. 0.20 = always keep at least 20% of portfolio value as cash.
CASH_RESERVE_RATIO = 0.20

# Maximum fraction of (available cash - reserve) that can be spent on a single trade.
# e.g. 0.15 = at most 15% of free-cash-above-reserve per trade order.
MAX_SINGLE_TRADE_RATIO = 0.15

# ---------------------------------------------------------------------------
# Historical UK Average Bank AER by Calendar Year (%)
# Used to compute the minimum benchmark return (must beat savings account).
# Sources: Bank of England base rate history + typical easy-access savings AER.
# ---------------------------------------------------------------------------
HISTORICAL_AER = {
    2015: 0.50,
    2016: 0.25,
    2017: 0.25,
    2018: 0.75,
    2019: 0.75,
    2020: 0.10,
    2021: 0.10,
    2022: 1.75,
    2023: 5.10,
    2024: 5.00,
    2025: 4.75,
    2026: 4.50,  # Estimated
}

# Target multiplier over the AER benchmark (1.25 = beat savings rate by 25%)
AER_TARGET_MULTIPLIER = 1.25

# ---------------------------------------------------------------------------
# Multi-Asset & Categories Configuration
# ---------------------------------------------------------------------------
ASSETS = {
    "SPY":    "ETF",
    "VWRL.L": "ETF",
    "IWY":    "ETF",
    "AIQ":    "ETF",
    "MSFT":   "Tech",
    "TSLA":   "Tech",
    "ASML":   "Tech",
    "META":   "Tech",
    "GOOGL":  "Tech",
    "MCD":    "Consumer",
    "COST":   "Consumer",
    "YUM":    "Consumer"
}
CATEGORIES = ["ETF", "Tech", "Consumer"]

TICKER_MAPPING = {
    "SPY":    "SPY_US_EQ",
    "VWRL.L": "VWRL_LSE_EQ",
    "IWY":    "IWY_US_EQ",
    "AIQ":    "AIQ_US_EQ",
    "MSFT":   "MSFT_US_EQ",
    "TSLA":   "TSLA_US_EQ",
    "ASML":   "ASML_US_EQ",
    "META":   "META_US_EQ",
    "GOOGL":  "GOOGL_US_EQ",
    "MCD":    "MCD_US_EQ",
    "COST":   "COST_US_EQ",
    "YUM":    "YUM_US_EQ"
}
INVERSE_MAPPING = {v: k for k, v in TICKER_MAPPING.items()}

# ---------------------------------------------------------------------------
# Data & Timeframe Profiles
# ---------------------------------------------------------------------------

TIMEFRAME_PROFILES = {
    "next_day": {
        "interval": "1d",
        "fetch_period": "10y",
        "seq_length": 60,   # 60 days of history
        "target_shift": 1   # predict 1 interval ahead
    },
    "next_week": {
        "interval": "1wk",
        "fetch_period": "max",
        "seq_length": 12,   # 12 weeks of history
        "target_shift": 1   # predict 1 interval ahead
    },
    "next_month": {
        "interval": "1mo",
        "fetch_period": "max",
        "seq_length": 12,   # 12 months of history
        "target_shift": 1   # predict 1 interval ahead
    },
    "next_year": {
        "interval": "1mo",
        "fetch_period": "max",
        "seq_length": 24,   # 24 months of history
        "target_shift": 12  # predict 12 intervals (months) ahead
    },
    "next_5_years": {
        "interval": "1mo",
        "fetch_period": "max",
        "seq_length": 60,   # 5 years of history
        "target_shift": 60  # predict 60 intervals (months) ahead
    }
}

# Change this to switch what the AI is predicting
ACTIVE_TIMEFRAME = "next_day"
