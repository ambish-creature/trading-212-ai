import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY_REAL = os.getenv("TRADING212_API_KEY_REAL")
API_SECRET_REAL = os.getenv("TRADING212_API_SECRET_REAL")
API_KEY_DEMO = os.getenv("TRADING212_API_KEY_DEMO")
API_SECRET_DEMO = os.getenv("TRADING212_API_SECRET_DEMO")

# Keep default fallback for backward compatibility
API_KEY = os.getenv("TRADING212_API_KEY") or API_KEY_DEMO
API_SECRET = os.getenv("TRADING212_API_SECRET") or API_SECRET_DEMO

BASE_URL_LIVE = "https://live.trading212.com/api/v0"
BASE_URL_DEMO = "https://demo.trading212.com/api/v0"
ENVIRONMENT = "DEMO"  # Switch to LIVE when ready

def get_base_url(real=False):
    return BASE_URL_LIVE if real else BASE_URL_DEMO


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
    # Default S&P 500 & ETFs
    "VOO":    "ETF",
    "SPY":    "ETF",
    "VWRL.L": "ETF",
    "IWY":    "ETF",
    "AIQ":    "ETF",
    
    # World's Top 20 Most Traded Stocks
    "AAPL":   "Tech",
    "MSFT":   "Tech",
    "NVDA":   "Tech",
    "AMZN":   "Tech",
    "GOOGL":  "Tech",
    "META":   "Tech",
    "TSLA":   "Tech",
    "BRK-B":  "Consumer",
    "LLY":    "Consumer",
    "JPM":    "Consumer",
    "AVGO":   "Tech",
    "TSM":    "Tech",
    "V":      "Consumer",
    "NVO":    "Consumer",
    "UNH":    "Consumer",
    "MA":     "Consumer",
    "ASML":   "Tech",
    "COST":   "Consumer",
    "NFLX":   "Tech",
    "AMD":    "Tech",
    
    # Cryptocurrencies
    "BTC-USD": "Crypto",
    
    # Commodities
    "CL=F":    "Commodity",
    "GC=F":    "Commodity",
    "SI=F":    "Commodity"
}
CATEGORIES = ["ETF", "Tech", "Consumer", "Crypto", "Commodity"]

TICKER_MAPPING = {
    "VOO":    "VOO_US_EQ",
    "SPY":    "SPY_US_EQ",
    "VWRL.L": "VWRL_LSE_EQ",
    "IWY":    "IWY_US_EQ",
    "AIQ":    "AIQ_US_EQ",
    "AAPL":   "AAPL_US_EQ",
    "MSFT":   "MSFT_US_EQ",
    "NVDA":   "NVDA_US_EQ",
    "AMZN":   "AMZN_US_EQ",
    "GOOGL":  "GOOGL_US_EQ",
    "META":   "META_US_EQ",
    "TSLA":   "TSLA_US_EQ",
    "BRK-B":  "BRKB_US_EQ",
    "LLY":    "LLY_US_EQ",
    "JPM":    "JPM_US_EQ",
    "AVGO":   "AVGO_US_EQ",
    "TSM":    "TSM_US_EQ",
    "V":      "V_US_EQ",
    "NVO":    "NVO_US_EQ",
    "UNH":    "UNH_US_EQ",
    "MA":     "MA_US_EQ",
    "ASML":   "ASML_US_EQ",
    "COST":   "COST_US_EQ",
    "NFLX":   "NFLX_US_EQ",
    "AMD":    "AMD_US_EQ",
    "BTC-USD": "BTC_USD",
    "CL=F":    "OIL_USD",
    "GC=F":    "GOLD_USD",
    "SI=F":    "SILVER_USD"
}
INVERSE_MAPPING = {v: k for k, v in TICKER_MAPPING.items()}

# ---------------------------------------------------------------------------
# Data & Timeframe Profiles
# ---------------------------------------------------------------------------

TIMEFRAME_PROFILES = {
    # Dynamic Short-Term daily horizons (1d to 28d)
    **{
        f"{d}d": {
            "interval": "1d",
            "fetch_period": "10y",
            "seq_length": 60,
            "target_shift": d
        } for d in range(1, 29)
    },
    "1mo": {
        "interval": "1d",
        "fetch_period": "10y",
        "seq_length": 60,
        "target_shift": 21
    },
    "2mo": {
        "interval": "1d",
        "fetch_period": "10y",
        "seq_length": 60,
        "target_shift": 42
    },
    "3mo": {
        "interval": "1d",
        "fetch_period": "10y",
        "seq_length": 60,
        "target_shift": 63
    },
    "6mo": {
        "interval": "1d",
        "fetch_period": "10y",
        "seq_length": 60,
        "target_shift": 126
    },
    "9mo": {
        "interval": "1d",
        "fetch_period": "10y",
        "seq_length": 60,
        "target_shift": 189
    },
    "1yr": {
        "interval": "1d",
        "fetch_period": "10y",
        "seq_length": 60,
        "target_shift": 252
    },
    "2yr": {
        "interval": "1d",
        "fetch_period": "10y",
        "seq_length": 60,
        "target_shift": 504
    }
}

# Change this to switch what the AI is predicting
ACTIVE_TIMEFRAME = "1mo"
