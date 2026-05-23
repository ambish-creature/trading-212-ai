import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("TRADING212_API_KEY")
API_SECRET = os.getenv("TRADING212_API_SECRET")

# Example configuration values
BASE_URL_LIVE = "https://live.trading212.com/api/v0"
BASE_URL_DEMO = "https://demo.trading212.com/api/v0"
ENVIRONMENT = "DEMO" # Switch to LIVE when ready

def get_base_url():
    return BASE_URL_DEMO if ENVIRONMENT == "DEMO" else BASE_URL_LIVE

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
