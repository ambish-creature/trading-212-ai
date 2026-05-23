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
