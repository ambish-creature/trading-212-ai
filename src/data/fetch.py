import yfinance as yf
import pandas as pd
import os
import sys

# Add the root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME, ASSETS

def fetch_historical_data(ticker, output_path):
    """
    Fetches historical data using yfinance based on the ACTIVE_TIMEFRAME profile.
    Saves it to a CSV including dividends (actions=True).
    """
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    period = profile['fetch_period']
    interval = profile['interval']
    
    print(f"\n📥 Fetching data for {ticker} using timeframe '{ACTIVE_TIMEFRAME}' (period: {period}, interval: {interval})...")
    
    # actions=True fetches dividend payments and stock split details
    data = yf.download(ticker, period=period, interval=interval, actions=True)
    
    # yfinance sometimes returns MultiIndex columns (Price, Ticker) even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    print(f"✅ Data for {ticker} saved to {output_path} (Rows: {len(data)})")

def fetch_all_assets():
    """Loops through all configured assets and fetches their data."""
    print("=" * 60)
    print("🌐 STARTING MULTI-ASSET HISTORICAL DATA SCRAPER")
    print("=" * 60)
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    
    for ticker in ASSETS.keys():
        output_path = os.path.join(root_dir, f'data/raw/{ticker}.csv')
        try:
            fetch_historical_data(ticker, output_path)
        except Exception as e:
            print(f"❌ Failed to fetch data for {ticker}: {e}")
            
    print("\n🏁 MULTI-ASSET HISTORICAL DATA SCRAPER COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    fetch_all_assets()
