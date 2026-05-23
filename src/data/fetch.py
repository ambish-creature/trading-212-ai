import yfinance as yf
import pandas as pd
import os
import sys

# Add the root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME

def fetch_historical_data(ticker, output_path):
    """
    Fetches historical data using yfinance based on the ACTIVE_TIMEFRAME profile and saves it to a CSV.
    Trading 212 doesn't have an endpoint for rich historical candlesticks.
    """
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    period = profile['fetch_period']
    interval = profile['interval']
    
    print(f"Fetching data for {ticker} using timeframe '{ACTIVE_TIMEFRAME}' (period: {period}, interval: {interval})...")
    data = yf.download(ticker, period=period, interval=interval)
    
    # yfinance sometimes returns MultiIndex columns (Price, Ticker) even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    raw_path = os.path.join(os.path.dirname(__file__), '../../data/raw/AAPL.csv')
    fetch_historical_data("AAPL", raw_path)
