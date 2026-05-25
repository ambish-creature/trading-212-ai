"""
fetch.py — OHLCV + Dividends + GBP/FX rate fetcher via Yahoo Finance.
Fetches all 12 asset tickers plus GBP/USD daily FX rate.
"""

import yfinance as yf
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME, ASSETS


def fetch_historical_data(ticker, output_path):
    """
    Fetches historical OHLCV data + dividends using yfinance.
    Saves it to a CSV.
    """
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    period = profile['fetch_period']
    interval = profile['interval']

    print(f"\n📥 Fetching data for {ticker} (period: {period}, interval: {interval})...")

    data = yf.download(ticker, period=period, interval=interval, actions=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    print(f"   ✅ Saved → {output_path}  ({len(data)} rows)")


def fetch_gbpusd_fx(output_path):
    """
    Fetches the daily GBP/USD FX rate from Yahoo Finance (ticker: GBPUSD=X).
    Used to convert USD-denominated assets to GBP in the backtest engine.
    """
    print(f"\n💱 Fetching GBP/USD FX rate (10y daily)...")
    data = yf.download("GBPUSD=X", period="10y", interval="1d", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Keep only the Close price (that's the daily FX rate)
    data[['Close']].rename(columns={'Close': 'GBPUSD'}).to_csv(output_path)
    print(f"   ✅ Saved → {output_path}  ({len(data)} rows)")


def fetch_all_assets():
    """Loops through all configured assets and fetches their OHLCV data, plus the GBP/USD FX rate."""
    print("=" * 60)
    print("🌐 STARTING MULTI-ASSET HISTORICAL DATA SCRAPER")
    print("=" * 60)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Fetch each stock/ETF + support Sector Indices
    tickers = list(ASSETS.keys())
    support_indices = ["XLK", "XLY", "XLV", "XLF", "XLP"]
    for ticker in tickers + support_indices:
        output_path = os.path.join(root_dir, f'data/raw/{ticker}.csv')
        try:
            fetch_historical_data(ticker, output_path)
        except Exception as e:
            print(f"   ❌ Failed to fetch {ticker}: {e}")

    # Fetch GBP/USD FX rate
    fx_path = os.path.join(root_dir, 'data/raw/GBPUSD.csv')
    try:
        fetch_gbpusd_fx(fx_path)
    except Exception as e:
        print(f"   ❌ Failed to fetch GBPUSD FX rate: {e}")

    print("\n🏁 MULTI-ASSET HISTORICAL DATA SCRAPER COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    fetch_all_assets()
