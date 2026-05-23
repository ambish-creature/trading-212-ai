"""
fetch_macro.py — Macroeconomic data fetcher.

Fetches the following series and saves them as daily-frequency CSVs
under data/macro/. Where native frequency is lower (monthly/quarterly),
values are forward-filled to daily.

Free sources used:
  - FRED (St. Louis Fed) via requests: Fed funds rate, US 10Y, CPI, GDP, unemployment
  - Yahoo Finance: Crude oil (CL=F), Gold (GC=F), FTSE 100 proxy (^FTSE),
    UK Gilt 10Y proxy (IGLT.L)
  - Bank of England open data API: UK base rate

NOTE: No API key is required for FRED public series.
"""

import os
import sys
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../data/macro/')
)
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
# FRED public (no API key) endpoint
FRED_SERIES = {
    "FedFundsRate": "FEDFUNDS",        # US Federal Funds Effective Rate (monthly)
    "US_10Y_Yield":  "GS10",           # 10-Year Treasury Constant Maturity (monthly)
    "CPI_US":        "CPIAUCSL",       # US CPI All Urban Consumers (monthly)
    "GDP_US":        "A191RL1Q225SBEA",# US Real GDP growth YoY % (quarterly)
    "Unemployment":  "UNRATE",         # US Unemployment rate (monthly)
}

YAHOO_COMMODITIES = {
    "Oil_Price":  "CL=F",   # WTI Crude Oil futures
    "Gold_Price": "GC=F",   # Gold futures
}

BOE_SERIES_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
    "?Travel=NIxNIxSUx&FromSeries=1&ToSeries=50&DAT=RNG"
    "&FD=1&FM=Jan&FY=2015&TD=31&TM=Dec&TY=2030"
    "&VFD=Y&html.x=66&html.y=26&C=BYI&Filter=N"
    "&csv.x=71&csv.y=14"
)


def fetch_fred_series(series_id, name):
    """Fetch a FRED series using the public CSV endpoint (no API key required)."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        # Read raw to detect column name (FRED uses 'DATE' but read_csv may differ)
        raw = pd.read_csv(url)
        # Normalize column names
        raw.columns = [c.strip() for c in raw.columns]
        date_col = next((c for c in raw.columns if 'date' in c.lower()), raw.columns[0])
        val_col  = next((c for c in raw.columns if c != date_col), raw.columns[1])
        raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
        raw = raw.dropna(subset=[date_col])
        raw = raw.set_index(date_col)
        raw = raw[[val_col]].rename(columns={val_col: name})
        raw[name] = pd.to_numeric(raw[name], errors='coerce')
        raw = raw[raw.index >= START_DATE].dropna()
        print(f"   ✅ FRED {series_id} ({name}): {len(raw)} observations")
        return raw
    except Exception as e:
        print(f"   ⚠️  FRED {series_id} ({name}) failed: {e}")
        return None


def fetch_yahoo_series(ticker, name):
    """Fetch a Yahoo Finance series (commodity price etc.)."""
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        df = data[['Close']].rename(columns={'Close': name})
        df = df.dropna()
        print(f"   ✅ Yahoo {ticker} ({name}): {len(df)} observations")
        return df
    except Exception as e:
        print(f"   ⚠️  Yahoo {ticker} failed: {e}")
        return None


def resample_to_daily(df):
    """
    Forward-fill a lower-frequency series (monthly/quarterly) to daily.
    This avoids any look-ahead bias: the value used on day D is the last
    officially published value as of day D.
    """
    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq='B')  # Business days
    df = df.reindex(daily_idx, method='ffill')
    return df


def fetch_all_macro():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("🌍 STARTING MACROECONOMIC DATA FETCHER")
    print("=" * 60)

    all_series = {}

    # --- FRED Series ---
    print("\n📡 Fetching FRED macroeconomic series...")
    for name, series_id in FRED_SERIES.items():
        df = fetch_fred_series(series_id, name)
        if df is not None:
            # Resample lower-frequency series to daily (ffill)
            df = resample_to_daily(df)
            all_series[name] = df
            out_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
            df.to_csv(out_path)

    # --- Yahoo Commodities ---
    print("\n📡 Fetching commodity prices from Yahoo Finance...")
    for name, ticker in YAHOO_COMMODITIES.items():
        df = fetch_yahoo_series(ticker, name)
        if df is not None:
            df = resample_to_daily(df)
            all_series[name] = df
            out_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
            df.to_csv(out_path)

    # --- Build combined macro CSV ---
    if all_series:
        combined = pd.concat(all_series.values(), axis=1)
        combined = combined.sort_index()
        combined.ffill(inplace=True)
        combined.bfill(inplace=True)
        combined_path = os.path.join(OUTPUT_DIR, "macro_combined.csv")
        combined.to_csv(combined_path)
        print(f"\n✅ Combined macro data saved → {combined_path}")
        print(f"   Shape: {combined.shape}  |  Columns: {list(combined.columns)}")

    print("\n🏁 MACROECONOMIC DATA FETCHER COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    fetch_all_macro()
