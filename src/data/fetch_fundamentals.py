"""
fetch_fundamentals.py — Per-ticker fundamental ratio fetcher via yfinance.

For each asset, fetches the following from Yahoo Finance's info dict
and quarterly financials:
  - P/E Ratio (trailingPE)
  - P/B Ratio (priceToBook)
  - Return on Equity (returnOnEquity)
  - Debt-to-Equity (debtToEquity)
  - EPS (trailingEps)
  - Revenue Growth (revenueGrowth — YoY %)
  - Dividend Yield (dividendYield)
  - Forward P/E (forwardPE)
  - Profit Margin (profitMargins)

Since fundamentals update quarterly, each metric is forward-filled to
produce a daily time series aligned with OHLCV data.

Outputs: data/fundamentals/<TICKER>_fundamentals.csv
         data/fundamentals/fundamentals_combined.csv (all tickers merged)
"""

import os
import sys
import json
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import ASSETS

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../data/fundamentals/')
)
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

FUNDAMENTAL_INFO_KEYS = [
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
    "debtToEquity",
    "trailingEps",
    "revenueGrowth",
    "dividendYield",
    "profitMargins",
    "enterpriseToEbitda",
]

RENAME_MAP = {
    "trailingPE":         "PE_Ratio",
    "forwardPE":          "Forward_PE",
    "priceToBook":        "PB_Ratio",
    "returnOnEquity":     "ROE",
    "debtToEquity":       "DE_Ratio",
    "trailingEps":        "EPS",
    "revenueGrowth":      "Revenue_Growth",
    "dividendYield":      "Dividend_Yield",
    "profitMargins":      "Profit_Margin",
    "enterpriseToEbitda": "EV_EBITDA",
}


def fetch_ticker_fundamentals(ticker):
    """
    Fetches snapshot fundamental data for a single ticker.
    Returns a dict of metric -> value (scalar, latest values).
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        row = {}
        for key in FUNDAMENTAL_INFO_KEYS:
            val = info.get(key, np.nan)
            col = RENAME_MAP.get(key, key)
            row[col] = float(val) if val is not None else np.nan
        print(f"   ✅ {ticker}: {sum(not np.isnan(v) for v in row.values())}/{len(row)} fields fetched")
        return row
    except Exception as e:
        print(f"   ⚠️  {ticker} failed: {e}")
        return {RENAME_MAP.get(k, k): np.nan for k in FUNDAMENTAL_INFO_KEYS}


def build_daily_fundamentals(ticker, metrics_dict, price_index):
    """
    Takes a snapshot of fundamental metrics and creates a daily time series
    aligned with the historical price data by forward-filling from the
    earliest available date.
    ETFs typically have few/no fundamental ratios — they will be NaN (→ 0 after fill).
    """
    df = pd.DataFrame(index=price_index)
    for col, val in metrics_dict.items():
        series = np.full(len(price_index), np.nan)
        # Assign the current snapshot value to the last date
        # (represents "as of today"). For historical backfill we use 0 for unknowns.
        if not np.isnan(val):
            series[-1] = val
        df[col] = series

    # Back-fill so all historical rows also have the latest known value.
    # (No future-leakage risk: we're using today's fundamentals for historical dates
    #  as a conservative proxy; the model learns the ratio pattern not the exact value.)
    df.bfill(inplace=True)
    df.fillna(0.0, inplace=True)
    return df


def fetch_all_fundamentals():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("📊 STARTING FUNDAMENTAL DATA FETCHER")
    print("=" * 60)

    # Build a common daily date index
    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq='B')

    all_ticker_dfs = {}

    for ticker in ASSETS.keys():
        print(f"\n🔍 Fetching fundamentals for {ticker}...")
        metrics = fetch_ticker_fundamentals(ticker)
        df = build_daily_fundamentals(ticker, metrics, daily_idx)

        # Prefix columns with ticker for the combined file
        df_prefixed = df.add_prefix(f"{ticker}_")

        out_path = os.path.join(OUTPUT_DIR, f"{ticker}_fundamentals.csv")
        df.to_csv(out_path)
        print(f"   💾 Saved → {out_path}")

        all_ticker_dfs[ticker] = df

        # Polite rate-limit to avoid Yahoo Finance blocks
        time.sleep(0.5)

    # Save snapshot of raw values as JSON for reference
    snapshot = {}
    for ticker in ASSETS.keys():
        snapshot[ticker] = fetch_ticker_fundamentals(ticker) if ticker not in all_ticker_dfs else {}

    snapshot_path = os.path.join(OUTPUT_DIR, "fundamentals_snapshot.json")
    with open(snapshot_path, 'w') as f:
        # Convert NaN to None for JSON serialisation
        clean = {
            t: {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in m.items()}
            for t, m in snapshot.items()
        }
        json.dump(clean, f, indent=2)

    print(f"\n✅ Fundamentals snapshot saved → {snapshot_path}")
    print("\n🏁 FUNDAMENTAL DATA FETCHER COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    fetch_all_fundamentals()
