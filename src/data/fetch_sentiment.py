"""
fetch_sentiment.py — Analyst ratings & news sentiment fetcher.

Data sources (all free, no API keys required):
  1. Yahoo Finance `.recommendations` — analyst Buy/Hold/Sell ratings
     → Aggregated into a monthly score (Buy=5, Outperform=4, Hold=3, Underperform=2, Sell=1)
  2. Yahoo Finance `.news` — recent news headlines for each ticker
     → Scored with VADER (Valence Aware Dictionary for sEntiment Reasoning),
       a lexicon-based sentiment analyser tuned for financial text.
  3. Yahoo Finance `.institutional_holders` — top institutional ownership %

Outputs:
  data/sentiment/<TICKER>_sentiment.csv  (daily)
  data/sentiment/sentiment_combined.csv  (all tickers, wide format)

IMPORTANT: VADER scores are in [-1.0, +1.0].
  +1.0 = extremely positive / bullish
   0.0 = neutral
  -1.0 = extremely negative / bearish
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import ASSETS

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../data/sentiment/')
)
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def get_vader_analyzer():
    """Lazy-load VADER. Falls back gracefully if vaderSentiment is not installed."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        try:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except Exception:
            return None


def score_recommendations(ticker_obj):
    """
    Converts analyst recommendation history to a numeric score (1–5 scale).
    Buy/StrongBuy = 5, Outperform = 4, Hold/Neutral = 3,
    Underperform = 2, Sell/StrongSell = 1.
    Returns a DataFrame indexed by date with column 'Analyst_Score'.
    """
    BUY_WORDS     = {'buy', 'strong buy', 'strongbuy', 'outperform', 'overweight', 'accumulate', 'add'}
    HOLD_WORDS    = {'hold', 'neutral', 'market perform', 'equal weight', 'equalweight', 'sector perform', 'in-line', 'inline'}
    SELL_WORDS    = {'sell', 'strong sell', 'strongsell', 'underperform', 'underweight', 'reduce'}

    try:
        recs = ticker_obj.recommendations
        if recs is None or len(recs) == 0:
            return None

        scores = []
        for idx, row in recs.iterrows():
            action = str(row.get('To Grade', row.get('Action', ''))).lower().strip()
            if any(w in action for w in BUY_WORDS):
                score = 5.0
            elif any(w in action for w in SELL_WORDS):
                score = 1.0
            else:
                score = 3.0  # Default to Hold

            # Date might be timezone-aware; normalize to date-only
            date = pd.Timestamp(idx).normalize().tz_localize(None) if hasattr(idx, 'normalize') else pd.Timestamp(idx)
            scores.append({'date': date, 'Analyst_Score': score})

        if not scores:
            return None

        df = pd.DataFrame(scores).groupby('date')['Analyst_Score'].mean().to_frame()
        return df
    except Exception:
        return None


def score_news_sentiment(ticker_obj, analyzer):
    """
    Scores recent news headlines using VADER.
    Returns a single compound score (mean of recent headlines).
    """
    if analyzer is None:
        return 0.0
    try:
        news = ticker_obj.news
        if not news:
            return 0.0
        scores = []
        for article in news[:20]:  # Use latest 20 headlines
            title = article.get('title', '')
            if title:
                score = analyzer.polarity_scores(title)['compound']
                scores.append(score)
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def get_institutional_pct(ticker_obj):
    """Returns total institutional ownership as a fraction (0.0 to 1.0)."""
    try:
        holders = ticker_obj.institutional_holders
        if holders is None or len(holders) == 0:
            return np.nan
        # '% Out' column is fractional ownership per institution
        pct_col = [c for c in holders.columns if 'out' in c.lower() or '%' in c.lower()]
        if pct_col:
            total = holders[pct_col[0]].sum()
            return min(float(total), 1.0)  # Cap at 100%
        return np.nan
    except Exception:
        return np.nan


def build_daily_sentiment(analyst_df, news_score, inst_pct, daily_idx):
    """
    Merges all sentiment signals into a single daily DataFrame.
    Analyst scores are forward-filled (last known rating persists until updated).
    News sentiment and institutional % are static snapshots (latest known).
    """
    df = pd.DataFrame(index=daily_idx)
    df.index.name = 'Date'

    # Analyst Score
    if analyst_df is not None:
        analyst_aligned = analyst_df.reindex(daily_idx, method='ffill')
        df['Analyst_Score'] = analyst_aligned['Analyst_Score']
    else:
        df['Analyst_Score'] = 3.0  # Default neutral

    df['Analyst_Score'].ffill(inplace=True)
    df['Analyst_Score'].fillna(3.0, inplace=True)

    # News Sentiment (snapshot value, forward-filled from today backward)
    df['News_Sentiment'] = 0.0
    if not np.isnan(news_score):
        df['News_Sentiment'] = news_score

    # Institutional Ownership
    df['Institutional_Pct'] = 0.0
    if not np.isnan(inst_pct):
        df['Institutional_Pct'] = inst_pct

    return df


def fetch_all_sentiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("🧠 STARTING SENTIMENT DATA FETCHER")
    print("=" * 60)

    analyzer = get_vader_analyzer()
    if analyzer:
        print("   ✅ VADER sentiment analyser loaded.")
    else:
        print("   ⚠️  VADER not available. Install with: pip install vaderSentiment")
        print("        News sentiment will default to 0.0 (neutral).")

    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    all_dfs = {}

    for ticker in ASSETS.keys():
        print(f"\n🔍 Fetching sentiment for {ticker}...")
        try:
            tk = yf.Ticker(ticker)

            analyst_df  = score_recommendations(tk)
            news_score  = score_news_sentiment(tk, analyzer)
            inst_pct    = get_institutional_pct(tk)

            n_analyst = len(analyst_df) if analyst_df is not None else 0
            print(f"   📰 News sentiment:  {news_score:+.3f}")
            print(f"   📋 Analyst records: {n_analyst}")
            print(f"   🏦 Institutional %: {inst_pct:.1%}" if not np.isnan(inst_pct) else "   🏦 Institutional %: N/A")

            df = build_daily_sentiment(analyst_df, news_score, inst_pct, daily_idx)
            all_dfs[ticker] = df

            out_path = os.path.join(OUTPUT_DIR, f"{ticker}_sentiment.csv")
            df.to_csv(out_path)
            print(f"   💾 Saved → {out_path}")

        except Exception as e:
            print(f"   ❌ {ticker} failed: {e}")
            # Provide neutral fallback
            df = pd.DataFrame(
                {'Analyst_Score': 3.0, 'News_Sentiment': 0.0, 'Institutional_Pct': 0.0},
                index=daily_idx
            )
            all_dfs[ticker] = df

        time.sleep(0.5)  # Rate-limit Yahoo Finance

    # Save combined sentiment file (one row per date, columns prefixed by ticker)
    combined_parts = []
    for ticker, df in all_dfs.items():
        prefixed = df.add_prefix(f"{ticker}_")
        combined_parts.append(prefixed)

    if combined_parts:
        combined = pd.concat(combined_parts, axis=1)
        combined.ffill(inplace=True)
        combined_path = os.path.join(OUTPUT_DIR, "sentiment_combined.csv")
        combined.to_csv(combined_path)
        print(f"\n✅ Combined sentiment saved → {combined_path}")
        print(f"   Shape: {combined.shape}")

    print("\n🏁 SENTIMENT DATA FETCHER COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    fetch_all_sentiment()
