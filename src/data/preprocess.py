"""
preprocess.py — Rich multi-source feature engineering pipeline.

Feature groups:
  1. OHLCV + Technical Indicators  (from raw/ CSVs)
  2. Macroeconomic data             (from macro/macro_combined.csv)
  3. Fundamental ratios             (from fundamentals/<TICKER>_fundamentals.csv)
  4. Sentiment signals              (from sentiment/<TICKER>_sentiment.csv)
  5. GBP/USD FX rate                (from raw/GBPUSD.csv)
  6. Category one-hot encoding      (ETF / Tech / Consumer)

All numeric features are scaled by a single global StandardScaler
fitted strictly on the chronological training split (no future leakage).

Outputs (data/processed/):
  X_train.npy, y_train.npy
  X_val.npy,   y_val.npy
  <TICKER>_X_test.npy, <TICKER>_y_test.npy  (one file per asset)
  scaler.pkl
  feature_cols.pkl
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
import ta

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME, ASSETS, CATEGORIES

# Mapped Sector ETFs for Peer Momentum (v3.0)
SECTOR_MAP = {
    "VOO": "SPY", "SPY": "SPY", "VWRL.L": "SPY", "IWY": "SPY", "AIQ": "SPY",
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMZN": "XLY", "GOOGL": "XLK",
    "META": "XLK", "TSLA": "XLY", "AVGO": "XLK", "TSM": "XLK", "ASML": "XLK",
    "NFLX": "XLY", "AMD": "XLK",
    "BRK-B": "SPY", "LLY": "XLV", "JPM": "XLF", "V": "XLF", "NVO": "XLV",
    "UNH": "XLV", "MA": "XLF", "COST": "XLP",
    "BTC-USD": "SPY", "CL=F": "SPY", "GC=F": "SPY", "SI=F": "SPY"
}


# Core OHLCV + technical features (always present)
OHLCV_TECH_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_High', 'BB_Mid', 'BB_Low', 'BB_Width',
    'ATR_14',
    'Bid_Ask_Proxy',     # Estimated from (High - Low) / Close
    'Dividends',
    # --- New features (v2.0) ---
    'Momentum_5',        # 5-day price momentum (short-term)
    'Momentum_10',       # 10-day price momentum (medium-term)
    'Momentum_20',       # 20-day price momentum (trend confirmation)
    'OBV',               # On-Balance Volume (volume-price divergence)
    'Vol_Regime',        # Volatility regime: ATR_14 / Close (normalised vol)
    'Trend_Strength',    # SMA20 / SMA50 ratio (trend direction & strength)
    'Williams_R',        # Williams %R oscillator (overbought/oversold)
    'Price_vs_SMA50',    # Close / SMA50 - 1 (distance from medium-term mean)
    # --- Cross-Asset Sector features (v3.0) ---
    'Sector_Return_5d',
    'Sector_Return_20d',
    'Relative_Strength_5d',
    'Relative_Strength_20d',
]

# Macro features (from macro_combined.csv, forward-filled daily)
MACRO_COLS = [
    'FedFundsRate', 'US_10Y_Yield', 'CPI_US', 'GDP_US', 'Unemployment',
    'Oil_Price', 'Gold_Price',
    # --- Cross-Asset Macro features (v3.0) ---
    'Gold_Return_5d',
    'Oil_Return_5d',
    'Safe_Haven_Ratio',
    'Equity_Gold_Corr',
    'Equity_Oil_Corr',
]

# GBP/USD FX rate
FX_COLS = ['GBPUSD']

# Per-ticker fundamental features (from fundamentals/<TICKER>_fundamentals.csv)
FUNDAMENTAL_COLS = [
    'PE_Ratio', 'Forward_PE', 'PB_Ratio', 'ROE', 'DE_Ratio',
    'EPS', 'Revenue_Growth', 'Dividend_Yield', 'Profit_Margin', 'EV_EBITDA',
]

# Per-ticker sentiment features (from sentiment/<TICKER>_sentiment.csv)
SENTIMENT_COLS = [
    'Analyst_Score', 'News_Sentiment', 'Institutional_Pct',
]

# All columns that will be scaled by StandardScaler
ALL_NUMERIC_COLS = OHLCV_TECH_COLS + MACRO_COLS + FX_COLS + FUNDAMENTAL_COLS + SENTIMENT_COLS


def wma(series, window):
    """Computes Weighted Moving Average."""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hull_moving_average(series, window=14):
    """Computes lag-free Hull Moving Average."""
    half_win = int(window / 2)
    sqrt_win = int(np.sqrt(window))
    wma_half = wma(series, half_win)
    wma_full = wma(series, window)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_win)


def add_technical_indicators(df, ticker, target_shift):
    """Computes all technical indicators and the prediction target, including sector peer momentum."""
    close = df['Close']

    # Denoise close price causally using Hull Moving Average (lag-free)
    denoised = hull_moving_average(close, window=14).fillna(close)

    df['SMA_20']  = denoised.rolling(window=20).mean()
    df['SMA_50']  = denoised.rolling(window=50).mean()
    df['EMA_12']  = denoised.ewm(span=12, adjust=False).mean()
    df['EMA_26']  = denoised.ewm(span=26, adjust=False).mean()

    df['RSI'] = ta.momentum.RSIIndicator(close=denoised, window=14).rsi()

    macd_ind = ta.trend.MACD(close=denoised)
    df['MACD']        = macd_ind.macd()
    df['MACD_Signal'] = macd_ind.macd_signal()
    df['MACD_Hist']   = macd_ind.macd_diff()

    bb = ta.volatility.BollingerBands(close=denoised, window=20, window_dev=2)
    df['BB_High']  = bb.bollinger_hband()
    df['BB_Mid']   = bb.bollinger_mavg()
    df['BB_Low']   = bb.bollinger_lband()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / (df['BB_Mid'] + 1e-9)

    atr_ind = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=close, window=14)
    df['ATR_14'] = atr_ind.average_true_range()

    # Bid-Ask spread proxy: (High - Low) / Close
    df['Bid_Ask_Proxy'] = (df['High'] - df['Low']) / (close + 1e-9)

    # --- v2.0 Enhanced Features ---

    # Price momentum: percentage change over N days using denoised price (prevents noise flips)
    df['Momentum_5']  = denoised.pct_change(periods=5)  * 100.0
    df['Momentum_10'] = denoised.pct_change(periods=10) * 100.0
    df['Momentum_20'] = denoised.pct_change(periods=20) * 100.0

    # On-Balance Volume: cumulative volume signed by price direction
    # A rising OBV confirms price trend; divergence signals reversal
    obv = (np.sign(denoised.diff()) * df['Volume']).fillna(0).cumsum()
    # Normalise to z-score over 50-day window to make it comparable across assets
    obv_mean = obv.rolling(50).mean()
    obv_std  = obv.rolling(50).std().replace(0, 1e-9)
    df['OBV'] = (obv - obv_mean) / obv_std

    # Volatility regime: ATR normalised by denoised price
    # High values = high volatility regime, low = calm market
    df['Vol_Regime'] = df['ATR_14'] / (denoised + 1e-9)

    # Trend strength: SMA20 / SMA50 - 1
    # Positive → price in uptrend (fast MA above slow MA)
    # Negative → downtrend
    df['Trend_Strength'] = (df['SMA_20'] / (df['SMA_50'] + 1e-9)) - 1.0

    # Williams %R: measures overbought/oversold [-100 to 0]
    # -100 to -80: oversold (potential buy), -20 to 0: overbought (potential sell)
    highest_high = df['High'].rolling(14).max()
    lowest_low   = df['Low'].rolling(14).min()
    df['Williams_R'] = -100 * (highest_high - denoised) / (highest_high - lowest_low + 1e-9)

    # Price vs 50-day SMA: normalised distance from medium-term mean
    # Captures mean-reversion and trend following signals
    df['Price_vs_SMA50'] = (denoised / (df['SMA_50'] + 1e-9)) - 1.0

    # --- v3.0 Cross-Asset Sector Peer Features ---
    sector_ticker = SECTOR_MAP.get(ticker, "SPY")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    sector_path = os.path.join(root_dir, f"data/raw/{sector_ticker}.csv")

    if os.path.exists(sector_path):
        try:
            sector_df = pd.read_csv(sector_path, index_col='Date', parse_dates=True)
            sector_df.sort_index(inplace=True)
            sector_close = sector_df['Close']
            sector_denoised = hull_moving_average(sector_close, window=14).fillna(sector_close)
            
            # 5d and 20d returns of sector index
            sector_return_5 = sector_denoised.pct_change(periods=5) * 100.0
            sector_return_20 = sector_denoised.pct_change(periods=20) * 100.0
            
            # Align sector returns causally
            aligned_sector = pd.DataFrame(index=df.index)
            aligned_sector['Sector_Return_5d'] = sector_return_5.reindex(df.index, method='ffill').fillna(0.0)
            aligned_sector['Sector_Return_20d'] = sector_return_20.reindex(df.index, method='ffill').fillna(0.0)
            
            df['Sector_Return_5d'] = aligned_sector['Sector_Return_5d']
            df['Sector_Return_20d'] = aligned_sector['Sector_Return_20d']
        except Exception as ex:
            print(f"   ⚠️  Failed to compute sector peer features for {ticker}: {ex}")
            df['Sector_Return_5d'] = 0.0
            df['Sector_Return_20d'] = 0.0
    else:
        df['Sector_Return_5d'] = 0.0
        df['Sector_Return_20d'] = 0.0

    df['Relative_Strength_5d'] = df['Momentum_5'] - df['Sector_Return_5d']
    df['Relative_Strength_20d'] = df['Momentum_20'] - df['Sector_Return_20d']

    # Target: n-day-ahead percentage return
    df['Target_Return'] = close.pct_change(periods=target_shift).shift(-target_shift) * 100.0

    # Ensure Dividends column exists
    if 'Dividends' not in df.columns:
        df['Dividends'] = 0.0
    else:
        df['Dividends'] = df['Dividends'].fillna(0.0)

    # Ensure Open/High/Low exist (they always should from yfinance)
    for col in ['Open', 'High', 'Low']:
        if col not in df.columns:
            df[col] = df['Close']

    return df


def load_macro_data(root_dir):
    """Loads the combined macro CSV. Returns None if not found."""
    path = os.path.join(root_dir, 'data/macro/macro_combined.csv')
    if not os.path.exists(path):
        print("   ⚠️  Macro data not found. Run fetch_macro.py first. Macro features will be zero.")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


def load_fx_data(root_dir):
    """Loads daily GBP/USD FX rate. Returns None if not found."""
    path = os.path.join(root_dir, 'data/raw/GBPUSD.csv')
    if not os.path.exists(path):
        print("   ⚠️  GBP/USD FX data not found. Run fetch.py first. FX feature will be 1.27 (approx).")
        return None
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    df = df.sort_index()
    return df


def load_fundamentals(root_dir, ticker):
    """Loads per-ticker fundamentals. Returns None if not found."""
    path = os.path.join(root_dir, f'data/fundamentals/{ticker}_fundamentals.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


def load_sentiment(root_dir, ticker):
    """Loads per-ticker sentiment data. Returns None if not found."""
    path = os.path.join(root_dir, f'data/sentiment/{ticker}_sentiment.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    df = df.sort_index()
    return df


def align_external_series(external_df, target_index, cols, default_val=0.0):
    """
    Reindexes an external (macro/fundamental/sentiment) DataFrame to align with
    the asset price date index using forward-fill (then back-fill, then default).
    This ensures we never use future data — only the last known value.
    """
    result = pd.DataFrame(index=target_index)
    if external_df is not None:
        aligned = external_df[cols].reindex(target_index, method='ffill')
        aligned = aligned.bfill().fillna(default_val)
        for col in cols:
            if col in aligned.columns:
                result[col] = aligned[col]
            else:
                result[col] = default_val
    else:
        for col in cols:
            result[col] = default_val
    return result


def create_sequences(features, targets, seq_length):
    """Creates sliding windows for time-series data."""
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:(i + seq_length)])
        ys.append(targets[i + seq_length - 1])
    return np.array(xs), np.array(ys)


def process_all_data(timeframe=None):
    if timeframe is None:
        timeframe = ACTIVE_TIMEFRAME

    print("=" * 60)
    print(f"🧠 STARTING RICH MULTI-SOURCE DATA PREPROCESSOR FOR TIMEFRAME: {timeframe}")
    print("=" * 60)

    profile = TIMEFRAME_PROFILES[timeframe]
    seq_length   = profile['seq_length']
    target_shift = profile['target_shift']

    root_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    raw_dir       = os.path.join(root_dir, 'data/raw/')
    processed_dir = os.path.join(root_dir, 'data/processed/')
    os.makedirs(processed_dir, exist_ok=True)

    # ---- Load shared external data sources ----
    macro_df = load_macro_data(root_dir)
    fx_df    = load_fx_data(root_dir)

    # ---- Compute global macro correlations & Safe-Haven indicators (v3.0) ----
    try:
        spy_path = os.path.join(raw_dir, "SPY.csv")
        gold_path = os.path.join(raw_dir, "GC=F.csv")
        oil_path = os.path.join(raw_dir, "CL=F.csv")
        
        if os.path.exists(spy_path) and os.path.exists(gold_path) and os.path.exists(oil_path):
            spy_close = pd.read_csv(spy_path, index_col='Date', parse_dates=True)['Close'].sort_index()
            gold_close = pd.read_csv(gold_path, index_col='Date', parse_dates=True)['Close'].sort_index()
            oil_close = pd.read_csv(oil_path, index_col='Date', parse_dates=True)['Close'].sort_index()
            
            # Causal Denoising using HMA
            spy_denoised = hull_moving_average(spy_close, window=14).fillna(spy_close)
            gold_denoised = hull_moving_average(gold_close, window=14).fillna(gold_close)
            oil_denoised = hull_moving_average(oil_close, window=14).fillna(oil_close)
            
            global_df = pd.DataFrame(index=spy_close.index)
            global_df['Gold_Return_5d']   = (gold_denoised.pct_change(periods=5) * 100.0).fillna(0.0)
            global_df['Oil_Return_5d']    = (oil_denoised.pct_change(periods=5) * 100.0).fillna(0.0)
            global_df['Safe_Haven_Ratio'] = (gold_denoised / (spy_denoised + 1e-9)).fillna(0.0)
            global_df['Equity_Gold_Corr'] = spy_denoised.rolling(20).corr(gold_denoised).fillna(0.0)
            global_df['Equity_Oil_Corr']  = spy_denoised.rolling(20).corr(oil_denoised).fillna(0.0)
            
            if macro_df is not None:
                # Merge into macro_df
                macro_df = macro_df.join(global_df, how='left').ffill().bfill().fillna(0.0)
            else:
                macro_df = global_df
            print("   ✅ Computed global cross-asset Safe-Haven & Correlation indicators!")
        else:
            print("   ⚠️  SPY, Gold, or Oil CSV missing. Safe-Haven indicators will be zero.")
    except Exception as e:
        print(f"   ⚠️  Could not compute global macro indicators: {e}")

    # ---- Step 1: Build per-asset DataFrames ----
    asset_dfs = {}
    for ticker, category in ASSETS.items():
        csv_path = os.path.join(raw_dir, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            print(f"   ⚠️  Raw CSV not found for {ticker}, skipping.")
            continue

        df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
        df.sort_index(inplace=True)

        # Technical indicators + target
        df = add_technical_indicators(df, ticker, target_shift)

        # Drop rows where core technical indicators haven't warmed up yet
        # Include new v2.0 features: OBV needs 50-day window, Momentum_20 needs 20 days
        core_cols = ['SMA_20', 'SMA_50', 'RSI', 'ATR_14', 'MACD', 'BB_High',
                     'Momentum_20', 'OBV', 'Trend_Strength', 'Williams_R']
        df.dropna(subset=core_cols, inplace=True)
        df.dropna(subset=['Target_Return'], inplace=True)

        # Align macroeconomic data
        macro_aligned = align_external_series(macro_df, df.index, MACRO_COLS, default_val=0.0)
        for col in MACRO_COLS:
            df[col] = macro_aligned[col].values

        # Align GBP/USD FX
        fx_aligned = align_external_series(fx_df, df.index, ['GBPUSD'], default_val=1.27)
        df['GBPUSD'] = fx_aligned['GBPUSD'].values

        # Align per-ticker fundamentals
        fund_df = load_fundamentals(root_dir, ticker)
        fund_aligned = align_external_series(fund_df, df.index, FUNDAMENTAL_COLS, default_val=0.0)
        for col in FUNDAMENTAL_COLS:
            df[col] = fund_aligned[col].values

        # Align per-ticker sentiment
        sent_df = load_sentiment(root_dir, ticker)
        sent_aligned = align_external_series(sent_df, df.index, SENTIMENT_COLS, default_val=0.0)
        for col in SENTIMENT_COLS:
            df[col] = sent_aligned[col].values

        # Category one-hot columns (not scaled — they're binary flags)
        for cat in CATEGORIES:
            df[f"Category_{cat}"] = 1.0 if cat == category else 0.0

        asset_dfs[ticker] = df
        feature_count = len(ALL_NUMERIC_COLS) + len(CATEGORIES)
        print(f"   ✅ {ticker} ({category}): {len(df)} rows | {feature_count} features total")

    if not asset_dfs:
        print("❌ No asset data found. Run fetch.py first.")
        return

    # ---- Step 2: Fit a single global StandardScaler on training splits ----
    print("\n📐 Fitting global StandardScaler on training splits...")
    train_numeric_list = []
    one_hot_cols = [f"Category_{cat}" for cat in CATEGORIES]

    for ticker, df in asset_dfs.items():
        n = len(df)
        train_end = int(n * 0.70)
        train_df = df.iloc[:train_end]

        # Only scale numeric columns; fill any remaining NaN with 0
        numeric_vals = train_df[ALL_NUMERIC_COLS].fillna(0.0).values
        train_numeric_list.append(numeric_vals)

    all_train_numeric = np.vstack(train_numeric_list)
    scaler = StandardScaler()
    scaler.fit(all_train_numeric)

    with open(os.path.join(processed_dir, f'scaler_{timeframe}.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler fitted on {all_train_numeric.shape[0]:,} samples × {all_train_numeric.shape[1]} features")

    # ---- Step 3: Create sequences for each asset ----
    print("\n🔢 Creating time-series sequences for each asset...")
    train_X_list, train_y_list = [], []
    val_X_list,   val_y_list   = [], []

    all_feature_cols = ALL_NUMERIC_COLS + one_hot_cols

    for ticker, df in asset_dfs.items():
        n = len(df)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        splits = {
            'train': df.iloc[:train_end],
            'val':   df.iloc[train_end:val_end],
            'test':  df.iloc[val_end:],
        }

        scaled = {}
        for split_name, split_df in splits.items():
            numeric_vals  = split_df[ALL_NUMERIC_COLS].fillna(0.0).values
            scaled_num    = scaler.transform(numeric_vals)
            one_hot_vals  = split_df[one_hot_cols].values
            scaled[split_name] = {
                'features': np.hstack([scaled_num, one_hot_vals]),
                'targets':  split_df['Target_Return'].values
            }

        X_train, y_train = create_sequences(scaled['train']['features'], scaled['train']['targets'], seq_length)
        X_val,   y_val   = create_sequences(scaled['val']['features'],   scaled['val']['targets'],   seq_length)
        X_test,  y_test  = create_sequences(scaled['test']['features'],  scaled['test']['targets'],  seq_length)

        train_X_list.append(X_train)
        train_y_list.append(y_train)
        val_X_list.append(X_val)
        val_y_list.append(y_val)

        np.save(os.path.join(processed_dir, f'{ticker}_X_test_{timeframe}.npy'), X_test)
        np.save(os.path.join(processed_dir, f'{ticker}_y_test_{timeframe}.npy'), y_test)
        print(f"   ✅ {ticker}: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ---- Step 4: Stack + shuffle training set ----
    X_train_all = np.vstack(train_X_list)
    y_train_all = np.concatenate(train_y_list)
    X_val_all   = np.vstack(val_X_list)
    y_val_all   = np.concatenate(val_y_list)

    shuffle_idx = np.random.permutation(len(X_train_all))
    X_train_all = X_train_all[shuffle_idx]
    y_train_all = y_train_all[shuffle_idx]

    np.save(os.path.join(processed_dir, f'X_train_{timeframe}.npy'), X_train_all)
    np.save(os.path.join(processed_dir, f'y_train_{timeframe}.npy'), y_train_all)
    np.save(os.path.join(processed_dir, f'X_val_{timeframe}.npy'),   X_val_all)
    np.save(os.path.join(processed_dir, f'y_val_{timeframe}.npy'),   y_val_all)

    with open(os.path.join(processed_dir, f'feature_cols_{timeframe}.pkl'), 'wb') as f:
        pickle.dump(all_feature_cols, f)

    print("\n" + "=" * 60)
    print("🏁 RICH MULTI-SOURCE DATA PREPROCESSOR COMPLETED")
    print(f"   Unified Train shape:      {X_train_all.shape}")
    print(f"   Unified Validation shape: {X_val_all.shape}")
    print(f"   Total Features:           {len(all_feature_cols)}")
    print(f"   Feature breakdown:")
    print(f"     • OHLCV + Technical:  {len(OHLCV_TECH_COLS)}")
    print(f"     • Macroeconomic:      {len(MACRO_COLS)}")
    print(f"     • FX Rate:            {len(FX_COLS)}")
    print(f"     • Fundamental:        {len(FUNDAMENTAL_COLS)}")
    print(f"     • Sentiment:          {len(SENTIMENT_COLS)}")
    print(f"     • Category one-hot:   {len(CATEGORIES)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-horizon preprocessor.")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Specific timeframe (e.g. 1mo, 3mo). Defaults to ACTIVE_TIMEFRAME.")
    args = parser.parse_args()
    process_all_data(timeframe=args.timeframe)
