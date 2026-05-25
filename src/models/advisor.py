"""
advisor.py — Personal Finance Advisor CLI

Fetches live data for ANY stock/ETF ticker, runs the AI model,
and gives you actionable buy/sell/hold advice with specific quantities,
timing recommendations, and price targets — like a personal finance advisor.

Usage examples:
  # Predict from today, N days forward (default 5)
  python src/models/advisor.py --ticker AAPL

  # Predict from today, looking 10 days forward
  python src/models/advisor.py --ticker TSLA --horizon 10

  # Predict from a specific past/future date (simulate what advice you'd get on that day)
  python src/models/advisor.py --ticker MSFT --date 2025-03-15 --horizon 5

  # With your current holding info (enables personalised hold/sell/buy-more advice)
  python src/models/advisor.py --ticker GOOGL --holding 50 --avg-cost 150.00

  # Non-interactive mode (no prompts, just output)
  python src/models/advisor.py --ticker SPY --horizon 5 --no-interactive
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import ta
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME, ASSETS, HISTORICAL_AER
from src.models.train import LSTMAttention
from src.models.predict import find_latest_model, mc_dropout_predict, load_model_and_params

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_category(ticker):
    """Returns the asset category, or 'Unknown' for tickers not in our training set."""
    return ASSETS.get(ticker.upper(), "Unknown")


def get_current_aer():
    """Returns the current year's AER rate."""
    year = datetime.now().year
    return HISTORICAL_AER.get(year, HISTORICAL_AER.get(max(HISTORICAL_AER.keys()), 4.5))


def fetch_ticker_info(ticker):
    """Fetches key info about the ticker from Yahoo Finance."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        return {
            'name':          info.get('longName', info.get('shortName', ticker)),
            'sector':        info.get('sector', info.get('category', 'N/A')),
            'currency':      info.get('currency', 'USD'),
            'dividend_yield': info.get('dividendYield', 0.0) or 0.0,
            'pe_ratio':      info.get('trailingPE', None),
            'market_cap':    info.get('marketCap', None),
            'beta':          info.get('beta', None),
            '52w_high':      info.get('fiftyTwoWeekHigh', None),
            '52w_low':       info.get('fiftyTwoWeekLow', None),
            'analyst_target': info.get('targetMeanPrice', None),
        }
    except Exception:
        return {
            'name': ticker, 'sector': 'N/A', 'currency': 'USD',
            'dividend_yield': 0.0, 'pe_ratio': None, 'market_cap': None,
            'beta': None, '52w_high': None, '52w_low': None, 'analyst_target': None,
        }


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


def prepare_input_for_date(ticker, reference_date, scaler, feature_cols, seq_length, timeframe=None):
    """
    Prepares the model input tensor using data up to (and including) `reference_date`.
    This lets you simulate advice as of any historical or current date.

    reference_date: datetime object (uses data up to this date)
    timeframe: the active timeframe string (e.g. '1mo', '3mo') — MUST be passed to get correct profile
    Returns: (input_tensor, current_price, actual_reference_date)
    """
    if timeframe is None:
        timeframe = ACTIVE_TIMEFRAME
    profile  = TIMEFRAME_PROFILES[timeframe]  # FIXED: was hardcoded to ACTIVE_TIMEFRAME
    interval = profile['interval']

    # Fetch enough history for indicators + sequence (fetch 3× seq_length days to be safe)
    fetch_start = reference_date - timedelta(days=seq_length * 3 + 100)
    fetch_end   = reference_date + timedelta(days=1)  # +1 to include the reference date

    data = yf.download(
        ticker,
        start=fetch_start.strftime('%Y-%m-%d'),
        end=fetch_end.strftime('%Y-%m-%d'),
        interval=interval,
        progress=False,
        actions=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    if len(data) < seq_length + 30:
        raise ValueError(
            f"Not enough data for {ticker} up to {reference_date.date()}. "
            f"Got {len(data)} rows, need at least {seq_length + 30}."
        )

    # Trim to data at or before reference_date
    data = data[data.index <= pd.Timestamp(reference_date)]
    actual_ref_date = data.index[-1]
    current_price = float(data['Close'].iloc[-1])

    # --- Compute all technical indicators ---
    close = data['Close']

    # Denoise close price causally using Hull Moving Average (lag-free)
    denoised = hull_moving_average(close, window=14).fillna(close)

    data['SMA_20']  = denoised.rolling(20).mean()
    data['SMA_50']  = denoised.rolling(50).mean()
    data['EMA_12']  = denoised.ewm(span=12, adjust=False).mean()
    data['EMA_26']  = denoised.ewm(span=26, adjust=False).mean()
    data['RSI']     = ta.momentum.RSIIndicator(close=denoised, window=14).rsi()

    macd_ind         = ta.trend.MACD(close=denoised)
    data['MACD']        = macd_ind.macd()
    data['MACD_Signal'] = macd_ind.macd_signal()
    data['MACD_Hist']   = macd_ind.macd_diff()

    bb = ta.volatility.BollingerBands(close=denoised, window=20, window_dev=2)
    data['BB_High']  = bb.bollinger_hband()
    data['BB_Mid']   = bb.bollinger_mavg()
    data['BB_Low']   = bb.bollinger_lband()
    data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / (data['BB_Mid'] + 1e-9)

    atr_ind = ta.volatility.AverageTrueRange(
        high=data['High'], low=data['Low'], close=close, window=14
    )
    data['ATR_14'] = atr_ind.average_true_range()
    data['Bid_Ask_Proxy'] = (data['High'] - data['Low']) / (close + 1e-9)

    # --- v2.0 Enhanced Features (must match preprocess.py) ---
    data['Momentum_5']  = denoised.pct_change(periods=5)  * 100.0
    data['Momentum_10'] = denoised.pct_change(periods=10) * 100.0
    data['Momentum_20'] = denoised.pct_change(periods=20) * 100.0

    # OBV (z-scored over 50-day window)
    obv = (np.sign(denoised.diff()) * data['Volume']).fillna(0).cumsum()
    obv_mean = obv.rolling(50).mean()
    obv_std  = obv.rolling(50).std().replace(0, 1e-9)
    data['OBV'] = (obv - obv_mean) / obv_std

    data['Vol_Regime']    = data['ATR_14'] / (denoised + 1e-9)
    data['Trend_Strength'] = (data['SMA_20'] / (data['SMA_50'] + 1e-9)) - 1.0

    highest_high = data['High'].rolling(14).max()
    lowest_low   = data['Low'].rolling(14).min()
    data['Williams_R'] = -100 * (highest_high - denoised) / (highest_high - lowest_low + 1e-9)
    data['Price_vs_SMA50'] = (denoised / (data['SMA_50'] + 1e-9)) - 1.0

    # --- v3.0 Cross-Asset Sector & Macro correlation features ---
    sector_ticker = SECTOR_MAP.get(ticker, "SPY")

    try:
        support_data = yf.download(
            [sector_ticker, "SPY", "GC=F", "CL=F"],
            start=fetch_start.strftime('%Y-%m-%d'),
            end=fetch_end.strftime('%Y-%m-%d'),
            interval=interval,
            progress=False
        )
        
        # Parse close prices causally
        if isinstance(support_data.columns, pd.MultiIndex):
            sec_close = support_data['Close'][sector_ticker].dropna()
            spy_close = support_data['Close']['SPY'].dropna()
            gold_close = support_data['Close']['GC=F'].dropna()
            oil_close = support_data['Close']['CL=F'].dropna()
        else:
            sec_close = support_data['Close'].dropna()
            spy_close = sec_close
            gold_close = sec_close
            oil_close = sec_close
            
        sec_denoised = hull_moving_average(sec_close, window=14).fillna(sec_close)
        spy_denoised = hull_moving_average(spy_close, window=14).fillna(spy_close)
        gold_denoised = hull_moving_average(gold_close, window=14).fillna(gold_close)
        oil_denoised = hull_moving_average(oil_close, window=14).fillna(oil_close)
        
        # Sector return
        sec_ret_5 = (sec_denoised.pct_change(periods=5) * 100.0).fillna(0.0)
        sec_ret_20 = (sec_denoised.pct_change(periods=20) * 100.0).fillna(0.0)
        
        # Global macro returns & correlations
        gold_ret_5 = (gold_denoised.pct_change(periods=5) * 100.0).fillna(0.0)
        oil_ret_5 = (oil_denoised.pct_change(periods=5) * 100.0).fillna(0.0)
        safe_haven = (gold_denoised / (spy_denoised + 1e-9)).fillna(0.0)
        eq_gold_corr = spy_denoised.rolling(20).corr(gold_denoised).fillna(0.0)
        eq_oil_corr = spy_denoised.rolling(20).corr(oil_denoised).fillna(0.0)
        
        # Align causally to data index
        data['Sector_Return_5d'] = sec_ret_5.reindex(data.index, method='ffill').fillna(0.0)
        data['Sector_Return_20d'] = sec_ret_20.reindex(data.index, method='ffill').fillna(0.0)
        data['Gold_Return_5d'] = gold_ret_5.reindex(data.index, method='ffill').fillna(0.0)
        data['Oil_Return_5d'] = oil_ret_5.reindex(data.index, method='ffill').fillna(0.0)
        data['Safe_Haven_Ratio'] = safe_haven.reindex(data.index, method='ffill').fillna(0.0)
        data['Equity_Gold_Corr'] = eq_gold_corr.reindex(data.index, method='ffill').fillna(0.0)
        data['Equity_Oil_Corr'] = eq_oil_corr.reindex(data.index, method='ffill').fillna(0.0)
        
    except Exception as ex:
        print(f"   ⚠️  Could not fetch/compute support indices: {ex}")
        for col in ['Sector_Return_5d', 'Sector_Return_20d', 'Gold_Return_5d', 
                    'Oil_Return_5d', 'Safe_Haven_Ratio', 'Equity_Gold_Corr', 'Equity_Oil_Corr']:
            data[col] = 0.0

    data['Relative_Strength_5d'] = data['Momentum_5'] - data['Sector_Return_5d']
    data['Relative_Strength_20d'] = data['Momentum_20'] - data['Sector_Return_20d']

    if 'Dividends' not in data.columns:
        data['Dividends'] = 0.0
    else:
        data['Dividends'] = data['Dividends'].fillna(0.0)


    # --- Load macro/FX/fundamental/sentiment data and align ---
    # Fallback values if external data files are not available
    macro_defaults = {
        'FedFundsRate': 5.0, 'US_10Y_Yield': 4.5, 'CPI_US': 315.0,
        'GDP_US': 2.5, 'Unemployment': 4.0, 'Oil_Price': 75.0, 'Gold_Price': 2300.0,
    }
    fx_default = 1.27

    macro_path = os.path.join(ROOT_DIR, 'data/macro/macro_combined.csv')
    if os.path.exists(macro_path):
        macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        for col, default in macro_defaults.items():
            if col in macro_df.columns:
                aligned = macro_df[[col]].reindex(data.index, method='ffill').bfill().fillna(default)
                data[col] = aligned[col].values
            else:
                data[col] = default
    else:
        for col, default in macro_defaults.items():
            data[col] = default

    fx_path = os.path.join(ROOT_DIR, 'data/raw/GBPUSD.csv')
    if os.path.exists(fx_path):
        fx_df = pd.read_csv(fx_path, index_col='Date', parse_dates=True)
        aligned_fx = fx_df[['GBPUSD']].reindex(data.index, method='ffill').bfill().fillna(fx_default)
        data['GBPUSD'] = aligned_fx['GBPUSD'].values
    else:
        data['GBPUSD'] = fx_default

    # Fundamental & sentiment — use ticker's saved files or zero
    fundamental_cols = [
        'PE_Ratio', 'Forward_PE', 'PB_Ratio', 'ROE', 'DE_Ratio',
        'EPS', 'Revenue_Growth', 'Dividend_Yield', 'Profit_Margin', 'EV_EBITDA',
    ]
    sentiment_cols = ['Analyst_Score', 'News_Sentiment', 'Institutional_Pct']

    fund_path = os.path.join(ROOT_DIR, f'data/fundamentals/{ticker}_fundamentals.csv')
    if os.path.exists(fund_path):
        fund_df = pd.read_csv(fund_path, index_col=0, parse_dates=True)
        for col in fundamental_cols:
            if col in fund_df.columns:
                aligned = fund_df[[col]].reindex(data.index, method='ffill').bfill().fillna(0.0)
                data[col] = aligned[col].values
            else:
                data[col] = 0.0
    else:
        for col in fundamental_cols:
            data[col] = 0.0

    sent_path = os.path.join(ROOT_DIR, f'data/sentiment/{ticker}_sentiment.csv')
    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path, index_col='Date', parse_dates=True)
        for col in sentiment_cols:
            if col in sent_df.columns:
                aligned = sent_df[[col]].reindex(data.index, method='ffill').bfill().fillna(0.0)
                data[col] = aligned[col].values
            else:
                data[col] = 0.0
    else:
        for col in sentiment_cols:
            data[col] = 0.0

    # Category one-hot (use best-guess from ASSETS or zeros for unknown tickers)
    category = get_category(ticker)
    from src.config import CATEGORIES
    for cat in CATEGORIES:
        data[f'Category_{cat}'] = 1.0 if cat == category else 0.0

    # --- Select & scale features ---
    # Numeric columns to scale (must match what scaler was fitted on)
    numeric_feature_cols = [c for c in feature_cols if not c.startswith('Category_')]
    one_hot_cols         = [c for c in feature_cols if c.startswith('Category_')]

    # Fill any remaining NaN
    for col in numeric_feature_cols:
        if col not in data.columns:
            data[col] = 0.0
    data[numeric_feature_cols] = data[numeric_feature_cols].fillna(0.0)
    data.dropna(subset=['RSI', 'SMA_50', 'ATR_14'], inplace=True)

    if len(data) < seq_length:
        raise ValueError(f"After indicator warmup, only {len(data)} rows remain (need {seq_length}).")

    scaled_num = scaler.transform(data[numeric_feature_cols].values)
    one_hot    = data[one_hot_cols].values
    all_feats  = np.hstack([scaled_num, one_hot])

    input_seq    = all_feats[-seq_length:]
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

    return input_tensor, current_price, actual_ref_date


# ---------------------------------------------------------------------------
# Advice generation logic
# ---------------------------------------------------------------------------

def generate_advice(
    ticker, current_price, pred_result, horizon_days,
    holding_shares=0.0, avg_cost=None, info=None, reference_date=None
):
    """
    Converts raw model output into human-readable personalised financial advice.

    Parameters:
        ticker:          Stock ticker symbol
        current_price:   Current (or reference-date) price in native currency
        pred_result:     Dict from mc_dropout_predict
        horizon_days:    How many trading days ahead the prediction is for
        holding_shares:  How many shares the user is currently holding (0 = not holding)
        avg_cost:        Average cost per share (if holding)
        info:            Dict of ticker metadata from Yahoo Finance
        reference_date:  The reference date for the prediction
    """
    mu         = pred_result['predicted_return']    # % return predicted
    confidence = pred_result['confidence']           # 0-100
    lower      = pred_result['lower_bound']          # 10th percentile return %
    upper      = pred_result['upper_bound']          # 90th percentile return %

    # Projected prices
    target_price = current_price * (1 + mu / 100.0)
    price_low    = current_price * (1 + lower / 100.0)
    price_high   = current_price * (1 + upper / 100.0)

    # Current holding value
    holding_value = holding_shares * current_price if holding_shares > 0 else 0.0
    current_pnl   = (current_price - avg_cost) * holding_shares if (avg_cost and holding_shares > 0) else 0.0
    current_pnl_pct = ((current_price / avg_cost) - 1) * 100 if (avg_cost and avg_cost > 0) else 0.0

    currency = info.get('currency', 'USD') if info else 'USD'
    curr_sym = '£' if currency == 'GBX' or currency == 'GBp' else (
               '£' if currency == 'GBP' else '$'
    )

    # Classify signal strength
    if confidence >= 70:
        strength = "STRONG"
        strength_emoji = "💪"
    elif confidence >= 45:
        strength = "MODERATE"
        strength_emoji = "👍"
    else:
        strength = "WEAK"
        strength_emoji = "⚠️"

    # Current AER for comparison
    current_aer = get_current_aer()
    daily_aer_rate = current_aer / 365.0 / 100.0
    bank_equivalent = mu / 100.0 / (horizon_days * daily_aer_rate) if horizon_days > 0 else 0.0

    lines = []
    lines.append("")
    lines.append("═" * 62)
    lines.append(f"  🤖 AI PERSONAL FINANCE ADVISOR — {ticker}")
    if info:
        lines.append(f"  📌 {info.get('name', ticker)} | {info.get('sector', 'N/A')}")
    lines.append("═" * 62)

    ref_str = reference_date.strftime('%Y-%m-%d') if reference_date else "Today"
    lines.append(f"  📅 Reference Date:      {ref_str}")
    lines.append(f"  📅 Prediction Horizon:  {horizon_days} trading day(s) later")
    lines.append(f"  💰 Current Price:       {curr_sym}{current_price:.2f}")
    lines.append(f"  🎯 AI Target Price:     {curr_sym}{target_price:.2f}  ({mu:+.2f}%)")
    lines.append(f"  📊 80%% Price Range:     {curr_sym}{price_low:.2f} — {curr_sym}{price_high:.2f}")
    lines.append(f"  🧠 AI Confidence:       {confidence:.0f}%  ({strength_emoji} {strength})")

    if info:
        if info.get('52w_high'):
            lines.append(f"  📈 52-Week High/Low:    {curr_sym}{info['52w_high']:.2f} / {curr_sym}{info['52w_low']:.2f}")
        if info.get('analyst_target'):
            lines.append(f"  👨‍💼 Analyst Consensus:   {curr_sym}{info['analyst_target']:.2f} (mean target)")
        if info.get('dividend_yield') and info['dividend_yield'] > 0:
            lines.append(f"  🎁 Dividend Yield:      {info['dividend_yield']*100:.2f}% p.a.")
        if info.get('beta'):
            risk_label = "Low" if info['beta'] < 0.8 else ("High" if info['beta'] > 1.3 else "Medium")
            lines.append(f"  ⚡ Beta (Volatility):   {info['beta']:.2f}  [{risk_label} risk vs market]")

    lines.append("─" * 62)

    # ── CURRENT HOLDING SECTION ──
    if holding_shares > 0 and avg_cost:
        lines.append(f"  📦 YOUR HOLDING:")
        lines.append(f"      {holding_shares:.4f} shares @ {curr_sym}{avg_cost:.2f} avg cost")
        lines.append(f"      Current Value:   {curr_sym}{holding_value:.2f}")
        pnl_emoji = "🟢" if current_pnl >= 0 else "🔴"
        lines.append(f"      Unrealised P&L:  {pnl_emoji} {curr_sym}{current_pnl:+.2f} ({current_pnl_pct:+.2f}%)")
        lines.append("─" * 62)
    lines.append("")
    lines.append("  📋 RECOMMENDATION:")
    lines.append("")

    if confidence < 75.0:
        # ─────────────────── SELECTIVE CLASSIFICATION (ACCURACY FILTER) ───────────────────
        if confidence < 50.0:
            lines.append(f"  ❓ Signal: UNCERTAIN MARKET / UNKNOWN PATTERN  (Confidence {confidence:.0f}% — Low Agreement)")
            lines.append("")
            lines.append(f"     The model indicates that market uncertainty is currently too high")
            lines.append("     to identify a reliable trend or pattern. Different simulation paths")
            lines.append("     heavily conflict, indicating a high-risk or neutral market regime.")
            if holding_shares > 0:
                lines.append(f"     ➡  Action: HOLD cash/shares. Do NOT average down or sell in panic.")
                lines.append(f"     ➡  Wait for market volatility to subside before adjusting your position.")
            else:
                lines.append(f"     ➡  Action: STAND ASIDE. Do NOT open a new position.")
                lines.append(f"     ➡  Hold 100% Cash for this asset. Capital preservation is priority.")
        else:
            lines.append(f"  ⏸️  Signal: HOLD / WAIT  (Confidence {confidence:.0f}% < 75% High-Accuracy Filter)")
            lines.append("")
            lines.append(f"     The AI predicts a return of {mu:+.2f}%, but the self-estimated confidence")
            lines.append("     is below the required 75% threshold to guarantee ≥80% direction accuracy.")
            if holding_shares > 0:
                lines.append(f"     ➡  HOLD your current {holding_shares:.4f} shares.")
                lines.append(f"     ➡  Do NOT add more until a high-confidence signal (>=75%) appears.")
            else:
                lines.append(f"     ➡  Do NOT open a new position yet.")
                lines.append(f"     ➡  Wait for a clearer, high-confidence signal. Re-run in 2-3 trading days.")
    elif mu > 0 and confidence >= 45:
        # ─────────────────── BULLISH ───────────────────
        lines.append(f"  📈 Signal: BUY / BUY MORE  ({strength} — {confidence:.0f}% confidence)")
        lines.append("")

        if holding_shares > 0:
            # Already holding — advice on adding more
            add_fraction = min(0.5, confidence / 100.0)  # buy up to 50% more
            lines.append(f"  ✅ You are already holding {ticker}. The AI suggests adding more.")
            lines.append(f"     ➡  Consider buying an additional ~{add_fraction*100:.0f}% of your")
            lines.append(f"        current holding ({holding_shares * add_fraction:.4f} shares ≈ "
                         f"{curr_sym}{holding_shares * add_fraction * current_price:.2f})")
            lines.append(f"     ➡  Timing: Buy within the next 1-2 trading days")
            lines.append("")
            lines.append(f"  📤 PROFIT-TAKING PLAN (if price rises):")
            lines.append(f"     • Sell ~30%% of TOTAL holding when price hits {curr_sym}{target_price:.2f}")
            lines.append(f"       (that's {holding_shares * 1.3 * 0.3:.4f} shares ≈ "
                         f"{curr_sym}{holding_shares * 1.3 * 0.3 * target_price:.2f})")
            lines.append(f"     • Sell another 20%% after {max(2, horizon_days // 2)} trading days "
                         f"regardless of price")
            lines.append(f"     • Hold remaining position until {curr_sym}{price_high:.2f} (upper range)")
        else:
            # Not holding — advice on entering a new position
            lines.append(f"  ✅ This looks like a good entry point for {ticker}.")
            if confidence >= 70:
                entry_size_pct = 15
            elif confidence >= 55:
                entry_size_pct = 10
            else:
                entry_size_pct = 7
            lines.append(f"     ➡  Suggested position size: ~{entry_size_pct}%% of your available cash")
            lines.append(f"     ➡  Enter in 1-2 tranches to average your cost")
            lines.append(f"     ➡  Buy Tranche 1: Now (today)")
            lines.append(f"     ➡  Buy Tranche 2: In 2 trading days (if price dips or holds)")
            lines.append("")
            lines.append(f"  📤 EXIT PLAN:")
            lines.append(f"     • Set take-profit at {curr_sym}{target_price:.2f} (+{mu:.1f}%)")
            lines.append(f"       → Sell 30%% of position at this level")
            lines.append(f"     • Set a stop-loss at {curr_sym}{current_price * 0.975:.2f} (-2.5%%)")
            lines.append(f"     • After {horizon_days} trading days: sell remaining if target not hit")

    elif mu < 0 and confidence >= 45:
        # ─────────────────── BEARISH ───────────────────
        lines.append(f"  📉 Signal: SELL / DO NOT BUY  ({strength} — {confidence:.0f}% confidence)")
        lines.append("")

        if holding_shares > 0:
            if current_pnl_pct > 5:
                # Holding at a profit — take some profit now
                sell_pct = min(70, int(confidence * 0.8))
                hold_pct = 100 - sell_pct
                lines.append(f"  ⚠️  You have a profit of {curr_sym}{current_pnl:+.2f} (+{current_pnl_pct:.1f}%).")
                lines.append(f"     The AI predicts a drop to ~{curr_sym}{target_price:.2f} over {horizon_days} days.")
                lines.append(f"     ➡  Sell {sell_pct}%% now  → "
                             f"{holding_shares * sell_pct/100:.4f} shares ≈ "
                             f"{curr_sym}{holding_shares * sell_pct/100 * current_price:.2f}")
                lines.append(f"     ➡  Hold {hold_pct}%% and reassess in {horizon_days} trading days")
            elif current_pnl_pct < -5:
                # Holding at a loss — cut losses
                lines.append(f"  🔴 You are currently at a loss ({curr_sym}{current_pnl:+.2f}, "
                             f"{current_pnl_pct:.1f}%).")
                lines.append(f"     The AI predicts further decline to ~{curr_sym}{target_price:.2f}.")
                lines.append(f"     ➡  Consider cutting 50%% of your position NOW to limit losses")
                lines.append(f"     ➡  Cut the remaining 50%% if price falls below {curr_sym}{price_low:.2f}")
            else:
                # Near breakeven — reduce risk
                lines.append(f"  ⚠️  You are near breakeven on {ticker}.")
                lines.append(f"     The AI predicts a {mu:.1f}%% move to {curr_sym}{target_price:.2f}.")
                lines.append(f"     ➡  Reduce position by 50%% now to protect capital")
                lines.append(f"     ➡  Re-enter when the AI signals bullish again")
        else:
            lines.append(f"  ❌ Do NOT open a new position in {ticker} right now.")
            lines.append(f"     The AI predicts the price may fall to ~{curr_sym}{target_price:.2f}")
            lines.append(f"     over the next {horizon_days} trading day(s).")
            lines.append(f"     ➡  Wait and watch. Consider re-checking in {horizon_days} days.")
            lines.append(f"     ➡  A possible buying opportunity may arise near {curr_sym}{price_low:.2f}")

    else:
        # ─────────────────── NEUTRAL / LOW CONFIDENCE ───────────────────
        lines.append(f"  ⏸️  Signal: HOLD / WAIT  (Low confidence — {confidence:.0f}%)")
        lines.append("")
        lines.append(f"     The AI is uncertain about {ticker}'s direction.")
        lines.append(f"     Predicted return is {mu:+.2f}% but confidence is too low to act.")
        if holding_shares > 0:
            lines.append(f"     ➡  HOLD your current {holding_shares:.4f} shares.")
            lines.append(f"     ➡  Do NOT add more until signal strengthens (>45% confidence).")
        else:
            lines.append(f"     ➡  Do NOT open a new position yet.")
            lines.append(f"     ➡  Wait for a clearer signal. Re-run in 2-3 trading days.")

    lines.append("")

    # ── RISK WARNINGS ──
    lines.append("─" * 62)
    lines.append("  ⚠️  RISK NOTES:")
    if pred_result['total_uncertainty'] > 3.0:
        lines.append("  • HIGH uncertainty detected — market is currently volatile.")
        lines.append("    Consider sizing your position 50%% smaller than usual.")
    if info and info.get('beta') and info['beta'] > 1.5:
        lines.append(f"  • High Beta ({info['beta']:.2f}): This stock moves ~{info['beta']:.1f}× the market.")
        lines.append("    A market correction would hit this stock hard.")
    if current_price == pred_result.get('52w_high'):
        lines.append("  • Price is near 52-week HIGH. Upside may be limited.")
    lines.append(f"  • AI accuracy is not guaranteed. Always diversify.")
    lines.append(f"  • Current savings account AER: {current_aer:.2f}%% p.a.")
    if abs(mu) / horizon_days * 252 < current_aer:
        lines.append(f"    The predicted return ({mu:+.2f}%% in {horizon_days} days) is "
                     f"BELOW annualised bank AER.")
        lines.append(f"    Consider whether this trade is worth the risk vs simply saving.")
    else:
        annualised_equiv = mu / horizon_days * 252
        lines.append(f"    Predicted return is equivalent to ~{annualised_equiv:.1f}%% p.a. "
                     f"(vs bank AER {current_aer:.1f}%%).")

    lines.append("")
    lines.append("═" * 62)
    lines.append("  ⚡ This advice is AI-generated. Not financial advice.")
    lines.append("     Always do your own research before investing.")
    lines.append("═" * 62)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_advisor(
    ticker="VOO",
    horizon_days="1mo",
    reference_date=None,
    holding_shares=0.0,
    avg_cost=None,
    ask_holdings=False,
):
    """Full advisor pipeline."""
    ticker = ticker.upper().strip()

    # Default reference date = today
    if reference_date is None:
        reference_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # --- Map horizon to timeframe ---
    timeframe = "1mo" # default
    
    if isinstance(horizon_days, str):
        horizon_str = horizon_days.strip().lower()
        if horizon_str in ["1mo", "2mo", "3mo", "6mo", "9mo", "1yr", "2yr"]:
            timeframe = horizon_str
            days_map = {"1mo": 21, "2mo": 42, "3mo": 63, "6mo": 126, "9mo": 189, "1yr": 252, "2yr": 504}
            horizon_days = days_map[horizon_str]
        else:
            try:
                horizon_days = int(horizon_days)
            except ValueError:
                print(f"⚠️ Unknown horizon format '{horizon_days}', using default '1mo' (21 trading days)")
                timeframe = "1mo"
                horizon_days = 21
                
    if isinstance(horizon_days, int):
        if horizon_days <= 10:
            timeframe = "1mo"
        elif horizon_days <= 30:
            timeframe = "1mo"
        elif horizon_days <= 50:
            timeframe = "2mo"
        elif horizon_days <= 80:
            timeframe = "3mo"
        elif horizon_days <= 150:
            timeframe = "6mo"
        elif horizon_days <= 220:
            timeframe = "9mo"
        elif horizon_days <= 380:
            timeframe = "1yr"
        else:
            timeframe = "2yr"

    print(f"\n⏳ Fetching data and running AI model for {ticker}...")
    print(f"   Reference date: {reference_date.date()} | Horizon: {horizon_days} trading days (using model: {timeframe})")

    # --- Load model ---
    model_path, params_path = find_latest_model(timeframe)
    with open(params_path, 'r') as f:
        best_params = json.load(f)

    data_dir     = os.path.join(ROOT_DIR, 'data/processed/')
    scaler_path  = os.path.join(data_dir, f'scaler_{timeframe}.pkl')
    feature_path = os.path.join(data_dir, f'feature_cols_{timeframe}.pkl')

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler for timeframe '{timeframe}' not found. Run preprocess.py first.")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feature_path, 'rb') as f:
        feature_cols = pickle.load(f)

    profile    = TIMEFRAME_PROFILES[timeframe]
    seq_length = profile['seq_length']

    # Adjust sequence for custom horizon (we still use the base model)
    input_tensor, current_price, actual_ref_date = prepare_input_for_date(
        ticker, reference_date, scaler, feature_cols, seq_length, timeframe=timeframe
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    # Use load_model_and_params for proper architecture detection (new vs legacy)
    model, best_params_loaded, model_path_used = load_model_and_params(timeframe, device)
    # Merge any extra params from saved file that might not be in best_params_loaded
    for k, v in best_params.items():
        if k not in best_params_loaded:
            best_params_loaded[k] = v
    best_params = best_params_loaded

    # Run MC Dropout (100 samples for high-quality uncertainty)
    opt_ratio = best_params.get('optimal_ratio_threshold', 0.15)  # FIXED: default 0.15 not 0.35
    pred_result = mc_dropout_predict(model, input_tensor, device, n_samples=100, optimal_ratio_threshold=opt_ratio)



    # Direct scaling from trained horizon target
    model_trained_days = {"1mo": 21, "2mo": 42, "3mo": 63, "6mo": 126, "9mo": 189, "1yr": 252, "2yr": 504}
    trained_days = model_trained_days.get(timeframe, 21)
    
    scale_factor = horizon_days / trained_days
    scaled_return = pred_result['predicted_return'] * scale_factor
    scaled_lower  = pred_result['lower_bound'] * scale_factor
    scaled_upper  = pred_result['upper_bound'] * scale_factor

    pred_scaled = dict(pred_result)
    pred_scaled['predicted_return'] = scaled_return
    pred_scaled['lower_bound']       = scaled_lower
    pred_scaled['upper_bound']       = scaled_upper

    # Fetch additional ticker metadata
    print(f"   📡 Fetching ticker metadata...")
    info = fetch_ticker_info(ticker)

    # --- Auto-load holding info from portfolios if not manually provided ---
    if holding_shares == 0.0 and not ask_holdings:
        try:
            from src.trading.execution import get_portfolio_positions
            
            # Fetch real positions first (to analyze your actual ISA assets)
            print(f"   📦 Checking your portfolios for {ticker}...")
            real_positions = get_portfolio_positions(real=True)
            
            matched_pos = None
            for pos in real_positions:
                pos_ticker = pos.get("ticker", "")
                from src.config import TICKER_MAPPING
                mapped_ticker = TICKER_MAPPING.get(ticker, ticker)
                if pos_ticker == mapped_ticker or ticker in pos_ticker or pos_ticker in ticker:
                    matched_pos = pos
                    print(f"      👉 Found in your REAL portfolio: {pos_ticker}")
                    break
                    
            if not matched_pos:
                # Fallback to DEMO portfolio
                demo_positions = get_portfolio_positions(real=False)
                for pos in demo_positions:
                    pos_ticker = pos.get("ticker", "")
                    from src.config import TICKER_MAPPING
                    mapped_ticker = TICKER_MAPPING.get(ticker, ticker)
                    if pos_ticker == mapped_ticker or ticker in pos_ticker or pos_ticker in ticker:
                        matched_pos = pos
                        print(f"      👉 Found in your DEMO portfolio: {pos_ticker}")
                        break
            
            if matched_pos:
                holding_shares = float(matched_pos.get("quantity", 0.0))
                # Fallback list of possible average price keys
                avg_cost = matched_pos.get("averagePrice", matched_pos.get("avgPrice", matched_pos.get("averagePricePaid", None)))
                if avg_cost is not None:
                    avg_cost = float(avg_cost)
                print(f"      👉 Auto-loaded: {holding_shares:.4f} shares @ avg cost {avg_cost}")
        except Exception as e:
            print(f"   ⚠️  Could not auto-load portfolio holdings: {e}")

    # --- Interactive prompt for holding info (manual fallback) ---
    if ask_holdings and holding_shares == 0.0:
        print(f"\n💬 Do you currently hold {ticker}? (Press Enter to use defaults)")
        try:
            h = input(f"   Shares held (e.g. 10.5, or 0): ").strip()
            if h and float(h) > 0:
                holding_shares = float(h)
                c = input(f"   Average cost per share (e.g. 150.00): ").strip()
                avg_cost = float(c) if c else None
        except (ValueError, EOFError):
            pass


    # --- Generate and print advice ---
    advice = generate_advice(
        ticker=ticker,
        current_price=current_price,
        pred_result=pred_scaled,
        horizon_days=horizon_days,
        holding_shares=holding_shares,
        avg_cost=avg_cost,
        info=info,
        reference_date=actual_ref_date,
    )
    print(advice)

    return pred_scaled, advice


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Personal Finance Advisor — get buy/sell/hold advice for any stock.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict VOO (default) 1 month ahead
  python src/models/advisor.py

  # Predict TSM 3 months ahead
  python src/models/advisor.py --ticker TSM --horizon 3mo

  # Simulate advice as of a specific past date (e.g. March 15, 2025)
  python src/models/advisor.py --ticker MSFT --date 2025-03-15 --horizon 6mo

  # Ask interactively for holdings details
  python src/models/advisor.py --ticker GOOGL --ask
        """
    )
    parser.add_argument("--ticker",         type=str,   default="VOO",
                        help="Stock/ETF ticker (e.g. VOO, AAPL, BTC-USD, GC=F) (default: VOO)")
    parser.add_argument("--horizon",        type=str,   default="1mo",
                        help="Horizon/timeframe (e.g. 1mo, 3mo, 6mo, 1yr, or trading days integer) (default: 1mo)")
    parser.add_argument("--date",           type=str,   default=None,
                        help="Reference date YYYY-MM-DD (default: today). "
                             "Use a past date to simulate what advice you would have received then.")
    parser.add_argument("--holding",        type=float, default=0.0,
                        help="Number of shares you currently hold (default: 0)")
    parser.add_argument("--avg-cost",       type=float, default=None,
                        help="Your average cost per share (used to calculate unrealised P&L)")
    parser.add_argument("--ask",            action="store_true",
                        help="Interactively ask for your holdings and average cost details (default: OFF)")

    args = parser.parse_args()

    ref_date = None
    if args.date:
        ref_date = datetime.strptime(args.date, '%Y-%m-%d')

    run_advisor(
        ticker=args.ticker,
        horizon_days=args.horizon,
        reference_date=ref_date,
        holding_shares=args.holding,
        avg_cost=args.avg_cost,
        ask_holdings=args.ask,
    )
