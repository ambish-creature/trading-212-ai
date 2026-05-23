"""
backtest.py — Multi-asset backtesting engine (GBP, cash reserves, AER interest).

Key upgrades over previous version:
  - Starting capital: £5,000 GBP total (distributed across all assets)
  - Fractional shares: qty = trade_cash_gbp / price_gbp (always fractional)
  - Cash reserve floor: CASH_RESERVE_RATIO × portfolio_value must always remain
    in free cash. Prevents buying when reserve would be breached.
  - Daily AER interest: Each trading day, free_cash × (AER_year / 365) is earned
    and tracked separately as bank_interest_earned.
  - GBP/USD conversion: USD-denominated assets use daily GBPUSD FX rate.
    GBP-native assets (e.g. VWRL.L) do not need conversion.
  - All reporting in GBP.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import json
import argparse
import ta
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import (
    TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME, ASSETS, CATEGORIES,
    STARTING_FUND_GBP, CASH_RESERVE_RATIO, MAX_SINGLE_TRADE_RATIO,
    HISTORICAL_AER
)
from src.models.train import LSTMAttention
from src.models.predict import find_latest_model

# Assets priced in GBP (London Stock Exchange / GBP-denominated)
GBP_NATIVE_TICKERS = {"VWRL.L", "IGLT.L"}

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


def load_gbpusd(root_dir):
    """Loads the daily GBP/USD FX rate. Returns a pandas Series indexed by date."""
    path = os.path.join(root_dir, 'data/raw/GBPUSD.csv')
    if not os.path.exists(path):
        print("   ⚠️  GBP/USD FX data not found. Defaulting to 1.27 for all days.")
        return None
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    return df['GBPUSD']


def get_gbpusd_rate(fx_series, date, default=1.27):
    """Returns the GBP/USD rate for a given date (forward-fill if missing)."""
    if fx_series is None:
        return default
    try:
        # Get the most recent rate at or before the given date
        available = fx_series[fx_series.index <= date]
        if len(available) == 0:
            return default
        return float(available.iloc[-1])
    except Exception:
        return default


def price_to_gbp(price_usd, gbpusd_rate, is_gbp_native):
    """Converts a USD price to GBP. If already GBP-native, returns as-is."""
    if is_gbp_native:
        return price_usd
    return price_usd / gbpusd_rate


def get_aer_for_year(year):
    """Returns the AER for a given calendar year from the historical table."""
    return HISTORICAL_AER.get(year, HISTORICAL_AER.get(max(HISTORICAL_AER.keys()), 4.5)) / 100.0


def compute_aer_benchmark(start_date, end_date, starting_capital):
    """
    Computes what a £starting_capital savings account would have grown to
    over the period [start_date, end_date] at historical AER rates.
    Returns (final_value, total_interest, annualised_rate).
    """
    current = float(starting_capital)
    date = pd.Timestamp(start_date)
    end  = pd.Timestamp(end_date)
    total_days = (end - date).days
    if total_days <= 0:
        return current, 0.0, 0.0

    while date < end:
        aer = get_aer_for_year(date.year)
        daily_rate = aer / 365.0
        current *= (1 + daily_rate)
        date += pd.Timedelta(days=1)

    total_interest = current - starting_capital
    annualised_rate = (current / starting_capital) ** (365 / total_days) - 1
    return current, total_interest, annualised_rate


def run_technical_indicators(df):
    """Recomputes all technical indicators needed by the backtester on the raw dataframe."""
    close = df['Close']
    df['RSI']        = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_ind         = ta.trend.MACD(close=close)
    df['MACD']       = macd_ind.macd()
    df['MACD_Signal']= macd_ind.macd_signal()
    bb               = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['BB_High']    = bb.bollinger_hband()
    df['BB_Low']     = bb.bollinger_lband()
    df['SMA_20']     = close.rolling(20).mean()
    df['SMA_50']     = close.rolling(50).mean()
    return df


# ---------------------------------------------------------------------------
# Core backtesting engine (single ticker)
# ---------------------------------------------------------------------------

def run_single_backtest(
    ticker="SPY",
    strategy="advanced_ai",
    holding_days=5,
    confidence_threshold=40.0,
    initial_cash_gbp=None,
    timeframe=None,
):
    """
    Simulates the full trading strategy for a single ticker.

    initial_cash_gbp: starting GBP cash allocated for this ticker.
    If None, defaults to STARTING_FUND_GBP / number_of_assets.
    """
    if initial_cash_gbp is None:
        initial_cash_gbp = STARTING_FUND_GBP / len(ASSETS)

    raw_path      = os.path.join(ROOT_DIR, f'data/raw/{ticker}.csv')
    processed_dir = os.path.join(ROOT_DIR, 'data/processed/')

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found for {ticker}. Run fetch.py first.")

    if timeframe is None:
        timeframe = ACTIVE_TIMEFRAME

    df = pd.read_csv(raw_path, index_col='Date', parse_dates=True)
    df.sort_index(inplace=True)
    profile      = TIMEFRAME_PROFILES[timeframe]
    seq_length   = profile['seq_length']
    target_shift = profile['target_shift']

    df = run_technical_indicators(df)

    if 'Dividends' not in df.columns:
        df['Dividends'] = 0.0
    else:
        df['Dividends'] = df['Dividends'].fillna(0.0)

    df['Target_Return'] = df['Close'].pct_change(periods=target_shift).shift(-target_shift) * 100.0
    model_feature_cols  = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'Dividends']
    df.dropna(subset=model_feature_cols, inplace=True)
    df.dropna(subset=['Target_Return'], inplace=True)

    # Recreate test split boundaries (70% Train, 15% Val, 15% Test)
    n       = len(df)
    val_end = int(n * 0.85)
    test_df = df.iloc[val_end:].copy()

    # Load precomputed test sequences FIRST — use their count as the authoritative
    # length for decision_df. preprocess.py may have dropped more warmup rows than
    # the backtest's own dropna, causing a size mismatch if we use seq_length directly.
    X_test_path = os.path.join(processed_dir, f'{ticker}_X_test_{timeframe}.npy')
    if not os.path.exists(X_test_path):
        raise FileNotFoundError(f"Test sequences not found for {ticker} under timeframe {timeframe}. Run preprocess.py first.")
    X_test        = np.load(X_test_path)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_sequences   = X_test.shape[0]  # authoritative number of decision days

    # Decision day prices — take the LAST n_sequences rows of test_df.
    # Sequence i in X_test corresponds to the window ending on the (i+seq_length-1)th
    # row of the test split, so the decision day for sequence i is test_df.iloc[i+seq_length-1].
    # Equivalently: the last n_sequences rows of test_df align 1-to-1 with sequences.
    if len(test_df) < n_sequences:
        # Edge case: backtest split is shorter than preprocess split (shouldn't normally happen)
        raise ValueError(
            f"{ticker}: test_df has {len(test_df)} rows but X_test has {n_sequences} sequences. "
            "Re-run preprocess.py to realign."
        )
    decision_df = test_df.iloc[len(test_df) - n_sequences:].copy()
    close_prices_usd = decision_df['Close'].values
    dividends_usd    = decision_df['Dividends'].values
    dates            = decision_df.index

    decision_df['SMA_20_prev'] = decision_df['SMA_20'].shift(1)
    decision_df['SMA_50_prev'] = decision_df['SMA_50'].shift(1)

    # Load GBP/USD FX
    fx_series = load_gbpusd(ROOT_DIR)
    is_gbp_native = ticker in GBP_NATIVE_TICKERS

    # Load AI Model
    model, device = None, None
    if strategy in ["ai", "pnl_box", "advanced_ai"]:
        model_path, params_path = find_latest_model(timeframe)
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            ('mps' if torch.backends.mps.is_available() else 'cpu')
        )
        model = LSTMAttention(
            input_size=X_test.shape[2],
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.train()  # Enable MC Dropout

    # ---- Simulation state ----
    free_cash_gbp     = initial_cash_gbp
    open_positions    = []
    portfolio_history = []
    trade_log         = []
    total_dividends_gbp   = 0.0
    total_bank_interest   = 0.0

    start_date = dates[0]
    end_date   = dates[-1]

    for i in range(len(dates)):
        date           = dates[i]
        close_usd      = close_prices_usd[i]
        div_per_share_usd = dividends_usd[i]
        gbpusd         = get_gbpusd_rate(fx_series, date)
        close_gbp      = price_to_gbp(close_usd, gbpusd, is_gbp_native)
        div_per_share_gbp = price_to_gbp(div_per_share_usd, gbpusd, is_gbp_native)

        # ----- A. Daily bank AER interest on free cash -----
        aer = get_aer_for_year(date.year)
        daily_interest = free_cash_gbp * (aer / 365.0)
        free_cash_gbp  += daily_interest
        total_bank_interest += daily_interest

        # ----- B. Collect dividends -----
        if div_per_share_gbp > 0.0 and open_positions:
            total_qty     = sum(pos["qty"] for pos in open_positions)
            div_payout    = total_qty * div_per_share_gbp
            free_cash_gbp += div_payout
            total_dividends_gbp += div_payout
            trade_log.append({
                "type": "DIVIDEND", "date": date.strftime('%Y-%m-%d'),
                "qty": total_qty, "price_gbp": div_per_share_gbp,
                "pnl_gbp": div_payout, "pnl_pct": 0.0
            })

        # ----- C. Increment days held -----
        for pos in open_positions:
            pos["days_held"] += 1

        # ----- D. Strategy exit checks -----
        active_positions = []
        for pos in open_positions:
            if pos["qty"] <= 0.0:
                continue

            qty       = pos["qty"]
            buy_gbp   = pos["buy_price_gbp"]
            days_held = pos["days_held"]
            pnl       = qty * (close_gbp - buy_gbp)
            pnl_pct   = (close_gbp - buy_gbp) / (buy_gbp + 1e-9) * 100.0

            exit_triggered = False
            exit_type      = ""

            if strategy == "pnl_box":
                if close_gbp >= buy_gbp * 1.03:
                    exit_triggered = True; exit_type = "TAKE PROFIT (+3.0%)"
                elif close_gbp <= buy_gbp * 0.985:
                    exit_triggered = True; exit_type = "STOP LOSS (-1.5%)"

            elif strategy == "advanced_ai":
                initial_qty = pos.get("initial_qty", qty)

                # Price-target fractional sell: 30% when price hits buy + 2% of GBP value
                if not pos.get("price_target_sold", False) and close_gbp >= buy_gbp * 1.02:
                    sell_qty = round(initial_qty * 0.3, 6)
                    if sell_qty > 0 and pos["qty"] >= sell_qty:
                        pos["qty"]   = round(pos["qty"] - sell_qty, 6)
                        free_cash_gbp += sell_qty * close_gbp
                        pos["price_target_sold"] = True
                        trade_log.append({
                            "type": "SELL (30% PRICE TARGET)",
                            "date": date.strftime('%Y-%m-%d'),
                            "qty": sell_qty, "price_gbp": close_gbp,
                            "pnl_gbp": sell_qty * (close_gbp - buy_gbp),
                            "pnl_pct": pnl_pct
                        })

                # Time-delayed downside sell
                trigger_day = pos.get("downside_trigger_days_held")
                if trigger_day is not None and days_held == trigger_day + 2:
                    sell_qty = round(initial_qty * 0.3, 6)
                    if sell_qty > 0 and pos["qty"] >= sell_qty:
                        pos["qty"]   = round(pos["qty"] - sell_qty, 6)
                        free_cash_gbp += sell_qty * close_gbp
                        pos["downside_trigger_days_held"] = None
                        trade_log.append({
                            "type": "SELL (30% DELAYED DOWNSIDE)",
                            "date": date.strftime('%Y-%m-%d'),
                            "qty": sell_qty, "price_gbp": close_gbp,
                            "pnl_gbp": sell_qty * (close_gbp - buy_gbp),
                            "pnl_pct": pnl_pct
                        })

                if pos["qty"] <= 0.0:
                    continue

                if days_held >= holding_days:
                    exit_triggered = True; exit_type = "TIME EXIT"
                    qty   = pos["qty"]
                    pnl   = qty * (close_gbp - buy_gbp)
                    pnl_pct = (close_gbp - buy_gbp) / (buy_gbp + 1e-9) * 100.0

            if strategy not in ["advanced_ai"]:
                if not exit_triggered and days_held >= holding_days:
                    exit_triggered = True; exit_type = "TIME EXIT"

            if exit_triggered:
                free_cash_gbp += qty * close_gbp
                trade_log.append({
                    "type": f"SELL ({exit_type})",
                    "date": date.strftime('%Y-%m-%d'),
                    "qty": qty, "price_gbp": close_gbp,
                    "pnl_gbp": pnl, "pnl_pct": pnl_pct
                })
            else:
                active_positions.append(pos)

        open_positions = active_positions

        # ----- E. Signal generation -----
        buy_signal     = False
        sell_all_signal = False
        trade_confidence = 100.0

        if strategy in ["ai", "pnl_box", "advanced_ai"]:
            input_tensor = X_test_tensor[i].unsqueeze(0).to(device)
            all_mu, all_sigma = [], []
            with torch.no_grad():
                for _ in range(30):
                    mu, log_sigma = model(input_tensor)
                    all_mu.append(mu.cpu().item())
                    all_sigma.append(torch.exp(log_sigma).cpu().item())
            pred_return    = np.mean(all_mu)
            mc_std         = np.std(all_mu)
            model_sigma    = np.mean(all_sigma)
            total_unc      = np.sqrt(mc_std**2 + model_sigma**2)
            trade_confidence = max(0, min(100, 100 * np.exp(-total_unc * 0.5)))

            if pred_return > 0 and trade_confidence >= confidence_threshold:
                buy_signal = True
            elif pred_return < 0 and trade_confidence >= confidence_threshold:
                if strategy == "advanced_ai":
                    if trade_confidence >= 65.0:
                        for pos in open_positions:
                            if pos.get("downside_trigger_days_held") is None:
                                pos["downside_trigger_days_held"] = pos["days_held"]
                else:
                    sell_all_signal = True

        elif strategy == "sma":
            row = decision_df.iloc[i]
            if (row['SMA_20'] > row['SMA_50']) and (row.get('SMA_20_prev', 0) <= row.get('SMA_50_prev', 0)):
                buy_signal = True
            elif (row['SMA_20'] < row['SMA_50']) and (row.get('SMA_20_prev', 0) >= row.get('SMA_50_prev', 0)):
                sell_all_signal = True

        elif strategy == "rsi_bb":
            row = decision_df.iloc[i]
            if row['RSI'] < 30 and close_usd <= row['BB_Low']:
                buy_signal = True
            elif row['RSI'] > 70 or close_usd >= row['BB_High']:
                sell_all_signal = True

        # ----- F. Execute trades with cash reserve enforcement -----
        holdings_val_gbp = sum(pos["qty"] * close_gbp for pos in open_positions)
        portfolio_val_gbp = free_cash_gbp + holdings_val_gbp

        # Reserve floor: always keep this much in cash
        reserve_floor = portfolio_val_gbp * CASH_RESERVE_RATIO
        available_above_reserve = max(0.0, free_cash_gbp - reserve_floor)

        if buy_signal:
            # Maximum spend = min(MAX_SINGLE_TRADE_RATIO of available, available_above_reserve)
            trade_cash_gbp = available_above_reserve * MAX_SINGLE_TRADE_RATIO * (trade_confidence / 100.0)

            if trade_cash_gbp > 0.01 and close_gbp > 0:
                # Fractional share quantity
                qty = trade_cash_gbp / close_gbp
                actual_spend = qty * close_gbp
                free_cash_gbp -= actual_spend
                open_positions.append({
                    "qty":            qty,
                    "initial_qty":    qty,
                    "buy_price_gbp":  close_gbp,
                    "buy_price_usd":  close_usd,
                    "days_held":      0,
                    "price_target_sold": False,
                    "downside_trigger_days_held": None
                })
                trade_log.append({
                    "type": "BUY",
                    "date": date.strftime('%Y-%m-%d'),
                    "qty": qty, "price_gbp": close_gbp,
                    "pnl_gbp": 0.0, "pnl_pct": 0.0
                })

        elif sell_all_signal and open_positions:
            for pos in open_positions:
                qty = pos["qty"]
                sell_val = qty * close_gbp
                free_cash_gbp += sell_val
                pnl = qty * (close_gbp - pos["buy_price_gbp"])
                pnl_pct = (close_gbp - pos["buy_price_gbp"]) / (pos["buy_price_gbp"] + 1e-9) * 100.0
                trade_log.append({
                    "type": "SELL (SIGNAL EXIT)",
                    "date": date.strftime('%Y-%m-%d'),
                    "qty": qty, "price_gbp": close_gbp,
                    "pnl_gbp": pnl, "pnl_pct": pnl_pct
                })
            open_positions = []

        # Snapshot portfolio value for this day
        holdings_val_gbp = sum(pos["qty"] * close_gbp for pos in open_positions)
        portfolio_history.append(free_cash_gbp + holdings_val_gbp)

    portfolio_history = np.array(portfolio_history)
    net_pnl           = portfolio_history[-1] - initial_cash_gbp
    net_pnl_pct       = net_pnl / initial_cash_gbp * 100.0

    peaks     = np.maximum.accumulate(portfolio_history)
    drawdowns = (portfolio_history - peaks) / (peaks + 1e-9) * 100.0
    max_drawdown = float(np.min(drawdowns))

    buys           = [t for t in trade_log if t["type"] == "BUY"]
    sells          = [t for t in trade_log if "SELL" in t["type"]]
    winning_trades = [s for s in sells if s["pnl_gbp"] > 0]
    win_rate       = (len(winning_trades) / len(sells) * 100.0) if sells else 0.0

    # AER benchmark for this single ticker's cash allocation
    aer_final, aer_interest, aer_annualised = compute_aer_benchmark(
        start_date, end_date, initial_cash_gbp
    )

    return {
        'net_pnl':            net_pnl,
        'net_pnl_pct':        net_pnl_pct,
        'max_drawdown':       max_drawdown,
        'win_rate':           win_rate,
        'buys':               len(buys),
        'sells':              len(sells),
        'final_value':        portfolio_history[-1],
        'dividends_gbp':      total_dividends_gbp,
        'bank_interest_gbp':  total_bank_interest,
        'aer_benchmark_val':  aer_final,
        'aer_interest':       aer_interest,
        'aer_annualised_pct': aer_annualised * 100.0,
        'start_date':         str(start_date.date()),
        'end_date':           str(end_date.date()),
    }


# ---------------------------------------------------------------------------
# Helper: technical indicator functions (standalone)
# ---------------------------------------------------------------------------

def ta_momentum_rsi(close, window=14):
    return ta.momentum.RSIIndicator(close=close, window=window).rsi()

def ta_macd(close):
    m = ta.trend.MACD(close=close)
    return m.macd(), m.macd_signal()

def ta_bollinger(close):
    b = ta.volatility.BollingerBands(close=close)
    return b.bollinger_hband(), b.bollinger_lband()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_backtests(ticker="VOO", strategy="advanced_ai", holding_days=5, confidence_threshold=40.0, timeframe=None):
    per_asset_cash = STARTING_FUND_GBP / len(ASSETS)

    if timeframe is None:
        timeframe = ACTIVE_TIMEFRAME

    if ticker != "all":
        res = run_single_backtest(
            ticker=ticker, strategy=strategy,
            holding_days=holding_days, confidence_threshold=confidence_threshold,
            initial_cash_gbp=per_asset_cash,
            timeframe=timeframe
        )
        print("=" * 70)
        print(f"🏁 STRATEGY RESULT: {ticker} ({strategy.upper()})")
        print("=" * 70)
        print(f"   💰 Net Profit:        £{res['net_pnl']:+,.2f} ({res['net_pnl_pct']:+.2f}%)")
        print(f"   📉 Max Drawdown:      {res['max_drawdown']:.2f}%")
        print(f"   🎁 Dividends Earned:  £{res['dividends_gbp']:.2f}")
        print(f"   🏦 Bank Interest:     £{res['bank_interest_gbp']:.2f}")
        print(f"   📊 AER Benchmark:     £{res['aer_benchmark_val']:.2f} (+£{res['aer_interest']:.2f} | {res['aer_annualised_pct']:.2f}% p.a.)")
        print(f"   🎯 Win Rate:          {res['win_rate']:.1f}%")
        print(f"   🔄 Total Trades:      {res['buys']} buys / {res['sells']} sells")
        print("=" * 70)
        return res

    # ---- Run on all assets ----
    print("\n⏳ Running backtests across ALL configured assets...")
    print("=" * 110)
    header = f"{'TICKER':<10} | {'CAT':<8} | {'FINAL £':<10} | {'NET PnL':<18} | {'DRAWDOWN':<10} | {'DIVS':<8} | {'INTEREST':<10} | {'AER BENCH':<10} | TRADES"
    print(header)
    print("-" * 110)

    total_net_pnl       = 0.0
    total_dividends     = 0.0
    total_interest      = 0.0
    total_aer_bench     = 0.0
    total_initial       = STARTING_FUND_GBP

    results = {}
    for tk, cat in ASSETS.items():
        try:
            res = run_single_backtest(
                ticker=tk, strategy=strategy,
                holding_days=holding_days, confidence_threshold=confidence_threshold,
                initial_cash_gbp=per_asset_cash,
                timeframe=timeframe
            )
            pnl_str  = f"£{res['net_pnl']:+,.2f} ({res['net_pnl_pct']:+.2f}%)"
            row = (
                f"{tk:<10} | {cat:<8} | £{res['final_value']:<9,.2f} | {pnl_str:<18} | "
                f"{res['max_drawdown']:<9.2f}% | £{res['dividends_gbp']:<7.2f} | "
                f"£{res['bank_interest_gbp']:<9.2f} | £{res['aer_benchmark_val']:<9.2f} | "
                f"{res['buys']}B / {res['sells']}S"
            )
            print(row)
            total_net_pnl   += res['net_pnl']
            total_dividends += res['dividends_gbp']
            total_interest  += res['bank_interest_gbp']
            total_aer_bench += res['aer_benchmark_val']
            results[tk] = res
        except Exception as e:
            print(f"{tk:<10} | ❌ Failed: {e}")

    total_return_pct  = total_net_pnl / total_initial * 100.0

    if not results:
        print("❌ All tickers failed. Cannot compute combined summary.")
        return results

    # AER benchmark for the FULL portfolio
    first_res = results[list(results.keys())[0]]
    aer_full_val, aer_full_int, aer_full_ann = compute_aer_benchmark(
        first_res['start_date'],
        first_res['end_date'],
        total_initial
    )

    print("=" * 110)
    print(f"🏆 COMBINED PORTFOLIO RESULTS (Starting: £{total_initial:,.2f} GBP)")
    print(f"   • Total Final Value:       £{total_initial + total_net_pnl:,.2f}")
    print(f"   • Total Net PnL (Trading): £{total_net_pnl:+,.2f} ({total_return_pct:+.2f}%)")
    print(f"   • Total Dividends:         £{total_dividends:,.2f}")
    print(f"   • Total Bank Interest:     £{total_interest:,.2f}")
    print(f"   • All-in Return:           £{total_net_pnl + total_dividends + total_interest:+,.2f}")
    print(f"   ─────────────────────────────────────────────────")
    print(f"   📊 AER Benchmark:          £{aer_full_val:,.2f} (savings account return)")
    print(f"   📊 AER Interest:           +£{aer_full_int:,.2f} ({aer_full_ann:.2f}% p.a. avg)")

    combined_return = total_net_pnl + total_dividends + total_interest
    beats_aer = combined_return > aer_full_int
    multiplier = combined_return / aer_full_int if aer_full_int > 0 else float('inf')
    print(f"\n   {'✅' if beats_aer else '❌'} Bot {'BEATS' if beats_aer else 'MISSES'} the AER benchmark")
    if beats_aer:
        print(f"   🎉 Return is {multiplier:.2f}× the savings account interest!")
    print("=" * 110 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-asset historical backtester (GBP, AER-aware).")
    parser.add_argument("--ticker", type=str, default="all",
                        help="Ticker to evaluate, or 'all' for all assets.")
    parser.add_argument("--strategy", type=str, default="advanced_ai",
                        choices=["ai", "pnl_box", "sma", "rsi_bb", "advanced_ai"],
                        help="Strategy to evaluate.")
    parser.add_argument("--holding-days", type=int, default=5,
                        help="Holding period in days.")
    parser.add_argument("--confidence-threshold", type=float, default=40.0,
                        help="Minimum AI confidence %% to place a trade.")
    args = parser.parse_args()

    run_backtests(
        ticker=args.ticker,
        strategy=args.strategy,
        holding_days=args.holding_days,
        confidence_threshold=args.confidence_threshold
    )
