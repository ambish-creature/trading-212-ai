import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import json
import argparse
import ta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME, ASSETS, CATEGORIES
from src.models.train import LSTMAttention
from src.models.predict import find_latest_model

# ---------------------------------------------------------------------------
# 1. Backtesting Engine with Dividends
# ---------------------------------------------------------------------------

def run_single_backtest(ticker="SPY", strategy="ai", holding_days=5, confidence_threshold=40.0, base_trade_amount=1000.0, max_allocation=0.10, initial_cash=5000.0):
    raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../data/raw/{ticker}.csv'))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/'))
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found for {ticker} at {raw_path}. Run fetch.py first.")
        
    df = pd.read_csv(raw_path, index_col='Date', parse_dates=True)
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    seq_length = profile['seq_length']
    target_shift = profile['target_shift']
    
    # Calculate indicators
    df['RSI'] = ta_momentum_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = ta_macd(df['Close'])
    df['BB_High'], df['BB_Low'] = ta_bollinger(df['Close'])
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Preprocess dividends
    if 'Dividends' not in df.columns:
        df['Dividends'] = 0.0
    else:
        df['Dividends'] = df['Dividends'].fillna(0.0)
        
    # Recreate target shift and drop NaNs exactly like preprocess.py to align splits
    df['Target_Return'] = df['Close'].pct_change(periods=target_shift).shift(-target_shift) * 100.0
    model_feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'Dividends']
    df.dropna(subset=model_feature_cols, inplace=True)
    df.dropna(subset=['Target_Return'], inplace=True)
    
    # Recreate test split boundaries (70% Train, 15% Val, 15% Test)
    n = len(df)
    val_end = int(n * (0.7 + 0.15))
    test_df = df.iloc[val_end:].copy()
    
    # Load individual test sequences
    X_test_path = os.path.join(processed_dir, f'{ticker}_X_test.npy')
    X_test = np.load(X_test_path)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # decision days close prices, dates, and dividends
    decision_df = test_df.iloc[seq_length:].copy()
    close_prices = decision_df['Close'].values
    dividends = decision_df['Dividends'].values
    dates = decision_df.index
    
    # Pre-calculate crossovers
    decision_df['SMA_20_prev'] = decision_df['SMA_20'].shift(1)
    decision_df['SMA_50_prev'] = decision_df['SMA_50'].shift(1)
    
    # Load AI Model (only needed for 'ai' and 'pnl_box' strategies)
    model = None
    device = None
    feature_cols_len = X_test.shape[2]
    if strategy in ["ai", "pnl_box", "advanced_ai"]:
        model_path, params_path = find_latest_model(ACTIVE_TIMEFRAME)
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        model = LSTMAttention(
            input_size=feature_cols_len,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.train() # Enable dropout for MC simulation

    # Simulation variables
    free_cash = initial_cash
    open_positions = [] # list of dicts: {"qty": float, "buy_price": float, "days_held": int}
    
    portfolio_history = []
    trade_log = []
    total_dividends_received = 0.0
    
    for i in range(len(dates)):
        date = dates[i]
        close_price = close_prices[i]
        div_per_share = dividends[i]
        
        # A. Collect daily dividends on open positions
        if div_per_share > 0.0 and len(open_positions) > 0:
            total_qty = sum(pos["qty"] for pos in open_positions)
            div_payout = total_qty * div_per_share
            free_cash += div_payout
            total_dividends_received += div_payout
            trade_log.append({
                "type": "DIVIDEND",
                "date": date.strftime('%Y-%m-%d'),
                "qty": total_qty,
                "price": div_per_share,
                "pnl": div_payout,
                "pnl_pct": 0.0
            })

        # B. Update days held
        for pos in open_positions:
            pos["days_held"] += 1
            
        # C. Check for Strategy exits
        active_positions = []
        for pos in open_positions:
            if pos["qty"] <= 0.0:
                continue
                
            qty = pos["qty"]
            buy_price = pos["buy_price"]
            days_held = pos["days_held"]
            
            pnl = qty * (close_price - buy_price)
            pnl_pct = (close_price - buy_price) / buy_price * 100.0
            
            exit_triggered = False
            exit_type = ""
            
            # PnL Box strategy check
            if strategy == "pnl_box":
                if close_price >= buy_price * 1.03:
                    exit_triggered = True
                    exit_type = "TAKE PROFIT (+3.0%)"
                elif close_price <= buy_price * 0.985:
                    exit_triggered = True
                    exit_type = "STOP LOSS (-1.5%)"
                    
            # Advanced AI strategy check
            elif strategy == "advanced_ai":
                initial_qty = pos.get("initial_qty", qty)
                
                # Check 1: Price-Target Sell (sell 30% of original holdings when value hits buy_price + 20)
                if not pos.get("price_target_sold", False) and close_price >= buy_price + 20.0:
                    sell_qty = round(initial_qty * 0.3, 2)
                    if sell_qty > 0 and pos["qty"] >= sell_qty:
                        pos["qty"] = round(pos["qty"] - sell_qty, 2)
                        free_cash += sell_qty * close_price
                        pos["price_target_sold"] = True
                        pnl_fractional = sell_qty * (close_price - buy_price)
                        pnl_fractional_pct = (close_price - buy_price) / buy_price * 100.0
                        trade_log.append({
                            "type": "SELL (30% PRICE TARGET)",
                            "date": date.strftime('%Y-%m-%d'),
                            "qty": sell_qty,
                            "price": close_price,
                            "pnl": pnl_fractional,
                            "pnl_pct": pnl_fractional_pct
                        })
                        
                # Check 2: Time-Delayed Downside Sell (sell 30% exactly 2 days after downside trigger)
                trigger_day = pos.get("downside_trigger_days_held")
                if trigger_day is not None and days_held == trigger_day + 2:
                    sell_qty = round(initial_qty * 0.3, 2)
                    if sell_qty > 0 and pos["qty"] >= sell_qty:
                        pos["qty"] = round(pos["qty"] - sell_qty, 2)
                        free_cash += sell_qty * close_price
                        pos["downside_trigger_days_held"] = None # Clear trigger
                        pnl_fractional = sell_qty * (close_price - buy_price)
                        pnl_fractional_pct = (close_price - buy_price) / buy_price * 100.0
                        trade_log.append({
                            "type": "SELL (30% DELAYED DOWNSIDE)",
                            "date": date.strftime('%Y-%m-%d'),
                            "qty": sell_qty,
                            "price": close_price,
                            "pnl": pnl_fractional,
                            "pnl_pct": pnl_fractional_pct
                        })
                
                # Check if the position is now fully closed by fractional sells
                if pos["qty"] <= 0.0:
                    continue
                
                # Final Clean Time Exit (sell all remaining after holding_days)
                if days_held >= holding_days:
                    exit_triggered = True
                    exit_type = "TIME EXIT"
                    qty = pos["qty"] # Update remaining qty to sell
                    pnl = qty * (close_price - buy_price)
                    pnl_pct = (close_price - buy_price) / buy_price * 100.0
                    
            # Time exit check for other strategies
            if strategy != "advanced_ai":
                if not exit_triggered and days_held >= holding_days:
                    exit_triggered = True
                    exit_type = "TIME EXIT"
                
            if exit_triggered:
                free_cash += qty * close_price
                trade_log.append({
                    "type": f"SELL ({exit_type})",
                    "date": date.strftime('%Y-%m-%d'),
                    "qty": qty,
                    "price": close_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct
                })
            else:
                active_positions.append(pos)
                
        open_positions = active_positions

        # D. Strategy Signal Generation
        buy_signal = False
        sell_all_signal = False
        trade_confidence = 100.0

        if strategy in ["ai", "pnl_box", "advanced_ai"]:
            # Run AI prediction with 30 passes
            input_tensor = X_test_tensor[i].unsqueeze(0).to(device)
            all_mu = []
            all_sigma = []
            with torch.no_grad():
                for _ in range(30):
                    mu, log_sigma = model(input_tensor)
                    all_mu.append(mu.cpu().item())
                    all_sigma.append(torch.exp(log_sigma).cpu().item())
            pred_return = np.mean(all_mu)
            mc_std = np.std(all_mu)
            model_sigma = np.mean(all_sigma)
            total_unc = np.sqrt(mc_std**2 + model_sigma**2)
            trade_confidence = max(0, min(100, 100 * np.exp(-total_unc * 0.5)))
            
            if pred_return > 0 and trade_confidence >= confidence_threshold:
                buy_signal = True
            elif pred_return < 0 and trade_confidence >= confidence_threshold:
                if strategy == "advanced_ai":
                    # For advanced_ai, downside confidence triggers a delayed sell rather than liquidating instantly
                    if trade_confidence >= 65.0:
                        for pos in open_positions:
                            if pos.get("downside_trigger_days_held") is None:
                                pos["downside_trigger_days_held"] = pos["days_held"]
                else:
                    sell_all_signal = True

        elif strategy == "sma":
            row = decision_df.iloc[i]
            if (row['SMA_20'] > row['SMA_50']) and (row['SMA_20_prev'] <= row['SMA_50_prev']):
                buy_signal = True
            elif (row['SMA_20'] < row['SMA_50']) and (row['SMA_20_prev'] >= row['SMA_50_prev']):
                sell_all_signal = True

        elif strategy == "rsi_bb":
            row = decision_df.iloc[i]
            if row['RSI'] < 30 and close_price <= row['BB_Low']:
                buy_signal = True
            elif row['RSI'] > 70 or close_price >= row['BB_High']:
                sell_all_signal = True

        # E. Execute Trade Decisions
        current_holdings_val = sum(pos["qty"] * close_price for pos in open_positions)
        portfolio_value = free_cash + current_holdings_val

        if buy_signal:
            trade_cash = base_trade_amount * (trade_confidence / 100.0)
            
            max_allowed_val = portfolio_value * max_allocation
            current_val = current_holdings_val
            remaining_allowable = max_allowed_val - current_val
            
            if remaining_allowable > 0:
                if trade_cash > remaining_allowable:
                    trade_cash = remaining_allowable
                if free_cash < trade_cash:
                    trade_cash = free_cash
                    
                if trade_cash > 0:
                    qty = round(trade_cash / close_price, 2)
                    if qty > 0:
                        actual_spend = qty * close_price
                        free_cash -= actual_spend
                        open_positions.append({
                            "qty": qty,
                            "initial_qty": qty,
                            "buy_price": close_price,
                            "days_held": 0,
                            "price_target_sold": False,
                            "downside_trigger_days_held": None
                        })
                        trade_log.append({
                            "type": "BUY",
                            "date": date.strftime('%Y-%m-%d'),
                            "qty": qty,
                            "price": close_price,
                            "pnl": 0.0,
                            "pnl_pct": 0.0
                        })
                        
        elif sell_all_signal:
            if open_positions:
                for pos in open_positions:
                    qty = pos["qty"]
                    sell_val = qty * close_price
                    free_cash += sell_val
                    pnl = qty * (close_price - pos["buy_price"])
                    pnl_pct = (close_price - pos["buy_price"]) / pos["buy_price"] * 100.0
                    
                    trade_log.append({
                        "type": "SELL (SIGNAL EXIT)",
                        "date": date.strftime('%Y-%m-%d'),
                        "qty": qty,
                        "price": close_price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct
                    })
                open_positions = []

        # Recalculate daily portfolio value
        current_holdings_val = sum(pos["qty"] * close_price for pos in open_positions)
        portfolio_history.append(free_cash + current_holdings_val)

    portfolio_history = np.array(portfolio_history)
    net_pnl = portfolio_history[-1] - initial_cash
    net_pnl_pct = net_pnl / initial_cash * 100.0
    
    peaks = np.maximum.accumulate(portfolio_history)
    drawdowns = (portfolio_history - peaks) / peaks * 100.0
    max_drawdown = np.min(drawdowns)
    
    buys = [t for t in trade_log if t["type"] == "BUY"]
    sells = [t for t in trade_log if "SELL" in t["type"]]
    winning_trades = [s for s in sells if s["pnl"] > 0]
    win_rate = (len(winning_trades) / len(sells) * 100.0) if sells else 0.0
    
    return {
        'net_pnl': net_pnl,
        'net_pnl_pct': net_pnl_pct,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'buys': len(buys),
        'sells': len(sells),
        'final_value': portfolio_history[-1],
        'dividends': total_dividends_received
    }

# ---------------------------------------------------------------------------
# Technical Indicator Helpers (Manual definition to keep script self-contained)
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
# Main Orchestrator for Backtests
# ---------------------------------------------------------------------------
def run_backtests(ticker="SPY", strategy="ai", holding_days=5, confidence_threshold=40.0):
    if ticker != "all":
        print(f"Running backtest for {ticker} using '{strategy}' strategy...")
        res = run_single_backtest(ticker=ticker, strategy=strategy, holding_days=holding_days, confidence_threshold=confidence_threshold)
        print("=" * 60)
        print(f"🏁 STRATEGY RESULT: {ticker} ({strategy.upper()})")
        print("=" * 60)
        print(f"💰 Net Profit:         ${res['net_pnl']:+,.2f} ({res['net_pnl_pct']:+.2f}%)")
        print(f"📉 Max Drawdown:        {res['max_drawdown']:.2f}%")
        print(f"🎁 Dividends Earned:   ${res['dividends']:.2f} (Included in profit!)")
        print(f"🎯 Win Rate:            {res['win_rate']:.1f}%")
        print(f"🔄 Total Trades:        {res['buys']} buys / {res['sells']} sells")
        print("=" * 60)
        return

    # Run on all assets
    print("\n⏳ Running comparative simulations across ALL configured assets...")
    print("=" * 90)
    print(f"{'TICKER':<10} | {'CATEGORY':<10} | {'FINAL VALUE':<12} | {'NET PROFIT':<15} | {'MAX DRAWDOWN':<14} | {'DIVIDENDS':<11} | {'TRADES'}")
    print("-" * 90)
    
    total_net_pnl = 0.0
    total_dividends = 0.0
    initial_total = len(ASSETS) * 5000.0
    
    for tk, cat in ASSETS.items():
        try:
            res = run_single_backtest(ticker=tk, strategy=strategy, holding_days=holding_days, confidence_threshold=confidence_threshold)
            profit_str = f"${res['net_pnl']:+,.2f} ({res['net_pnl_pct']:+.2f}%)"
            print(f"{tk:<10} | {cat:<10} | ${res['final_value']:<11,.2f} | {profit_str:<15} | {res['max_drawdown']:<13.2f}% | ${res['dividends']:<10.2f} | {res['buys']} B / {res['sells']} S")
            total_net_pnl += res['net_pnl']
            total_dividends += res['dividends']
        except Exception as e:
            print(f"{tk:<10} | Failed: {e}")
            
    total_return_pct = total_net_pnl / initial_total * 100.0
    print("=" * 90)
    print(f"🏆 COMBINED MULTI-ASSET PORTFOLIO RESULTS:")
    print(f"   • Total Starting Portfolio:  ${initial_total:,.2f}")
    print(f"   • Total Portfolio Net PnL:   ${total_net_pnl:+,.2f} ({total_return_pct:+.2f}%)")
    print(f"   • Total Dividends Collected: ${total_dividends:,.2f}")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-asset historical backtester with dividends.")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker to evaluate, or 'all' to run on all assets.")
    parser.add_argument("--strategy", type=str, default="ai", choices=["ai", "pnl_box", "sma", "rsi_bb", "advanced_ai"], help="Strategy to evaluate.")
    parser.add_argument("--holding-days", type=int, default=5, help="Holding period in days.")
    parser.add_argument("--confidence-threshold", type=float, default=40.0, help="Confidence threshold.")
    args = parser.parse_args()
    
    run_backtests(ticker=args.ticker, strategy=args.strategy, holding_days=args.holding_days, confidence_threshold=args.confidence_threshold)
