import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME
from src.models.train import LSTMAttention
from src.models.predict import find_latest_model
from src.data.preprocess import add_technical_indicators

# ---------------------------------------------------------------------------
# 1. Backtesting Engine
# ---------------------------------------------------------------------------

def run_single_backtest(strategy="ai", holding_days=3, confidence_threshold=45.0, base_trade_amount=1000.0, max_allocation=0.10, initial_cash=5000.0):
    raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/AAPL.csv'))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/'))
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run fetch.py first.")
        
    df = pd.read_csv(raw_path, index_col='Date', parse_dates=True)
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    seq_length = profile['seq_length']
    target_shift = profile['target_shift']
    
    # Calculate indicators
    df = add_technical_indicators(df, target_shift)
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    feature_cols = [col for col in df.columns if col not in ['Target_Return', 'SMA_20', 'SMA_50']]
    
    # Recreate test split boundaries
    n = len(df)
    val_end = int(n * (0.7 + 0.15))
    test_df = df.iloc[val_end:].copy()
    
    # Load scaler
    with open(os.path.join(processed_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    test_features = scaler.transform(test_df[feature_cols])
    test_targets = test_df['Target_Return'].values
    
    # Recreate sequences
    xs = []
    for i in range(len(test_features) - seq_length):
        xs.append(test_features[i:(i + seq_length)])
    
    X_test = np.array(xs)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # decision days
    decision_df = test_df.iloc[seq_length:].copy()
    close_prices = decision_df['Close'].values
    dates = decision_df.index
    
    # Pre-calculate crossover columns to avoid warnings
    decision_df['SMA_20_prev'] = decision_df['SMA_20'].shift(1)
    decision_df['SMA_50_prev'] = decision_df['SMA_50'].shift(1)
    
    # Load AI Model (only needed for 'ai' and 'pnl_box' strategies)
    model = None
    device = None
    if strategy in ["ai", "pnl_box"]:
        model_path, params_path = find_latest_model(ACTIVE_TIMEFRAME)
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        model = LSTMAttention(
            input_size=len(feature_cols),
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
    
    for i in range(len(dates)):
        date = dates[i]
        close_price = close_prices[i]
        
        # A. Update days held
        for pos in open_positions:
            pos["days_held"] += 1
            
        # B. Check for Strategy exits
        active_positions = []
        
        for pos in open_positions:
            qty = pos["qty"]
            buy_price = pos["buy_price"]
            days_held = pos["days_held"]
            
            pnl = qty * (close_price - buy_price)
            pnl_pct = (close_price - buy_price) / buy_price * 100.0
            
            exit_triggered = False
            exit_type = ""
            
            # PnL Box strategy check (TP at +3.0%, SL at -1.5%)
            if strategy == "pnl_box":
                if close_price >= buy_price * 1.03:
                    exit_triggered = True
                    exit_type = "TAKE PROFIT (+3.0%)"
                elif close_price <= buy_price * 0.985:
                    exit_triggered = True
                    exit_type = "STOP LOSS (-1.5%)"
                    
            # Time exit check
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

        # C. Strategy Signal Generation
        buy_signal = False
        sell_all_signal = False
        trade_confidence = 100.0 # Default for non-AI strategies

        if strategy in ["ai", "pnl_box"]:
            # Run AI prediction with 30 passes for speed
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
                sell_all_signal = True

        elif strategy == "sma":
            # SMA Crossover (Fast 20 crosses Slow 50)
            row = decision_df.iloc[i]
            # Check if SMA_20 crossed above SMA_50
            if (row['SMA_20'] > row['SMA_50']) and (row['SMA_20_prev'] <= row['SMA_50_prev']):
                buy_signal = True
            # Check if SMA_20 crossed below SMA_50
            elif (row['SMA_20'] < row['SMA_50']) and (row['SMA_20_prev'] >= row['SMA_50_prev']):
                sell_all_signal = True

        elif strategy == "rsi_bb":
            # Mean Reversion (RSI oversold & breaking Bollinger lower band)
            row = decision_df.iloc[i]
            if row['RSI'] < 30 and close_price <= row['BB_Low']:
                buy_signal = True
            elif row['RSI'] > 70 or close_price >= row['BB_High']:
                sell_all_signal = True

        # D. Execute Trade Decisions
        current_holdings_val = sum(pos["qty"] * close_price for pos in open_positions)
        portfolio_value = free_cash + current_holdings_val

        if buy_signal:
            trade_cash = base_trade_amount * (trade_confidence / 100.0)
            
            # Position sizing caps
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
                            "buy_price": close_price,
                            "days_held": 0
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

    # Calculate metrics
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
        'final_value': portfolio_history[-1]
    }

def run_backtests(strategy="all", holding_days=3, confidence_threshold=45.0):
    if strategy != "all":
        print(f"Running single backtest for strategy '{strategy}'...")
        res = run_single_backtest(strategy=strategy, holding_days=holding_days, confidence_threshold=confidence_threshold)
        print("=" * 60)
        print(f"🏁 STRATEGY RESULT: {strategy.upper()}")
        print("=" * 60)
        print(f"💰 Net Profit:  ${res['net_pnl']:+,.2f} ({res['net_pnl_pct']:+.2f}%)")
        print(f"📉 Max Drawdown: {res['max_drawdown']:.2f}%")
        print(f"🎯 Win Rate:     {res['win_rate']:.1f}%")
        print(f"🔄 Total Trades: {res['buys']} buys / {res['sells']} sells")
        print("=" * 60)
        return

    # Run all strategies comparatively
    strategies = ["ai", "pnl_box", "sma", "rsi_bb"]
    results = {}
    
    print("\n⏳ Running comparative simulations across all 4 strategies...")
    for strat in strategies:
        try:
            results[strat] = run_single_backtest(strategy=strat, holding_days=holding_days, confidence_threshold=confidence_threshold)
            print(f"   ✓ Strategy '{strat.upper()}' evaluated.")
        except Exception as e:
            print(f"   ✗ Strategy '{strat.upper()}' failed: {e}")

    # Display comparison table
    print("\n" + "=" * 80)
    print("🏆 COMPARATIVE TRADING STRATEGIES BACKTEST SUMMARY")
    print("=" * 80)
    print(f"{'STRATEGY':<15} | {'FINAL VALUE':<12} | {'NET PROFIT':<15} | {'MAX DRAWDOWN':<14} | {'WIN RATE':<10} | {'TRADES'}")
    print("-" * 80)
    
    for strat, res in results.items():
        strat_display = {
            "ai": "AI Model (Base)",
            "pnl_box": "PnL Box (3%/1.5%)",
            "sma": "SMA Crossover",
            "rsi_bb": "RSI + BB Mean"
        }.get(strat, strat)
        
        profit_str = f"${res['net_pnl']:+,.2f} ({res['net_pnl_pct']:+.2f}%)"
        print(f"{strat_display:<15} | ${res['final_value']:<11,.2f} | {profit_str:<15} | {res['max_drawdown']:<13.2f}% | {res['win_rate']:<9.1f}% | {res['buys']} B / {res['sells']} S")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-strategy historical backtester.")
    parser.add_argument("--strategy", type=str, default="all", choices=["ai", "pnl_box", "sma", "rsi_bb", "all"], help="Strategy to evaluate.")
    parser.add_argument("--holding-days", type=int, default=5, help="Holding period (time exit) in days.")
    parser.add_argument("--confidence-threshold", type=float, default=45.0, help="Model confidence threshold.")
    args = parser.parse_args()
    
    run_backtests(strategy=args.strategy, holding_days=args.holding_days, confidence_threshold=args.confidence_threshold)
