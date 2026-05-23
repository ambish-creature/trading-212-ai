import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import json
import glob
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME
from src.models.train import LSTMAttention
from src.models.predict import find_latest_model
from src.data.preprocess import add_technical_indicators

# ---------------------------------------------------------------------------
# 1. Backtesting Engine
# ---------------------------------------------------------------------------

def run_backtest(holding_days=3, confidence_threshold=45.0, base_trade_amount=1000.0, max_allocation=0.10, initial_cash=5000.0):
    print("=" * 60)
    print(f"📊 STARTING HISTORICAL BACKTEST (AAPL)")
    print(f"   Holding Period: {holding_days} trading days")
    print(f"   Confidence Threshold: {confidence_threshold}%")
    print(f"   Base Trade Cash: ${base_trade_amount:,.2f}")
    print(f"   Max Ticker Allocation: {max_allocation*100:.0f}%")
    print(f"   Initial Portfolio Cash: ${initial_cash:,.2f}")
    print("=" * 60)

    # 1. Load raw data and recreate the test split
    raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/AAPL.csv'))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/'))
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run fetch.py first.")
        
    df = pd.read_csv(raw_path, index_col='Date', parse_dates=True)
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    seq_length = profile['seq_length']
    target_shift = profile['target_shift']
    
    # Calculate indicators and target
    df = add_technical_indicators(df, target_shift)
    feature_cols = [col for col in df.columns if col != 'Target_Return']
    
    # Recreate test split boundaries (70% Train, 15% Val, 15% Test)
    n = len(df)
    val_end = int(n * (0.7 + 0.15))
    test_df = df.iloc[val_end:]
    
    # Load scaler and features
    with open(os.path.join(processed_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    test_features = scaler.transform(test_df[feature_cols])
    test_targets = test_df['Target_Return'].values
    
    # Recreate sliding window sequences
    xs, ys = [], []
    for i in range(len(test_features) - seq_length):
        xs.append(test_features[i:(i + seq_length)])
        ys.append(test_targets[i + seq_length - 1])
    
    X_test = np.array(xs)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # decision_days holds dates and close prices corresponding to the decision day of each sequence
    decision_df = test_df.iloc[seq_length:]
    close_prices = decision_df['Close'].values
    dates = decision_df.index
    
    print(f"\n📂 Loaded test split from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"   Total test steps (trading days): {len(dates)}")

    # 2. Load latest model
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
    model.train() # Keep dropout active for MC Dropout

    # 3. Simulate day-by-day trading
    free_cash = initial_cash
    open_positions = [] # list of dicts: {"qty": float, "buy_price": float, "days_held": int}
    
    portfolio_history = []
    cash_history = []
    trade_log = []
    
    print("\n⏳ Simulating trades chronologically...")
    
    for i in range(len(dates)):
        date = dates[i]
        close_price = close_prices[i]
        
        # A. Update days held for existing positions
        for pos in open_positions:
            pos["days_held"] += 1
            
        # B. Check for HOLDING PERIOD exits (e.g. sell after 3 days)
        liquidated_cash = 0.0
        active_positions = []
        
        for pos in open_positions:
            if pos["days_held"] >= holding_days:
                # Sell position
                qty = pos["qty"]
                sell_val = qty * close_price
                free_cash += sell_val
                pnl = qty * (close_price - pos["buy_price"])
                pnl_pct = (close_price - pos["buy_price"]) / pos["buy_price"] * 100.0
                
                trade_log.append({
                    "ticker": "AAPL",
                    "type": "SELL (TIME EXIT)",
                    "date": date.strftime('%Y-%m-%d'),
                    "qty": qty,
                    "price": close_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct
                })
            else:
                active_positions.append(pos)
                
        open_positions = active_positions

        # C. Generate Model Prediction (MC Dropout - 50 passes for speed)
        input_tensor = X_test_tensor[i].unsqueeze(0).to(device)
        all_mu = []
        all_sigma = []
        
        with torch.no_grad():
            for _ in range(50):
                mu, log_sigma = model(input_tensor)
                all_mu.append(mu.cpu().item())
                all_sigma.append(torch.exp(log_sigma).cpu().item())
                
        pred_return = np.mean(all_mu)
        mc_std = np.std(all_mu)
        model_sigma = np.mean(all_sigma)
        total_unc = np.sqrt(mc_std**2 + model_sigma**2)
        confidence = max(0, min(100, 100 * np.exp(-total_unc * 0.5)))

        # D. Current portfolio values
        current_holdings_val = sum(pos["qty"] * close_price for pos in open_positions)
        portfolio_value = free_cash + current_holdings_val

        # E. Process signals
        if pred_return > 0 and confidence >= confidence_threshold:
            # --- BUY SIGNAL ---
            # Sizing: base trade size weighted by confidence percentage
            trade_cash = base_trade_amount * (confidence / 100.0)
            
            # Risk Cap: Max 10% of portfolio in this stock
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
                            "ticker": "AAPL",
                            "type": "BUY",
                            "date": date.strftime('%Y-%m-%d'),
                            "qty": qty,
                            "price": close_price,
                            "pnl": 0.0,
                            "pnl_pct": 0.0
                        })
                        
        elif pred_return < 0 and confidence >= confidence_threshold:
            # --- SELL SIGNAL (Early Exit) ---
            if open_positions:
                # Liquidate all open positions early to preserve capital
                for pos in open_positions:
                    qty = pos["qty"]
                    sell_val = qty * close_price
                    free_cash += sell_val
                    pnl = qty * (close_price - pos["buy_price"])
                    pnl_pct = (close_price - pos["buy_price"]) / pos["buy_price"] * 100.0
                    
                    trade_log.append({
                        "ticker": "AAPL",
                        "type": "SELL (EARLY SIGNAL)",
                        "date": date.strftime('%Y-%m-%d'),
                        "qty": qty,
                        "price": close_price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct
                    })
                open_positions = []

        # Recalculate daily portfolio value after orders
        current_holdings_val = sum(pos["qty"] * close_price for pos in open_positions)
        portfolio_value = free_cash + current_holdings_val
        
        portfolio_history.append(portfolio_value)
        cash_history.append(free_cash)

    # 4. Generate Backtest Metrics
    portfolio_history = np.array(portfolio_history)
    net_pnl = portfolio_history[-1] - initial_cash
    net_pnl_pct = (portfolio_history[-1] - initial_cash) / initial_cash * 100.0
    
    # Calculate Drawdown
    peaks = np.maximum.accumulate(portfolio_history)
    drawdowns = (portfolio_history - peaks) / peaks * 100.0
    max_drawdown = np.min(drawdowns)
    
    # Analyze trades
    buys = [t for t in trade_log if t["type"] == "BUY"]
    sells = [t for t in trade_log if "SELL" in t["type"]]
    
    winning_trades = [s for s in sells if s["pnl"] > 0]
    win_rate = (len(winning_trades) / len(sells) * 100.0) if sells else 0.0

    print("\n" + "=" * 60)
    print("🏁 BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"💵 Initial Portfolio Cash: ${initial_cash:,.2f}")
    print(f"📈 Final Portfolio Value:  ${portfolio_history[-1]:,.2f}")
    print(f"💰 Net Profit/Loss:        ${net_pnl:+,.2f} ({net_pnl_pct:+.2f}%)")
    print(f"📉 Maximum Drawdown:       {max_drawdown:.2f}%")
    print("-" * 60)
    print(f"🔄 Total Actions Logged:   {len(trade_log)}")
    print(f"   • BUY Trades placed:    {len(buys)}")
    print(f"   • SELL Exits executed:  {len(sells)}")
    print(f"🏆 Winning Trade Exits:    {len(winning_trades)}")
    print(f"🎯 Strategy Win Rate:      {win_rate:.1f}%")
    print("=" * 60)
    
    # Print sample of trades
    if sells:
        print("\n📝 Sample Trade Exits Log (Last 5 Sells):")
        for s in sells[-5:]:
            print(f"   • {s['date']} | Sell {s['qty']:.2f} shares @ ${s['price']:.2f} | PnL: ${s['pnl']:+,.2f} ({s['pnl_pct']:+.2f}%) [{s['type']}]")
            
    return {
        'net_pnl': net_pnl,
        'net_pnl_pct': net_pnl_pct,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_buys': len(buys),
        'total_sells': len(sells)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence-aware historical backtester.")
    parser.add_argument("--holding-days", type=int, default=3, help="Number of trading days to hold a position before exit.")
    parser.add_argument("--confidence-threshold", type=float, default=45.0, help="Confidence threshold to trigger orders.")
    args = parser.parse_args()
    
    run_backtest(holding_days=args.holding_days, confidence_threshold=args.confidence_threshold)
