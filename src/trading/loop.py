import os
import sys
import torch
import numpy as np
import pickle
import json
import argparse
import pandas as pd
import yfinance as yf
import ta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME, ASSETS, CATEGORIES, TICKER_MAPPING, INVERSE_MAPPING
from src.models.train import LSTMAttention
from src.models.predict import find_latest_model
from src.trading.execution import get_portfolio_positions, get_account_cash, place_market_order

# ---------------------------------------------------------------------------
# 1. Trading Configurations & Risk Parameters
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 40.0       # Only trade if model confidence is >= 40%
MAX_ALLOCATION_PER_TICKER = 0.10   # Max 10% of portfolio value per ticker
BASE_TRADE_AMOUNT = 1000.0        # Base trade cash size ($1,000)

WATCH_LIST = list(ASSETS.keys())   # Watch and trade all configured ETFs and stocks



def run_trading_loop(dry_run=False):
    print("=" * 60)
    print(f"🚀 STARTING AUTOMATED TRADING LOOP ({'DRY RUN MODE' if dry_run else 'ACTIVE MODE'})")
    print(f"   Active Timeframe Profile: '{ACTIVE_TIMEFRAME}'")
    print("=" * 60)
    
    # 1. Get current portfolio states and cash for both DEMO and REAL accounts
    print("\n🔍 Fetching portfolio state from Trading 212...")
    free_cash_demo = get_account_cash(real=False)
    positions_demo = get_portfolio_positions(real=False)
    
    try:
        free_cash_real = get_account_cash(real=True)
        positions_real = get_portfolio_positions(real=True)
    except Exception as e:
        print(f"⚠️  Could not fetch REAL portfolio (Rate limit or network): {e}")
        free_cash_real = 0.0
        positions_real = []

    # Dry-run fallback for DEMO
    if free_cash_demo == 0.0 and dry_run:
        print("⚠️  DEMO Free cash returned $0.0 or failed. Simulating with $50,000.00 base balance for dry-run.")
        free_cash_demo = 50000.0

    # Calculate DEMO Invested and Value
    total_holdings_value_demo = 0.0
    pos_dict = {}
    for pos in positions_demo:
        t212_ticker = pos.get("ticker")
        ticker = INVERSE_MAPPING.get(t212_ticker, t212_ticker)
        qty = float(pos.get("quantity", 0.0))
        price = float(pos.get("currentPrice", 0.0))
        value = qty * price
        pos_dict[ticker] = {
            "quantity": qty,
            "currentPrice": price,
            "value": value
        }
        total_holdings_value_demo += value
    portfolio_value_demo = free_cash_demo + total_holdings_value_demo

    # Calculate REAL Invested and Value
    total_holdings_value_real = 0.0
    real_pos_dict = {}
    for pos in positions_real:
        t212_ticker = pos.get("ticker")
        ticker = INVERSE_MAPPING.get(t212_ticker, t212_ticker)
        qty = float(pos.get("quantity", 0.0))
        price = float(pos.get("currentPrice", 0.0))
        value = qty * price
        real_pos_dict[ticker] = {
            "quantity": qty,
            "currentPrice": price,
            "value": value
        }
        total_holdings_value_real += value
    portfolio_value_real = free_cash_real + total_holdings_value_real

    # Print beautiful comparative balances table
    print("\n💰 PORTFOLIO BALANCES SUMMARY:")
    print("┌────────────────────────┬────────────────────┬────────────────────┐")
    print("│ Metric                 │ DEMO (Paper)       │ REAL (Live)        │")
    print("├────────────────────────┼────────────────────┼────────────────────┤")
    print(f"│ Available Cash         │ ${free_cash_demo:14,.2f} │ £{free_cash_real:14,.2f} │")
    print(f"│ Invested Capital       │ ${total_holdings_value_demo:14,.2f} │ £{total_holdings_value_real:14,.2f} │")
    print(f"│ Total Portfolio Value  │ ${portfolio_value_demo:14,.2f} │ £{portfolio_value_real:14,.2f} │")
    print("└────────────────────────┴────────────────────┴────────────────────┘")

    # Print beautiful comparative open positions table
    print("\n📦 OPEN POSITIONS COMPARISON:")
    all_held_tickers = set(pos_dict.keys()).union(real_pos_dict.keys())
    if not all_held_tickers:
        print("   No open positions currently in either portfolio.")
    else:
        print("┌──────────┬──────────────────────────┬──────────────────────────┐")
        print("│ Ticker   │ DEMO Portfolio           │ REAL Portfolio           │")
        print("├──────────┬──────────────────────────┬──────────────────────────┤")
        for ticker in sorted(all_held_tickers):
            demo_str = "None"
            if ticker in pos_dict:
                d = pos_dict[ticker]
                demo_str = f"{d['quantity']:.2f} @ ${d['currentPrice']:.2f} (${d['value']:.2f})"
                
            real_str = "None"
            if ticker in real_pos_dict:
                r = real_pos_dict[ticker]
                real_str = f"{r['quantity']:.2f} @ £{r['currentPrice']:.2f} (£{r['value']:.2f})"
                
            print(f"│ {ticker:8} │ {demo_str:24} │ {real_str:24} │")
        print("└──────────┬──────────────────────────┬──────────────────────────┘")

    # Downstream execution logic operates strictly on the DEMO portfolio
    free_cash = free_cash_demo
    portfolio_value = portfolio_value_demo

    # 1.1 Load and Synchronize persistent active plans for advanced fractional sells
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/'))
    plans_path = os.path.join(processed_dir, 'active_plans.json')
    active_plans = {}
    if os.path.exists(plans_path):
        try:
            with open(plans_path, 'r') as f:
                active_plans = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading active_plans.json: {e}")
            
    # Clean up plans for tickers we no longer hold
    for tk in list(active_plans.keys()):
        if tk not in pos_dict:
            active_plans.pop(tk)
            
    # Sync and increment days_held for held positions
    for tk, pos in pos_dict.items():
        if tk not in active_plans:
            active_plans[tk] = {
                "entry_price": pos["currentPrice"],
                "initial_qty": pos["quantity"],
                "days_held": 0,
                "price_target_sold": False,
                "downside_trigger_day": None
            }
        else:
            active_plans[tk]["days_held"] += 1
            
    # 1.2 Process daily active conditional plans before running new evaluations
    print("\n⏳ Processing active fractional sell plans...")
    for tk, plan in list(active_plans.items()):
        held_qty = pos_dict.get(tk, {}).get("quantity", 0.0)
        current_price = pos_dict.get(tk, {}).get("currentPrice", 0.0)
        t212_ticker = TICKER_MAPPING.get(tk, tk)
        
        if held_qty <= 0:
            continue
            
        # Check 1: Price-Target Sell (30% when price >= entry_price + 20)
        if not plan.get("price_target_sold", False) and current_price >= plan["entry_price"] + 20.0:
            sell_qty = round(plan["initial_qty"] * 0.3, 2)
            if sell_qty > 0 and held_qty >= sell_qty:
                print(f"   🎯 PRICE TARGET TRIGGERED: {tk} rose above buy_price + 20 (${current_price:.2f})")
                print(f"   💰 Sized Order: Fractional Sell {sell_qty:.2f} shares of {t212_ticker}")
                if dry_run:
                    print(f"   📝 [DRY RUN] Simulating SELL order for {sell_qty:.2f} shares of {t212_ticker}.")
                else:
                    place_market_order(t212_ticker, -sell_qty)
                plan["price_target_sold"] = True
                pos_dict[tk]["quantity"] = round(held_qty - sell_qty, 2) # Update local quantity
                held_qty = pos_dict[tk]["quantity"]
                
        # Check 2: Time-Delayed Downside Sell (30% exactly 2 days after trigger)
        trigger_day = plan.get("downside_trigger_day")
        if trigger_day is not None and plan["days_held"] == trigger_day + 2:
            sell_qty = round(plan["initial_qty"] * 0.3, 2)
            if sell_qty > 0 and held_qty >= sell_qty:
                print(f"   📉 DOWNSIDE DELAYED TRIGGERED: {tk} downside trigger + 2 days reached (Age: {plan['days_held']} days)")
                print(f"   💰 Sized Order: Fractional Sell {sell_qty:.2f} shares of {t212_ticker}")
                if dry_run:
                    print(f"   📝 [DRY RUN] Simulating SELL order for {sell_qty:.2f} shares of {t212_ticker}.")
                else:
                    place_market_order(t212_ticker, -sell_qty)
                plan["downside_trigger_day"] = None # Clear trigger
                pos_dict[tk]["quantity"] = round(held_qty - sell_qty, 2)
                held_qty = pos_dict[tk]["quantity"]
                
        # Check 3: Final Time-Exit (5 days)
        if plan["days_held"] >= 5:
            print(f"   ⏰ TIME EXIT TRIGGERED: {tk} age reached {plan['days_held']} days. Liquidating all remaining shares.")
            print(f"   💰 Sized Order: Liquidate {held_qty:.2f} shares of {t212_ticker}")
            if dry_run:
                print(f"   📝 [DRY RUN] Simulating SELL order for {held_qty:.2f} shares of {t212_ticker}.")
            else:
                place_market_order(t212_ticker, -held_qty)
            active_plans.pop(tk)
            pos_dict.pop(tk, None)

    # 2. Find and load the latest trained model
    print("\n📦 Loading confidence-aware model...")
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    try:
        from src.models.predict import load_model_and_params, prepare_live_input, mc_dropout_predict
        model, best_params, model_path = load_model_and_params(ACTIVE_TIMEFRAME, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"   Loaded: {os.path.basename(model_path)}")
    print(f"   Params: hidden={best_params['hidden_size']}, layers={best_params['num_layers']}, dropout={best_params['dropout']:.3f}")

    # Load scaler and feature cols
    scaler_path  = os.path.join(processed_dir, f'scaler_{ACTIVE_TIMEFRAME}.pkl')
    feature_path = os.path.join(processed_dir, f'feature_cols_{ACTIVE_TIMEFRAME}.pkl')
    
    # Fallback to old paths if active profile paths don't exist
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(processed_dir, 'scaler.pkl')
    if not os.path.exists(feature_path):
        feature_path = os.path.join(processed_dir, 'feature_cols.pkl')
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    with open(feature_path, 'rb') as f:
        feature_cols = pickle.load(f)

    # 3. Evaluate watch list
    print("\n🎯 Evaluating watch list tickers...")
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    
    for ticker in WATCH_LIST:
        print("-" * 50)
        print(f"📈 Analyzing {ticker} ({ASSETS.get(ticker)})...")
        
        try:
            # Prepare scaled conditional data and predict
            input_tensor, current_price = prepare_live_input(ticker, profile, scaler, feature_cols)
            opt_ratio = best_params.get('optimal_ratio_threshold', 0.15)
            result = mc_dropout_predict(model, input_tensor, device, n_samples=30, optimal_ratio_threshold=opt_ratio)
        except Exception as e:
            print(f"❌ Failed to run predictions for {ticker}: {e}")
            continue


        predicted_return = result['predicted_return']
        confidence = result['confidence']
        
        print(f"   Current Price:    ${current_price:.2f}")
        print(f"   Predicted Return: {predicted_return:+.4f}%")
        print(f"   Model Confidence: {confidence:.1f}%")

        # Get the Trading 212 specific ticker symbol
        t212_ticker = TICKER_MAPPING.get(ticker, ticker)

        # 4. Action Logic
        if predicted_return > 0 and confidence >= CONFIDENCE_THRESHOLD:
            # --- BUY SIGNAL ---
            print(f"   📢 BUY Signal identified (Confidence: {confidence:.1f}%)")
            
            # Sizing: base trade size weighted by confidence percentage
            trade_cash = BASE_TRADE_AMOUNT * (confidence / 100.0)
            
            # Risk Management: Max allocation limit per ticker
            max_allowed_val = portfolio_value * MAX_ALLOCATION_PER_TICKER
            current_val = pos_dict.get(ticker, {}).get("value", 0.0)
            
            remaining_allowable = max_allowed_val - current_val
            if remaining_allowable <= 0:
                print(f"   ⚠️  Skipping BUY: Current position in {ticker} (${current_val:.2f}) "
                      f"already exceeds max allocation ({MAX_ALLOCATION_PER_TICKER*100:.0f}% of portfolio).")
                continue
            
            if trade_cash > remaining_allowable:
                print(f"   ⚠️  Trimming trade size from ${trade_cash:.2f} to ${remaining_allowable:.2f} "
                      f"to respect maximum single ticker allocation limits.")
                trade_cash = remaining_allowable

            if free_cash < trade_cash:
                print(f"   ⚠️  Trimming trade size from ${trade_cash:.2f} to available free cash ${free_cash:.2f}")
                trade_cash = free_cash

            if trade_cash <= 0:
                print("   ⚠️  Insufficient cash to execute trade.")
                continue

            # Calculate trade quantity and round to 2 decimal places for T212 precision
            qty = trade_cash / current_price
            qty = round(qty, 2)
            
            if qty > 0:
                print(f"   💰 Sized Order: Buy {qty:.2f} shares of {t212_ticker} (~${trade_cash:.2f})")
                if dry_run:
                    print(f"   📝 [DRY RUN] Simulating BUY order for {qty:.2f} shares of {t212_ticker}.")
                else:
                    place_market_order(t212_ticker, qty)
                
                # Add to persistent active plans
                active_plans[ticker] = {
                    "entry_price": current_price,
                    "initial_qty": qty,
                    "days_held": 0,
                    "price_target_sold": False,
                    "downside_trigger_day": None
                }
            else:
                print("   ⚠️  Sized quantity rounded to 0.0 shares. Skipping.")

        elif predicted_return < 0 and confidence >= CONFIDENCE_THRESHOLD:
            # --- SELL / DOWNSIDE SIGNAL ---
            print(f"   📢 SELL/DOWNSIDE Signal identified (Confidence: {confidence:.1f}%)")
            
            # If high-confidence downside warning (>= 65%), trigger the delayed fractional plan instead of selling everything
            if confidence >= 65.0 and ticker in active_plans:
                plan = active_plans[ticker]
                if plan.get("downside_trigger_day") is None:
                    plan["downside_trigger_day"] = plan["days_held"]
                    print(f"   📉 Registered Downside Sell Plan: Sell 30% of {ticker} holdings 2 days from now.")
            else:
                # Normal liquidation signal
                held_qty = pos_dict.get(ticker, {}).get("quantity", 0.0)
                if held_qty > 0:
                    print(f"   💰 Sized Order: Liquidate all {held_qty:.2f} shares of {t212_ticker}")
                    if dry_run:
                        print(f"   📝 [DRY RUN] Simulating SELL order for {held_qty:.2f} shares of {t212_ticker}.")
                    else:
                        place_market_order(t212_ticker, -held_qty)
                    active_plans.pop(ticker, None)
                else:
                    print(f"   ℹ️  No open positions currently held for {t212_ticker}. Skipping short-sell.")

        else:
            print(f"   💤 Signal weak or below confidence threshold ({confidence:.1f}% < {CONFIDENCE_THRESHOLD}%). No action taken.")

    # 5. Persist all updated active plans to active_plans.json
    try:
        with open(plans_path, 'w') as f:
            json.dump(active_plans, f, indent=4)
        print("\n📝 Active plans successfully saved to active_plans.json.")
    except Exception as e:
        print(f"❌ Failed to save active plans: {e}")

    print("\n" + "=" * 60)
    print("🏁 AUTOMATED TRADING LOOP EXECUTION COMPLETED")
    print("=" * 60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence-aware automated Trading 212 bot.")
    parser.add_argument("--dry-run", action="store_true", help="Run the bot in mock/simulation mode without placing actual orders.")
    args = parser.parse_args()
    
    run_trading_loop(dry_run=args.dry_run)
