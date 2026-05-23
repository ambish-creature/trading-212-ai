import os
import sys
import torch
import numpy as np
import pickle
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME
from src.models.train import LSTMAttention
from src.models.predict import prepare_live_input, mc_dropout_predict, find_latest_model
from src.trading.execution import get_portfolio_positions, get_account_cash, place_market_order

# ---------------------------------------------------------------------------
# 1. Trading Configurations & Risk Parameters
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 45.0       # Only trade if model confidence is >= 45%
MAX_ALLOCATION_PER_TICKER = 0.10   # Max 10% of portfolio value per ticker
BASE_TRADE_AMOUNT = 1000.0        # Base trade cash size ($1,000)

WATCH_LIST = ["AAPL"]             # Tickers to watch and trade

def run_trading_loop(dry_run=False):
    print("=" * 60)
    print(f"🚀 STARTING AUTOMATED TRADING LOOP ({'DRY RUN MODE' if dry_run else 'ACTIVE MODE'})")
    print(f"   Active Timeframe Profile: '{ACTIVE_TIMEFRAME}'")
    print("=" * 60)

    # 1. Get current portfolio state and cash
    print("\n🔍 Fetching portfolio state from Trading 212...")
    free_cash = get_account_cash()
    positions = get_portfolio_positions()

    # Dry-run fallback: If account summary fails or demo cash is 0, simulate with a standard $10,000 balance
    if free_cash == 0.0 and dry_run:
        print("⚠️  Free cash returned $0.0 or failed. Simulating with $10,000.00 base balance for dry-run.")
        free_cash = 10000.0

    print(f"💰 Available Cash: ${free_cash:,.2f}")
    
    # Process portfolio holdings
    pos_dict = {}
    total_holdings_value = 0.0
    print("\n📦 Open Positions:")
    if not positions:
        print("   No open positions currently.")
    else:
        for pos in positions:
            ticker = pos.get("ticker")
            qty = float(pos.get("quantity", 0.0))
            price = float(pos.get("currentPrice", 0.0))
            value = qty * price
            pos_dict[ticker] = {
                "quantity": qty,
                "currentPrice": price,
                "value": value
            }
            total_holdings_value += value
            print(f"   • {ticker}: {qty:.4f} shares @ ${price:.2f} (Current Value: ${value:.2f})")

    portfolio_value = free_cash + total_holdings_value
    print(f"📊 Total Portfolio Value: ${portfolio_value:,.2f}")

    # 2. Find and load the latest trained model
    print("\n📦 Loading confidence-aware model...")
    try:
        model_path, params_path = find_latest_model(ACTIVE_TIMEFRAME)
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        return

    with open(params_path, 'r') as f:
        best_params = json.load(f)
    print(f"   Loaded: {os.path.basename(model_path)}")
    print(f"   Params: hidden={best_params['hidden_size']}, layers={best_params['num_layers']}, dropout={best_params['dropout']:.3f}")

    # Load scaler and features
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/'))
    with open(os.path.join(processed_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(processed_dir, 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)

    # Initialize PyTorch device and model
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model = LSTMAttention(
        input_size=len(feature_cols),
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # 3. Evaluate watch list
    print("\n🎯 Evaluating watch list tickers...")
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    
    for ticker in WATCH_LIST:
        print("-" * 50)
        print(f"📈 Analyzing {ticker}...")
        
        try:
            # Prepare data and predict
            input_tensor, current_price = prepare_live_input(ticker, profile, scaler, feature_cols)
            result = mc_dropout_predict(model, input_tensor, device, n_samples=100)
        except Exception as e:
            print(f"❌ Failed to run predictions for {ticker}: {e}")
            continue

        predicted_return = result['predicted_return']
        confidence = result['confidence']
        
        print(f"   Current Price:    ${current_price:.2f}")
        print(f"   Predicted Return: {predicted_return:+.4f}%")
        print(f"   Model Confidence: {confidence:.1f}%")

        # 4. Action Logic
        if predicted_return > 0 and confidence >= CONFIDENCE_THRESHOLD:
            # --- BUY SIGNAL ---
            print(f"   📢 BUY Signal identified (Confidence: {confidence:.1f}%)")
            
            # Base sizing: base trade size weighted by confidence percentage
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

            # Calculate trade quantity
            qty = trade_cash / current_price
            
            print(f"   💰 Sized Order: Buy {qty:.4f} shares (~${trade_cash:.2f})")
            if dry_run:
                print(f"   📝 [DRY RUN] Simulating BUY order for {qty:.4f} shares of {ticker}.")
            else:
                place_market_order(ticker, qty)

        elif predicted_return < 0 and confidence >= CONFIDENCE_THRESHOLD:
            # --- SELL SIGNAL ---
            print(f"   📢 SELL Signal identified (Confidence: {confidence:.1f}%)")
            
            # Verify if we currently hold this ticker
            held_qty = pos_dict.get(ticker, {}).get("quantity", 0.0)
            if held_qty > 0:
                print(f"   💰 Sized Order: Liquidate all {held_qty:.4f} shares")
                if dry_run:
                    print(f"   📝 [DRY RUN] Simulating SELL order for {held_qty:.4f} shares of {ticker}.")
                else:
                    # Trading 212 API places sells by passing a negative quantity
                    place_market_order(ticker, -held_qty)
            else:
                print("   ℹ️  No open positions currently held. Skipping short-sell.")

        else:
            print(f"   💤 Signal weak or below confidence threshold ({confidence:.1f}% < {CONFIDENCE_THRESHOLD}%). No action taken.")

    print("\n" + "=" * 60)
    print("🏁 AUTOMATED TRADING LOOP EXECUTION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence-aware automated Trading 212 bot.")
    parser.add_argument("--dry-run", action="store_true", help="Run the bot in mock/simulation mode without placing actual orders.")
    args = parser.parse_args()
    
    run_trading_loop(dry_run=args.dry_run)
