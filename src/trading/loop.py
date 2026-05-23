import os
import sys
import torch
import numpy as np
import pickle
import json
import argparse
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

def prepare_live_input_conditional(ticker, profile, scaler):
    """
    Fetches the latest data with dividends, adds indicators,
    scales numeric features, appends category one-hot flags, and returns sequence tensor.
    """
    interval = profile['interval']
    seq_length = profile['seq_length']
    
    # Fetch enough extra data for technical indicators
    buffer = max(60, seq_length + 40)
    data = yf.download(ticker, period=f"{buffer * 2}d" if interval == "1d" else "max", interval=interval, actions=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    current_price = data['Close'].iloc[-1]
    
    df = data.copy()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    
    # Handle dividends
    if 'Dividends' not in df.columns:
        df['Dividends'] = 0.0
    else:
        df['Dividends'] = df['Dividends'].fillna(0.0)
        
    numeric_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'Dividends']
    df.dropna(subset=numeric_cols, inplace=True)
    
    # Scale numeric indicators
    scaled_num = scaler.transform(df[numeric_cols])
    
    # Append one-hot category columns
    category = ASSETS.get(ticker)
    one_hot = [1.0 if c == category else 0.0 for c in CATEGORIES]
    one_hot_array = np.tile(one_hot, (len(scaled_num), 1))
    
    # Stack features: shape (N, 11)
    combined_feats = np.hstack([scaled_num, one_hot_array])
    
    # Extract sequence window
    input_seq = combined_feats[-seq_length:]
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
    
    return input_tensor, current_price

def run_trading_loop(dry_run=False):
    print("=" * 60)
    print(f"🚀 STARTING AUTOMATED TRADING LOOP ({'DRY RUN MODE' if dry_run else 'ACTIVE MODE'})")
    print(f"   Active Timeframe Profile: '{ACTIVE_TIMEFRAME}'")
    print("=" * 60)

    # 1. Get current portfolio state and cash
    print("\n🔍 Fetching portfolio state from Trading 212...")
    free_cash = get_account_cash()
    positions = get_portfolio_positions()

    if free_cash == 0.0 and dry_run:
        print("⚠️  Free cash returned $0.0 or failed. Simulating with $50,000.00 base balance for dry-run.")
        free_cash = 50000.0

    print(f"💰 Available Cash: ${free_cash:,.2f}")
    
    # Process portfolio holdings
    pos_dict = {}
    total_holdings_value = 0.0
    print("\n📦 Open Positions:")
    if not positions:
        print("   No open positions currently.")
    else:
        for pos in positions:
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
            total_holdings_value += value
            print(f"   • {ticker} ({t212_ticker}): {qty:.2f} shares @ ${price:.2f} (Current Value: ${value:.2f})")

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

    # Load scaler
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/'))
    with open(os.path.join(processed_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # Initialize PyTorch device and model
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # 8 numeric features + 3 category one-hot features = 11 feature inputs
    model = LSTMAttention(
        input_size=11,
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
        print(f"📈 Analyzing {ticker} ({ASSETS.get(ticker)})...")
        
        try:
            # Prepare scaled conditional data and predict
            input_tensor, current_price = prepare_live_input_conditional(ticker, profile, scaler)
            result = mc_dropout_predict_local(model, input_tensor, device)
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
            else:
                print("   ⚠️  Sized quantity rounded to 0.0 shares. Skipping.")

        elif predicted_return < 0 and confidence >= CONFIDENCE_THRESHOLD:
            # --- SELL SIGNAL ---
            print(f"   📢 SELL Signal identified (Confidence: {confidence:.1f}%)")
            
            # Verify if we currently hold this ticker
            held_qty = pos_dict.get(ticker, {}).get("quantity", 0.0)
            if held_qty > 0:
                print(f"   💰 Sized Order: Liquidate all {held_qty:.2f} shares of {t212_ticker}")
                if dry_run:
                    print(f"   📝 [DRY RUN] Simulating SELL order for {held_qty:.2f} shares of {t212_ticker}.")
                else:
                    # Trading 212 API places sells by passing a negative quantity
                    place_market_order(t212_ticker, -held_qty)
            else:
                print(f"   ℹ️  No open positions currently held for {t212_ticker}. Skipping short-sell.")

        else:
            print(f"   💤 Signal weak or below confidence threshold ({confidence:.1f}% < {CONFIDENCE_THRESHOLD}%). No action taken.")

    print("\n" + "=" * 60)
    print("🏁 AUTOMATED TRADING LOOP EXECUTION COMPLETED")
    print("=" * 60)

def mc_dropout_predict_local(model, input_tensor, device, n_samples=30):
    """Local helper to run fast MC Dropout predictions on 11 feature sequence."""
    model.train() # Keep dropout ON
    all_mu = []
    all_sigma = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            mu, log_sigma = model(input_tensor.to(device))
            all_mu.append(mu.cpu().item())
            all_sigma.append(torch.exp(log_sigma).cpu().item())
            
    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    
    pred_return = np.mean(all_mu)
    mc_std = np.std(all_mu)
    model_sigma = np.mean(all_sigma)
    total_unc = np.sqrt(mc_std**2 + model_sigma**2)
    confidence = max(0, min(100, 100 * np.exp(-total_unc * 0.5)))
    
    return {
        'predicted_return': pred_return,
        'confidence': confidence
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence-aware automated Trading 212 bot.")
    parser.add_argument("--dry-run", action="store_true", help="Run the bot in mock/simulation mode without placing actual orders.")
    args = parser.parse_args()
    
    run_trading_loop(dry_run=args.dry_run)
