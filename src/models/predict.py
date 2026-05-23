import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import json
import glob
import yfinance as yf
import ta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME
from src.models.train import LSTMAttention

# ---------------------------------------------------------------------------
# 1. Data Preparation for Live Prediction
# ---------------------------------------------------------------------------

def prepare_live_input(ticker, profile, scaler, feature_cols):
    """
    Fetches the latest data for a ticker, adds indicators,
    scales using the training scaler, and returns the input tensor
    plus the current price.
    """
    interval = profile['interval']
    seq_length = profile['seq_length']
    
    # Fetch enough extra data to calculate indicators (need ~30 extra bars for RSI/BB)
    buffer = max(60, seq_length + 40)
    data = yf.download(ticker, period=f"{buffer * 2}d" if interval == "1d" else "max", interval=interval)
    
    # Flatten MultiIndex columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Store the current price before processing
    current_price = data['Close'].iloc[-1]
    
    # Add technical indicators (no target needed for prediction)
    df = data.copy()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df.dropna(inplace=True)
    
    # Select only the feature columns used during training
    df = df[feature_cols]
    
    # Scale using the saved training scaler
    scaled = scaler.transform(df)
    
    # Take the last seq_length rows as our input window
    input_seq = scaled[-seq_length:]
    
    # Convert to tensor: shape (1, seq_length, num_features)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
    
    return input_tensor, current_price

# ---------------------------------------------------------------------------
# 2. Monte Carlo Dropout Prediction
# ---------------------------------------------------------------------------

def mc_dropout_predict(model, input_tensor, device, n_samples=100):
    """
    Runs Monte Carlo Dropout: performs n_samples forward passes
    with dropout ENABLED to get a distribution of predictions.
    
    Combined with the model's own learned uncertainty (sigma head),
    this gives us a robust confidence estimate.
    """
    model.train()  # Keep dropout ON during inference
    
    all_mu = []
    all_sigma = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            mu, log_sigma = model(input_tensor.to(device))
            sigma = torch.exp(log_sigma)  # Convert log_sigma to sigma
            all_mu.append(mu.cpu().item())
            all_sigma.append(sigma.cpu().item())
    
    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    
    # --- Aggregate Results ---
    
    # Predicted return: mean of all MC samples
    predicted_return = np.mean(all_mu)
    
    # MC Dropout spread (epistemic uncertainty - what the model doesn't know)
    mc_std = np.std(all_mu)
    
    # Model's learned uncertainty (aleatoric uncertainty - inherent data noise)
    avg_model_sigma = np.mean(all_sigma)
    
    # Combined total uncertainty
    total_uncertainty = np.sqrt(mc_std**2 + avg_model_sigma**2)
    
    # Confidence score (0-100%)
    # We map uncertainty to confidence using an exponential decay
    # Lower uncertainty = higher confidence
    confidence = max(0, min(100, 100 * np.exp(-total_uncertainty * 50)))
    
    # Error range (10th-90th percentile from MC samples)
    lower_bound = np.percentile(all_mu, 10)
    upper_bound = np.percentile(all_mu, 90)
    
    return {
        'predicted_return': predicted_return,
        'confidence': confidence,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'mc_std': mc_std,
        'model_sigma': avg_model_sigma,
        'total_uncertainty': total_uncertainty,
    }

# ---------------------------------------------------------------------------
# 3. Find and Load the Latest Saved Model
# ---------------------------------------------------------------------------

def find_latest_model(timeframe):
    """Finds the most recently saved model for the given timeframe."""
    save_dir = os.path.join(os.path.dirname(__file__), '../../models/saved/')
    
    # Find all model files matching this timeframe
    pattern = os.path.join(save_dir, f'model_{timeframe}_*.pt')
    model_files = sorted(glob.glob(pattern))
    
    if not model_files:
        raise FileNotFoundError(f"No saved models found for timeframe '{timeframe}' in {save_dir}")
    
    latest_model = model_files[-1]
    latest_params = latest_model.replace('.pt', '_params.json')
    
    return latest_model, latest_params

# ---------------------------------------------------------------------------
# 4. Main Prediction Pipeline
# ---------------------------------------------------------------------------

def predict(ticker="AAPL"):
    """Full prediction pipeline: load model -> fetch data -> MC Dropout -> report."""
    
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    
    print(f"🔮 Predicting '{ACTIVE_TIMEFRAME}' for {ticker}...")
    print(f"   Using {profile['seq_length']} bars of {profile['interval']} data\n")
    
    # 1. Find the latest saved model
    model_path, params_path = find_latest_model(ACTIVE_TIMEFRAME)
    
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    
    print(f"📦 Loaded model: {os.path.basename(model_path)}")
    print(f"   Params: hidden={best_params['hidden_size']}, layers={best_params['num_layers']}, "
          f"dropout={best_params['dropout']:.3f}, lr={best_params['lr']:.6f}\n")
    
    # 2. Load the scaler and feature columns from preprocessing
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/processed/')
    
    with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(data_dir, 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    
    # 3. Prepare live input data
    input_tensor, current_price = prepare_live_input(ticker, profile, scaler, feature_cols)
    input_size = input_tensor.shape[2]
    
    # 4. Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    model = LSTMAttention(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # 5. Run Monte Carlo Dropout prediction
    result = mc_dropout_predict(model, input_tensor, device, n_samples=100)
    
    # 6. Convert returns to prices
    predicted_price = current_price * (1 + result['predicted_return'])
    price_lower = current_price * (1 + result['lower_bound'])
    price_upper = current_price * (1 + result['upper_bound'])
    
    # 7. Display results
    print("=" * 55)
    print(f"  📊 PREDICTION REPORT — {ticker} ({ACTIVE_TIMEFRAME})")
    print("=" * 55)
    print(f"  Current Price:       ${current_price:.2f}")
    print(f"  Predicted Price:     ${predicted_price:.2f}")
    print(f"  Predicted Return:    {result['predicted_return']*100:+.4f}%")
    print(f"  Confidence:          {result['confidence']:.1f}%")
    print(f"  Price Range (80%):   ${price_lower:.2f} — ${price_upper:.2f}")
    print("-" * 55)
    print(f"  MC Dropout Std:      {result['mc_std']*100:.4f}%")
    print(f"  Model Uncertainty:   {result['model_sigma']*100:.4f}%")
    print(f"  Total Uncertainty:   {result['total_uncertainty']*100:.4f}%")
    print("=" * 55)
    
    direction = "📈 BUY signal" if result['predicted_return'] > 0 else "📉 SELL signal"
    strength = "STRONG" if result['confidence'] > 70 else "MODERATE" if result['confidence'] > 40 else "WEAK"
    print(f"\n  → {direction} ({strength}, {result['confidence']:.0f}% confidence)\n")
    
    return result

if __name__ == "__main__":
    predict("AAPL")
