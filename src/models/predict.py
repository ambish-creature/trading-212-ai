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
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME, ASSETS
from src.models.train import LSTMAttention

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


def prepare_live_input(ticker, profile, scaler, feature_cols):
    """
    Fetches the latest data for a ticker, adds indicators,
    scales using the training scaler, and returns the input tensor
    plus the current price.
    """
    interval   = profile['interval']
    seq_length = profile['seq_length']

    # Fetch enough extra data to calculate indicators (need ~60 extra bars)
    buffer = max(100, seq_length + 60)
    data = yf.download(
        ticker,
        period=f"{buffer * 2}d" if interval == "1d" else "max",
        interval=interval,
        progress=False,
        actions=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    current_price = float(data['Close'].iloc[-1])

    # Technical indicators
    close = data['Close']
    df = data.copy()

    # Denoise close price causally using Hull Moving Average (lag-free)
    denoised = hull_moving_average(close, window=14).fillna(close)

    df['SMA_20']  = denoised.rolling(window=20).mean()
    df['SMA_50']  = denoised.rolling(window=50).mean()
    df['EMA_12']  = denoised.ewm(span=12, adjust=False).mean()
    df['EMA_26']  = denoised.ewm(span=26, adjust=False).mean()
    df['RSI']     = ta.momentum.RSIIndicator(close=denoised, window=14).rsi()

    macd_ind          = ta.trend.MACD(close=denoised)
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
    df['Bid_Ask_Proxy'] = (df['High'] - df['Low']) / (close + 1e-9)

    # --- v2.0 features (must match preprocess.py and advisor.py) ---
    df['Momentum_5']  = denoised.pct_change(periods=5)  * 100.0
    df['Momentum_10'] = denoised.pct_change(periods=10) * 100.0
    df['Momentum_20'] = denoised.pct_change(periods=20) * 100.0

    # OBV z-scored over 50-day rolling window
    obv = (np.sign(denoised.diff()) * df['Volume']).fillna(0).cumsum()
    obv_mean = obv.rolling(50).mean()
    obv_std  = obv.rolling(50).std().replace(0, 1e-9)
    df['OBV'] = (obv - obv_mean) / obv_std

    df['Vol_Regime']    = df['ATR_14'] / (denoised + 1e-9)
    df['Trend_Strength'] = (df['SMA_20'] / (df['SMA_50'] + 1e-9)) - 1.0

    highest_high = df['High'].rolling(14).max()
    lowest_low   = df['Low'].rolling(14).min()
    df['Williams_R']   = -100 * (highest_high - denoised) / (highest_high - lowest_low + 1e-9)
    df['Price_vs_SMA50'] = (denoised / (df['SMA_50'] + 1e-9)) - 1.0

    # --- v3.0 Cross-Asset Sector & Macro correlation features ---
    sector_ticker = SECTOR_MAP.get(ticker, "SPY")

    try:
        support_data = yf.download(
            [sector_ticker, "SPY", "GC=F", "CL=F"],
            period=f"{buffer * 2}d" if interval == "1d" else "max",
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
        
        # Align causally to df
        df['Sector_Return_5d'] = sec_ret_5.reindex(df.index, method='ffill').fillna(0.0)
        df['Sector_Return_20d'] = sec_ret_20.reindex(df.index, method='ffill').fillna(0.0)
        df['Gold_Return_5d'] = gold_ret_5.reindex(df.index, method='ffill').fillna(0.0)
        df['Oil_Return_5d'] = oil_ret_5.reindex(df.index, method='ffill').fillna(0.0)
        df['Safe_Haven_Ratio'] = safe_haven.reindex(df.index, method='ffill').fillna(0.0)
        df['Equity_Gold_Corr'] = eq_gold_corr.reindex(df.index, method='ffill').fillna(0.0)
        df['Equity_Oil_Corr'] = eq_oil_corr.reindex(df.index, method='ffill').fillna(0.0)
        
    except Exception as ex:
        print(f"   ⚠️  Could not fetch/compute support indices: {ex}")
        for col in ['Sector_Return_5d', 'Sector_Return_20d', 'Gold_Return_5d', 
                    'Oil_Return_5d', 'Safe_Haven_Ratio', 'Equity_Gold_Corr', 'Equity_Oil_Corr']:
            df[col] = 0.0

    df['Relative_Strength_5d'] = df['Momentum_5'] - df['Sector_Return_5d']
    df['Relative_Strength_20d'] = df['Momentum_20'] - df['Sector_Return_20d']

    if 'Dividends' not in df.columns:
        df['Dividends'] = 0.0
    else:
        df['Dividends'] = df['Dividends'].fillna(0.0)

    for col in ['Open', 'High', 'Low']:
        if col not in df.columns:
            df[col] = df['Close']

    df.dropna(inplace=True)

    # Split into numeric (scaled) and one-hot categorical (unscaled)
    numeric_feature_cols = [c for c in feature_cols if not c.startswith('Category_')]
    one_hot_cols         = [c for c in feature_cols if c.startswith('Category_')]

    # Fill missing columns for compatibility
    for c in feature_cols:
        if c not in df.columns:
            if c.startswith('Category_'):
                category = ASSETS.get(ticker.upper(), "Unknown")
                cat_name = c.split('Category_')[1]
                df[c] = 1.0 if cat_name == category else 0.0
            else:
                df[c] = 0.0

    df[numeric_feature_cols] = df[numeric_feature_cols].fillna(0.0)
    df.dropna(subset=['RSI', 'SMA_50', 'ATR_14'], inplace=True)

    if len(df) < seq_length:
        raise ValueError(f"After indicator warmup, only {len(df)} rows remain (need {seq_length}).")

    scaled_num = scaler.transform(df[numeric_feature_cols].values)
    one_hot    = df[one_hot_cols].values
    all_feats  = np.hstack([scaled_num, one_hot])

    input_seq = all_feats[-seq_length:]
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

    return input_tensor, current_price


# ---------------------------------------------------------------------------
# 2. Monte Carlo Dropout Prediction — FIXED CONFIDENCE MAPPING
# ---------------------------------------------------------------------------

def mc_dropout_predict(model, input_tensor, device, n_samples=100, optimal_ratio_threshold=0.15):
    """
    Runs Monte Carlo Dropout: performs n_samples forward passes
    with dropout ENABLED to get a distribution of predictions.

    Confidence is computed using a calibrated sigmoid over the ratio:
        ratio = mc_std / model_sigma  (dimensionless uncertainty)

    The mapping is:
        - ratio <= optimal_ratio_threshold  → confidence ≥ 75%  (high-confidence zone)
        - ratio = 0                         → confidence = 100%
        - ratio = optimal_ratio_threshold   → confidence = 75%  (calibration boundary)
        - ratio > optimal_ratio_threshold   → confidence < 75%  (low confidence)

    The optimal_ratio_threshold is calibrated on the validation set to be the
    MINIMUM ratio threshold where ≥80% directional accuracy is achieved.
    """
    model.train()  # Keep dropout ON during inference

    all_mu    = []
    all_sigma = []

    with torch.no_grad():
        for _ in range(n_samples):
            mu, log_sigma = model(input_tensor.to(device))
            sigma = torch.exp(log_sigma)
            all_mu.append(mu.cpu().item())
            all_sigma.append(sigma.cpu().item())

    all_mu    = np.array(all_mu)
    all_sigma = np.array(all_sigma)

    # Predicted return: mean of all MC samples
    predicted_return = np.mean(all_mu)

    # Epistemic uncertainty (model disagreement)
    mc_std = np.std(all_mu)

    # Aleatoric uncertainty (inherent data noise)
    avg_model_sigma = np.mean(all_sigma)

    # Combined total uncertainty
    total_uncertainty = np.sqrt(mc_std**2 + avg_model_sigma**2)

    # Dimensionless uncertainty ratio (self-calibrating across assets/horizons)
    ratio = mc_std / (avg_model_sigma + 1e-9)

    # FIXED: Piecewise-linear confidence mapping
    # - At ratio=0: confidence=100% (perfect certainty)
    # - At ratio=optimal_ratio_threshold: confidence=75% (the calibration boundary)
    # - At ratio >= optimal_ratio_threshold*3: confidence=0%
    opt = optimal_ratio_threshold + 1e-9

    if ratio <= opt:
        # High-confidence zone: linear from 100% (at 0) to 75% (at threshold)
        confidence = 100.0 - 25.0 * (ratio / opt)
    elif ratio <= opt * 3.0:
        # Declining zone: linear from 75% (at threshold) to 0% (at 3x threshold)
        confidence = 75.0 * (1.0 - (ratio - opt) / (opt * 2.0 + 1e-9))
    else:
        # Very high uncertainty: 0%
        confidence = 0.0

    confidence = float(max(0.0, min(100.0, confidence)))

    # Error range (10th-90th percentile from MC samples)
    lower_bound = np.percentile(all_mu, 10)
    upper_bound = np.percentile(all_mu, 90)

    return {
        'predicted_return': predicted_return,
        'confidence':       confidence,
        'lower_bound':      lower_bound,
        'upper_bound':      upper_bound,
        'mc_std':           mc_std,
        'model_sigma':      avg_model_sigma,
        'total_uncertainty': total_uncertainty,
        'ratio':            ratio,
        'optimal_ratio_threshold': optimal_ratio_threshold,
    }


# ---------------------------------------------------------------------------
# 3. Find and Load the Latest Saved Model
# ---------------------------------------------------------------------------

def find_latest_model(timeframe):
    """Finds the most recently saved model for the given timeframe."""
    save_dir = os.path.join(os.path.dirname(__file__), '../../models/saved/')

    # Find all model files matching this timeframe (exclude 'next_day' models)
    pattern     = os.path.join(save_dir, f'model_{timeframe}_*.pt')
    model_files = sorted(glob.glob(pattern))

    if not model_files:
        raise FileNotFoundError(f"No saved models found for timeframe '{timeframe}' in {save_dir}")

    latest_model  = model_files[-1]
    latest_params = latest_model.replace('.pt', '_params.json')

    if not os.path.exists(latest_params):
        raise FileNotFoundError(f"Params file not found: {latest_params}")

    return latest_model, latest_params


def load_model_and_params(timeframe, device):
    """
    Loads the latest saved model and its parameters.
    Handles both old (basic LSTM) and new (improved) architectures by
    detecting num_heads in params.
    """
    model_path, params_path = find_latest_model(timeframe)

    with open(params_path, 'r') as f:
        best_params = json.load(f)

    # Load model — use num_heads if present (new architecture)
    num_heads  = best_params.get('num_heads', 4)

    # Need to infer input_size from model file
    state = torch.load(model_path, map_location=device, weights_only=True)
    # Detect architecture: if 'input_grn.linear1.weight' exists → new arch
    is_new_arch = 'input_grn.linear1.weight' in state

    if is_new_arch:
        # New architecture: get input_size from input_grn.linear1.weight
        input_size = state['input_grn.linear1.weight'].shape[1]
        model = LSTMAttention(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            num_heads=num_heads
        ).to(device)
    else:
        # Legacy architecture: simple LSTM+Attention
        if 'lstm.weight_ih_l0' in state:
            input_size = state['lstm.weight_ih_l0'].shape[1]
        else:
            input_size = 46  # default fallback

        model = _build_legacy_model(input_size, best_params, device)
        # strict=False allows loading even if some keys are missing (e.g. bidirectional proj)
        model.load_state_dict(state, strict=True)
        return model, best_params, model_path

    # New architecture: load state dict
    model.load_state_dict(state, strict=True)
    return model, best_params, model_path


def _build_legacy_model(input_size, best_params, device):
    """
    Builds a legacy LSTMAttention model compatible with old checkpoints.
    Used for backward compatibility when loading pre-v2.0 models.
    """
    import torch.nn as nn

    class LegacyAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
            self.softmax = nn.Softmax(dim=1)

        def forward(self, lstm_out):
            weights = self.attention(lstm_out)
            weights = self.softmax(weights)
            return torch.sum(weights * lstm_out, dim=1)

    class LegacyLSTMAttention(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0)
            self.lstm_dropout = nn.Dropout(dropout)
            self.attention = LegacyAttention(hidden_size)
            self.fc_mu = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1)
            )
            self.fc_log_sigma = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            lstm_out = self.lstm_dropout(lstm_out)
            attn_out = self.attention(lstm_out)
            mu = self.fc_mu(attn_out).squeeze(-1)
            log_sigma = self.fc_log_sigma(attn_out).squeeze(-1)
            return mu, log_sigma

    return LegacyLSTMAttention(
        input_size, best_params['hidden_size'],
        best_params['num_layers'], best_params['dropout']
    ).to(device)


# ---------------------------------------------------------------------------
# 4. Main Prediction Pipeline
# ---------------------------------------------------------------------------

def predict(ticker="VOO"):
    """Full prediction pipeline: load model → fetch data → MC Dropout → report."""

    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]

    print(f"🔮 Predicting '{ACTIVE_TIMEFRAME}' for {ticker}...")
    print(f"   Using {profile['seq_length']} bars of {profile['interval']} data\n")

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Load model
    model, best_params, model_path = load_model_and_params(ACTIVE_TIMEFRAME, device)
    print(f"📦 Loaded model: {os.path.basename(model_path)}")

    # Load scaler and feature columns
    data_dir     = os.path.join(os.path.dirname(__file__), '../../data/processed/')
    scaler_path  = os.path.join(data_dir, f'scaler_{ACTIVE_TIMEFRAME}.pkl')
    feat_path    = os.path.join(data_dir, f'feature_cols_{ACTIVE_TIMEFRAME}.pkl')

    # Fallback for legacy (non-timeframe-suffixed) files
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_dir, 'scaler.pkl')
    if not os.path.exists(feat_path):
        feat_path = os.path.join(data_dir, 'feature_cols.pkl')

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feat_path, 'rb') as f:
        feature_cols = pickle.load(f)

    # Prepare live input
    input_tensor, current_price = prepare_live_input(ticker, profile, scaler, feature_cols)

    # MC Dropout prediction
    opt_ratio = best_params.get('optimal_ratio_threshold', 0.15)
    result = mc_dropout_predict(model, input_tensor, device, n_samples=100,
                                optimal_ratio_threshold=opt_ratio)

    # Convert returns to prices
    predicted_price = current_price * (1 + result['predicted_return'] / 100.0)
    price_lower     = current_price * (1 + result['lower_bound'] / 100.0)
    price_upper     = current_price * (1 + result['upper_bound'] / 100.0)

    print("=" * 55)
    print(f"  📊 PREDICTION REPORT — {ticker} ({ACTIVE_TIMEFRAME})")
    print("=" * 55)
    print(f"  Current Price:       ${current_price:.2f}")
    print(f"  Predicted Price:     ${predicted_price:.2f}")
    print(f"  Predicted Return:    {result['predicted_return']:+.4f}%")
    print(f"  Confidence:          {result['confidence']:.1f}%")
    print(f"  Price Range (80%):   ${price_lower:.2f} — ${price_upper:.2f}")
    print("-" * 55)
    print(f"  Uncertainty Ratio:   {result['ratio']:.4f} (threshold: {opt_ratio:.4f})")
    print(f"  MC Dropout Std:      {result['mc_std']:.4f}%")
    print(f"  Model Uncertainty:   {result['model_sigma']:.4f}%")
    print(f"  Total Uncertainty:   {result['total_uncertainty']:.4f}%")
    print("=" * 55)

    direction = "📈 BUY signal" if result['predicted_return'] > 0 else "📉 SELL signal"
    strength  = "STRONG" if result['confidence'] > 70 else "MODERATE" if result['confidence'] > 40 else "WEAK"
    print(f"\n  → {direction} ({strength}, {result['confidence']:.0f}% confidence)\n")

    return result


if __name__ == "__main__":
    predict("VOO")
