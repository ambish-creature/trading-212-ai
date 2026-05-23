import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.preprocessing import StandardScaler
import ta

# Add the root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME, ASSETS, CATEGORIES

def add_technical_indicators(df, target_shift):
    """
    Adds Technical Indicators and dividends to the DataFrame.
    """
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    
    # Target: Future Percentage Return (scaled by 100 for stability)
    df['Target_Return'] = df['Close'].pct_change(periods=target_shift).shift(-target_shift) * 100.0
    
    # Fill any missing dividends with 0.0 (yfinance downloads it as 'Dividends')
    if 'Dividends' not in df.columns:
        df['Dividends'] = 0.0
    else:
        df['Dividends'] = df['Dividends'].fillna(0.0)
        
    df.dropna(subset=[col for col in df.columns if col != 'Target_Return'], inplace=True)
    df.dropna(subset=['Target_Return'], inplace=True) # Target NaNs are dropped too
    return df

def create_sequences(features, targets, seq_length):
    """
    Creates sliding windows for time-series data.
    """
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:(i + seq_length)])
        ys.append(targets[i + seq_length - 1])
    return np.array(xs), np.array(ys)

def process_all_data():
    print("=" * 60)
    print("🧠 STARTING MULTI-ASSET CONDITIONAL DATA PREPROCESSOR")
    print("=" * 60)
    
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    seq_length = profile['seq_length']
    target_shift = profile['target_shift']
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    raw_dir = os.path.join(root_dir, 'data/raw/')
    processed_dir = os.path.join(root_dir, 'data/processed/')
    
    # Define numeric features (which need scaling)
    numeric_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'Dividends']
    
    # Step 1: Load all asset dataframes and apply technical analysis
    asset_dfs = {}
    for ticker, category in ASSETS.items():
        csv_path = os.path.join(raw_dir, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  Raw CSV not found for {ticker}, skipping.")
            continue
            
        df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
        df = add_technical_indicators(df, target_shift)
        
        # Add One-Hot Category Columns
        for cat in CATEGORIES:
            df[f"Category_{cat}"] = 1.0 if cat == category else 0.0
            
        asset_dfs[ticker] = df

    # Step 2: Split chronologically per asset and collect training features to fit a single global scaler
    train_features_list = []
    
    for ticker, df in asset_dfs.items():
        n = len(df)
        train_end = int(n * 0.7)
        train_df = df.iloc[:train_end]
        train_features_list.append(train_df[numeric_cols].values)
        
    # Fit the global scaler on stacked training sets of all assets combined
    all_train_numeric = np.vstack(train_features_list)
    scaler = StandardScaler()
    scaler.fit(all_train_numeric)
    
    # Save the global scaler
    os.makedirs(processed_dir, exist_ok=True)
    with open(os.path.join(processed_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Global StandardScaler fitted and saved successfully.")

    # Step 3: Process and scale sequences for each asset individually
    train_X_list, train_y_list = [], []
    val_X_list, val_y_list = [], []
    
    # Define final features list order
    one_hot_cols = [f"Category_{cat}" for cat in CATEGORIES]
    feature_cols = numeric_cols + one_hot_cols
    
    for ticker, df in asset_dfs.items():
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * (0.7 + 0.15))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        # Scale numeric columns
        train_scaled_num = scaler.transform(train_df[numeric_cols])
        val_scaled_num = scaler.transform(val_df[numeric_cols])
        test_scaled_num = scaler.transform(test_df[numeric_cols])
        
        # Re-combine numeric scaled features with unscaled one-hot category indicators
        train_feats = np.hstack([train_scaled_num, train_df[one_hot_cols].values])
        val_feats = np.hstack([val_scaled_num, val_df[one_hot_cols].values])
        test_feats = np.hstack([test_scaled_num, test_df[one_hot_cols].values])
        
        # Targets
        train_targets = train_df['Target_Return'].values
        val_targets = val_df['Target_Return'].values
        test_targets = test_df['Target_Return'].values
        
        # Create sequences
        X_train, y_train = create_sequences(train_feats, train_targets, seq_length)
        X_val, y_val = create_sequences(val_feats, val_targets, seq_length)
        X_test, y_test = create_sequences(test_feats, test_targets, seq_length)
        
        train_X_list.append(X_train)
        train_y_list.append(y_train)
        val_X_list.append(X_val)
        val_y_list.append(y_val)
        
        # Save individual test datasets for backtester/trading checks
        np.save(os.path.join(processed_dir, f'{ticker}_X_test.npy'), X_test)
        np.save(os.path.join(processed_dir, f'{ticker}_y_test.npy'), y_test)
        print(f"   ✓ {ticker}: test split processed. Shapes - X: {X_test.shape}, y: {y_test.shape}")

    # Step 4: Stacking and Shuffling the combined training/validation sets
    X_train_all = np.vstack(train_X_list)
    y_train_all = np.concatenate(train_y_list)
    X_val_all = np.vstack(val_X_list)
    y_val_all = np.concatenate(val_y_list)
    
    # Shuffle training set to ensure the model generalizes across assets
    shuffle_idx = np.random.permutation(len(X_train_all))
    X_train_all = X_train_all[shuffle_idx]
    y_train_all = y_train_all[shuffle_idx]
    
    # Save unified training and validation sets
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train_all)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train_all)
    np.save(os.path.join(processed_dir, 'X_val.npy'), X_val_all)
    np.save(os.path.join(processed_dir, 'y_val.npy'), y_val_all)
    
    # Save feature names for configuration
    with open(os.path.join(processed_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
        
    print("\n🏁 MULTI-ASSET CONDITIONAL DATA PREPROCESSOR COMPLETED")
    print(f"   Unified Train shape:      {X_train_all.shape}")
    print(f"   Unified Validation shape: {X_val_all.shape}")
    print(f"   Total Feature Dimension:  {len(feature_cols)} features ({feature_cols})")
    print("=" * 60)

if __name__ == "__main__":
    process_all_data()
