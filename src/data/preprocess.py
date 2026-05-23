import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import ta
import pickle

def add_technical_indicators(df):
    """
    Adds Technical Indicators to the DataFrame using the 'ta' library.
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
    
    # Target: Next Day's Percentage Return
    # We shift -1 so today's row contains tomorrow's return
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    
    # Drop NaNs created by indicators and shifting
    df.dropna(inplace=True)
    return df

def create_sequences(features, targets, seq_length):
    """
    Creates sliding windows for time-series data.
    """
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i:(i + seq_length)]
        y = targets[i + seq_length - 1] # Target corresponding to the end of the window
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def process_and_split_data(input_path, output_dir, seq_length=60, train_split=0.7, val_split=0.15):
    """
    Full pipeline: Load -> Indicators -> Scale -> Sequence -> Split -> Save
    """
    print(f"Processing {input_path}...")
    df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
    
    # Add indicators and target
    df = add_technical_indicators(df)
    
    # Define feature columns (exclude target)
    feature_cols = [col for col in df.columns if col != 'Target_Return']
    
    # Chronological Split boundaries
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Scale Features (Fit only on training data to prevent data leakage)
    scaler = StandardScaler()
    
    train_features = scaler.fit_transform(train_df[feature_cols])
    val_features = scaler.transform(val_df[feature_cols])
    test_features = scaler.transform(test_df[feature_cols])
    
    train_targets = train_df['Target_Return'].values
    val_targets = val_df['Target_Return'].values
    test_targets = test_df['Target_Return'].values
    
    # Create sliding windows
    X_train, y_train = create_sequences(train_features, train_targets, seq_length)
    X_val, y_val = create_sequences(val_features, val_targets, seq_length)
    X_test, y_test = create_sequences(test_features, test_targets, seq_length)
    
    # Save the processed data and the scaler
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save the scaler to inverse_transform during inference
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Data processed and saved to {output_dir}")
    print(f"Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Save the feature column names for reference in training
    with open(os.path.join(output_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)

if __name__ == "__main__":
    # Assumes fetch.py was run to generate AAPL.csv
    raw_path = os.path.join(os.path.dirname(__file__), '../../data/raw/AAPL.csv')
    processed_dir = os.path.join(os.path.dirname(__file__), '../../data/processed/')
    
    if os.path.exists(raw_path):
        process_and_split_data(raw_path, processed_dir, seq_length=60)
    else:
        print(f"Raw data not found at {raw_path}. Please run src/data/fetch.py first.")
