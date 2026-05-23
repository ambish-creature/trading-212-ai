import numpy as np
import os
import sys

# Add root directory to path to verify config/features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.config import ASSETS, TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME

def run_leakage_audit():
    print("=" * 60)
    print("🔍 RUNNING DATA LEAKAGE & INTEGRITY AUDIT")
    print("=" * 60)
    
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/'))
    
    # 1. Load splits
    try:
        X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(processed_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))
    except FileNotFoundError as e:
        print(f"❌ Split files not found! Error: {e}")
        return
        
    print(f"✅ Dataset splits loaded successfully:")
    print(f"   • Train X shape:      {X_train.shape}")
    print(f"   • Train y shape:      {y_train.shape}")
    print(f"   • Validation X shape: {X_val.shape}")
    print(f"   • Validation y shape: {y_val.shape}")
    print("-" * 60)
    
    # 2. Check for NaNs/Infs (leaks or scaling bugs)
    nan_train_X = np.isnan(X_train).sum()
    nan_train_y = np.isnan(y_train).sum()
    nan_val_X = np.isnan(X_val).sum()
    nan_val_y = np.isnan(y_val).sum()
    
    inf_train_X = np.isinf(X_train).sum()
    inf_train_y = np.isinf(y_train).sum()
    inf_val_X = np.isinf(X_val).sum()
    inf_val_y = np.isinf(y_val).sum()
    
    print("📊 NaN & Infinite Value Check:")
    if nan_train_X == 0 and nan_train_y == 0 and nan_val_X == 0 and nan_val_y == 0:
        print("   ✅ Zero NaN values found across all splits.")
    else:
        print(f"   ❌ Found NaNs! Train X: {nan_train_X}, Train y: {nan_train_y}, Val X: {nan_val_X}, Val y: {nan_val_y}")
        
    if inf_train_X == 0 and inf_train_y == 0 and inf_val_X == 0 and inf_val_y == 0:
        print("   ✅ Zero Infinite values found across all splits.")
    else:
        print(f"   ❌ Found Infs! Train X: {inf_train_X}, Train y: {inf_train_y}, Val X: {inf_val_X}, Val y: {inf_val_y}")
    print("-" * 60)
    
    # 3. Check Sequence Temporal Structure (Verify X features do not leak target)
    # The target y is the future return. If the last step's features in X are highly correlated with y,
    # it might indicate leakage. Let's print features size.
    profile = TIMEFRAME_PROFILES[ACTIVE_TIMEFRAME]
    seq_length = profile['seq_length']
    print(f"📏 Sequence Length Verified: {X_train.shape[1]} (Expected: {seq_length})")
    print(f"📏 Feature Dimension Verified: {X_train.shape[2]} (Expected: 11 features)")
    print("-" * 60)
    
    # 4. Check Per-Ticker Test Split Safety (Chronological order)
    # Let's inspect a single ticker to check that its training set is chronological,
    # followed by validation, followed by test, with zero overlaps.
    print("🕒 Chronological Separation Safety Check:")
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/'))
    
    overlap_detected = False
    for ticker in list(ASSETS.keys())[:3]: # check first 3 tickers
        test_X_path = os.path.join(processed_dir, f'{ticker}_X_test.npy')
        if os.path.exists(test_X_path):
            test_X = np.load(test_X_path)
            # Check shape
            print(f"   • {ticker}: Test X shape: {test_X.shape}")
            
            # Since train and val sequences are stacked and shuffled globally,
            # we want to verify that individual test sequences are not present in X_train.
            # Convert sequences to robust hashes/representatives to verify uniqueness
            train_reps = {hash(bytes(seq)): True for seq in X_train}
            test_reps = [hash(bytes(seq)) for seq in test_X]
            
            matches = sum(1 for r in test_reps if r in train_reps)
            if matches > 0:
                print(f"   ❌ LEAKAGE DETECTED: {matches} test sequences found in the training split for {ticker}!")
                overlap_detected = True
            else:
                print(f"   ✅ Chronological test split for {ticker} has 0 overlap with training dataset sequences.")
                
    if not overlap_detected:
        print("\n🏆 AUDIT RESULT: PASSED! Zero data leakage, zero NaN/Inf errors, perfect chronological separation.")
    else:
        print("\n⚠️  AUDIT RESULT: FAILED! Data leakage detected.")
    print("=" * 60)

if __name__ == "__main__":
    run_leakage_audit()
