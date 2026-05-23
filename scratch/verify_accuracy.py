import os
import sys
import torch
import numpy as np
import json
import argparse
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.config import TIMEFRAME_PROFILES, ASSETS
from src.models.train import LSTMAttention
from src.models.predict import find_latest_model, mc_dropout_predict

def run_accuracy_verification(timeframe="1mo", confidence_threshold=75.0, ticker="VOO"):
    print("=" * 70)
    print(f"📊 VERIFYING HIGH-CONFIDENCE SIGN ACCURACY FOR TIMEFRAME: {timeframe} | TICKER: {ticker}")
    print("=" * 70)
    
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/'))
    
    # Load model
    try:
        model_path, params_path = find_latest_model(timeframe)
        print(f"📦 Loaded model: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"❌ Failed to find saved model for timeframe '{timeframe}': {e}")
        return False
        
    with open(params_path, 'r') as f:
        best_params = json.load(f)
        
    # Load test split for ticker
    X_test_path = os.path.join(processed_dir, f'{ticker}_X_test_{timeframe}.npy')
    y_test_path = os.path.join(processed_dir, f'{ticker}_y_test_{timeframe}.npy')
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"❌ Test split files not found for {ticker} under {timeframe} timeframe!")
        return False
        
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    print(f"   Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"   Device: {device}")
    
    # Initialize and load model weights
    input_size = X_test.shape[2]
    model = LSTMAttention(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Run predictions
    print(f"🔮 Executing MC Dropout predictions on {len(X_test)} samples (selective classification)...")
    
    preds = []
    confidences = []
    trues = []
    
    med_u = best_params.get('median_uncertainty', None)
    for i in range(len(X_test)):
        seq = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        res = mc_dropout_predict(model, seq, device, n_samples=50, median_uncertainty=med_u) # 50 samples is faster for validation
        preds.append(res['predicted_return'])
        confidences.append(res['confidence'])
        trues.append(y_test[i])

        
    preds = np.array(preds)
    confidences = np.array(confidences)
    trues = np.array(trues)
    
    # Evaluate Sign Accuracy
    pred_signs = np.sign(preds)
    true_signs = np.sign(trues)
    
    # Unfiltered metrics
    unfiltered_acc = np.mean(pred_signs == true_signs)
    print(f"\n📈 Unfiltered Results (Total: {len(X_test)}):")
    print(f"   • Overall Directional Accuracy: {unfiltered_acc * 100:.2f}%")
    
    # High-confidence metrics
    high_conf_mask = confidences >= confidence_threshold
    high_conf_count = np.sum(high_conf_mask)
    
    if high_conf_count == 0:
        print(f"\n⚠️  Zero predictions met the high-confidence threshold of {confidence_threshold}%!")
        # Print some info about the max confidence achieved
        print(f"   Max confidence achieved: {np.max(confidences):.2f}%")
        return False
        
    high_conf_preds = preds[high_conf_mask]
    high_conf_trues = trues[high_conf_mask]
    high_conf_pred_signs = pred_signs[high_conf_mask]
    high_conf_true_signs = true_signs[high_conf_mask]
    
    high_conf_acc = np.mean(high_conf_pred_signs == high_conf_true_signs)
    
    print(f"\n🎯 High-Confidence Results (Threshold >= {confidence_threshold}%, Count: {high_conf_count}/{len(X_test)}):")
    print(f"   • High-Confidence Accuracy: {high_conf_acc * 100:.2f}%")
    print(f"   • Coverage Rate:            {high_conf_count / len(X_test) * 100:.2f}%")
    
    is_success = high_conf_acc >= 0.80
    if is_success:
        print("\n✅ GOAL ACHIEVED! Directional accuracy is at least 80% on high-confidence predictions.")
    else:
        print("\n❌ GOAL NOT YET ACHIEVED: Directional accuracy is below 80%. Need more tuning or training epochs.")
        
    print("=" * 70)
    return is_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify high-confidence sign accuracy.")
    parser.add_argument("--timeframe", type=str, default="1mo")
    parser.add_argument("--threshold", type=float, default=75.0)
    parser.add_argument("--ticker", type=str, default="VOO")
    args = parser.parse_args()
    
    run_accuracy_verification(args.timeframe, args.threshold, args.ticker)
