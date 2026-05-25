"""
calibrate.py — Post-hoc calibration for all trained models.

This script:
1. Loads each trained model (all timeframes)
2. Runs the fixed calibration sweep on the validation set
3. Saves optimal_ratio_threshold back to the params.json

Run this once after training to fix calibration on pre-v2.0 models.

Usage:
  python scratch/calibrate.py                     # calibrate all timeframes
  python scratch/calibrate.py --timeframe 1mo     # calibrate specific timeframe
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.config import TIMEFRAME_PROFILES
from src.models.train import LSTMAttention, compute_optimal_ratio_threshold
from src.models.predict import find_latest_model, _build_legacy_model

ROOT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
ALL_TIMEFRAMES = ['1mo', '2mo', '3mo', '6mo', '9mo', '1yr', '2yr']


def calibrate_timeframe(timeframe, n_mc_samples=30):
    """
    Load model for the given timeframe, run calibration on validation set,
    and write optimal_ratio_threshold to the params.json.
    """
    print(f"\n{'='*60}")
    print(f"🔧 Calibrating timeframe: {timeframe}")
    print(f"{'='*60}")

    # Load model
    try:
        model_path, params_path = find_latest_model(timeframe)
    except FileNotFoundError as e:
        print(f"   ❌ No model found for '{timeframe}': {e}")
        return False

    with open(params_path) as f:
        params = json.load(f)

    print(f"   📦 Model: {os.path.basename(model_path)}")
    print(f"   Current optimal_ratio_threshold: {params.get('optimal_ratio_threshold', 'MISSING')}")

    device = torch.device('cuda' if torch.cuda.is_available() else
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Load validation data
    X_val_path = os.path.join(PROCESSED_DIR, f'X_val_{timeframe}.npy')
    y_val_path = os.path.join(PROCESSED_DIR, f'y_val_{timeframe}.npy')

    if not os.path.exists(X_val_path) or not os.path.exists(y_val_path):
        print(f"   ❌ Validation data not found for '{timeframe}'. Run preprocess.py first.")
        return False

    X_val = torch.tensor(np.load(X_val_path), dtype=torch.float32)
    y_val = torch.tensor(np.load(y_val_path), dtype=torch.float32)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    print(f"   📊 Validation shape: X={X_val.shape}, y={y_val.shape}")

    # Build model — detect architecture
    state = torch.load(model_path, map_location=device, weights_only=True)
    is_new_arch = 'input_grn.linear1.weight' in state

    if is_new_arch:
        input_size = state['input_grn.linear1.weight'].shape[1]
        num_heads  = params.get('num_heads', 4)
        model = LSTMAttention(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            num_heads=num_heads
        ).to(device)
        print(f"   🏗️  Detected NEW architecture (v2.0+)")
    else:
        if 'lstm.weight_ih_l0' in state:
            input_size = state['lstm.weight_ih_l0'].shape[1]
        else:
            input_size = X_val.shape[2]
        model = _build_legacy_model(input_size, params, device)
        print(f"   🏗️  Detected LEGACY architecture (v1.x)")

    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"   ❌ Failed to load model weights: {e}")
        return False

    print(f"   ✅ Model loaded successfully (input_size={input_size})")

    # Run calibration
    print(f"   🔍 Running calibration with {n_mc_samples} MC samples per sample...")
    optimal_r, calibration_achieved = compute_optimal_ratio_threshold(
        model, val_loader, device, n_mc_samples=n_mc_samples
    )

    # Update params
    params['optimal_ratio_threshold']   = optimal_r
    params['calibration_achieved_80pct'] = calibration_achieved

    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    status = "✅ 80% accuracy achieved" if calibration_achieved else "⚠️  Best possible (< 80%)"
    print(f"\n   Result: threshold={optimal_r:.4f} — {status}")
    print(f"   ✅ Saved to: {os.path.basename(params_path)}")
    return True


def run_all_calibrations(timeframes=None, n_mc_samples=30):
    """Calibrate all specified (or all available) timeframes."""
    if timeframes is None:
        timeframes = ALL_TIMEFRAMES

    results = {}
    for tf in timeframes:
        success = calibrate_timeframe(tf, n_mc_samples=n_mc_samples)
        results[tf] = success

    print(f"\n{'='*60}")
    print("📋 CALIBRATION SUMMARY")
    print(f"{'='*60}")
    for tf, ok in results.items():
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"   {tf:6s}: {status}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-hoc calibration for all trained models.")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Calibrate only this timeframe (e.g. 1mo). Default: all.")
    parser.add_argument("--mc-samples", type=int, default=30,
                        help="Number of MC dropout samples per data point (default: 30).")
    args = parser.parse_args()

    if args.timeframe:
        calibrate_timeframe(args.timeframe, n_mc_samples=args.mc_samples)
    else:
        run_all_calibrations(n_mc_samples=args.mc_samples)
