"""
verify_accuracy.py — Comprehensive accuracy verification for all trained models.

Runs selective classification analysis:
- Loads test split for a ticker/timeframe
- Runs MC Dropout predictions
- Evaluates high-confidence directional accuracy (goal: ≥80% at ≥75% confidence)
- Prints breakdown by confidence bucket
- Reports calibration curve

Usage:
  python scratch/verify_accuracy.py                          # 1mo VOO (default)
  python scratch/verify_accuracy.py --timeframe 3mo --ticker NVDA
  python scratch/verify_accuracy.py --all-timeframes         # sweep all 7 horizons
  python scratch/verify_accuracy.py --all-timeframes --ticker TSM
"""

import os
import sys
import torch
import numpy as np
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.config import TIMEFRAME_PROFILES
from src.models.train import LSTMAttention, compute_optimal_ratio_threshold
from src.models.predict import find_latest_model, mc_dropout_predict, _build_legacy_model

ROOT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
ALL_TIMEFRAMES = ['1mo', '2mo', '3mo', '6mo', '9mo', '1yr', '2yr']


def run_accuracy_verification(timeframe="1mo", confidence_threshold=75.0, ticker="VOO", n_mc_samples=50):
    print("=" * 70)
    print(f"📊 ACCURACY VERIFICATION: {timeframe} | {ticker}")
    print("=" * 70)

    # Load model
    try:
        model_path, params_path = find_latest_model(timeframe)
        print(f"📦 Model: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"❌ No model found for '{timeframe}': {e}")
        return None

    with open(params_path) as f:
        params = json.load(f)

    # Load test data
    X_test_path = os.path.join(PROCESSED_DIR, f'{ticker}_X_test_{timeframe}.npy')
    y_test_path = os.path.join(PROCESSED_DIR, f'{ticker}_y_test_{timeframe}.npy')

    if not os.path.exists(X_test_path):
        print(f"❌ Test data not found: {X_test_path}")
        return None

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    print(f"   Test shapes: X={X_test.shape}, y={y_test.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Load model with architecture detection
    state       = torch.load(model_path, map_location=device, weights_only=True)
    is_new_arch = 'input_grn.linear1.weight' in state
    num_heads   = params.get('num_heads', 4)

    if is_new_arch:
        input_size = state['input_grn.linear1.weight'].shape[1]
        model = LSTMAttention(input_size, params['hidden_size'], params['num_layers'],
                              params['dropout'], num_heads).to(device)
    else:
        input_size = state.get('lstm.weight_ih_l0', torch.zeros(1, X_test.shape[2], 1)).shape[1] \
                     if 'lstm.weight_ih_l0' in state else X_test.shape[2]
        model = _build_legacy_model(input_size, params, device)

    model.load_state_dict(state, strict=False)
    print(f"   Architecture: {'NEW v2.0' if is_new_arch else 'LEGACY v1.x'} | device={device}")

    opt_ratio = params.get('optimal_ratio_threshold', 0.15)
    print(f"   Optimal ratio threshold: {opt_ratio:.4f}")
    if opt_ratio > 0.5:
        print(f"   ⚠️  WARNING: threshold {opt_ratio:.4f} appears uncalibrated (> 0.5). "
              f"Run scratch/calibrate.py to fix.")

    # Run predictions
    print(f"\n🔮 Running MC Dropout on {len(X_test)} samples ({n_mc_samples} passes each)...")
    predictions  = []
    confidences  = []
    ratios       = []
    true_returns = list(y_test)

    model.train()
    with torch.no_grad():
        for i in range(len(X_test)):
            if i % 50 == 0:
                print(f"   ... {i}/{len(X_test)}", end='\r')
            seq = torch.tensor(X_test[i:i+1], dtype=torch.float32)
            res = mc_dropout_predict(model, seq, device, n_samples=n_mc_samples,
                                     optimal_ratio_threshold=opt_ratio)
            predictions.append(res['predicted_return'])
            confidences.append(res['confidence'])
            ratios.append(res['ratio'])

    predictions  = np.array(predictions)
    confidences  = np.array(confidences)
    ratios       = np.array(ratios)
    true_returns = np.array(true_returns)

    pred_signs = np.sign(predictions)
    true_signs = np.sign(true_returns)
    correct    = (pred_signs == true_signs)

    print(f"\n📈 Ratio Distribution (diagnostic):")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"   P{pct:02d}: {np.percentile(ratios, pct):.4f}")
    print(f"   Optimal threshold: {opt_ratio:.4f}")
    fractions_below = (ratios <= opt_ratio).mean() * 100
    print(f"   Fraction with ratio ≤ threshold: {fractions_below:.1f}%")

    print(f"\n📊 Unfiltered Results (all {len(X_test)} samples):")
    overall_acc = correct.mean()
    print(f"   • Directional Accuracy: {overall_acc*100:.2f}%")
    print(f"   • Bullish predictions: {(predictions > 0).sum()} ({(predictions > 0).mean()*100:.1f}%)")
    print(f"   • Bearish predictions: {(predictions < 0).sum()} ({(predictions < 0).mean()*100:.1f}%)")

    # Confidence bucket breakdown
    print(f"\n📈 Breakdown by Confidence Bucket:")
    buckets = [(90, 100), (80, 90), (75, 80), (60, 75), (40, 60), (0, 40)]
    for lo, hi in buckets:
        mask  = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        if count > 0:
            acc = correct[mask].mean()
            bull_pct = (predictions[mask] > 0).mean() * 100
            print(f"   [{lo:3d}%–{hi:3d}%): {count:4d} samples | Acc={acc*100:.1f}% | Bull%={bull_pct:.0f}%")
        else:
            print(f"   [{lo:3d}%–{hi:3d}%): {count:4d} samples")

    # High-confidence evaluation
    high_conf_mask  = confidences >= confidence_threshold
    high_conf_count = high_conf_mask.sum()

    print(f"\n🎯 High-Confidence Results (≥{confidence_threshold:.0f}%, Count: {high_conf_count}/{len(X_test)}):")
    if high_conf_count == 0:
        print(f"   ❌ ZERO predictions met the threshold!")
        print(f"   Max confidence achieved: {confidences.max():.2f}%")
        is_success = False
    else:
        high_conf_acc      = correct[high_conf_mask].mean()
        coverage_rate      = high_conf_count / len(X_test)
        bull_high          = (predictions[high_conf_mask] > 0).mean() * 100
        print(f"   • High-Confidence Accuracy: {high_conf_acc*100:.2f}%  (goal: ≥80%)")
        print(f"   • Coverage Rate:            {coverage_rate*100:.2f}%  (goal: ≥5%)")
        print(f"   • Bull % at high conf:      {bull_high:.1f}%")

        is_success = (high_conf_acc >= 0.80) and (high_conf_count >= max(5, int(len(X_test) * 0.05)))

        if is_success:
            print("\n✅ GOAL ACHIEVED! ≥80% directional accuracy on high-confidence predictions.")
        else:
            if high_conf_acc < 0.80:
                print(f"\n❌ Accuracy {high_conf_acc*100:.2f}% < 80% goal. Model needs more training or recalibration.")
            if high_conf_count < max(5, int(len(X_test) * 0.05)):
                print(f"❌ Coverage too low ({high_conf_count} samples). Recalibrate or lower threshold.")

    print("=" * 70)
    return {
        'timeframe': timeframe,
        'ticker':    ticker,
        'n_samples': len(X_test),
        'overall_accuracy': float(overall_acc),
        'high_conf_count': int(high_conf_count),
        'high_conf_accuracy': float(correct[high_conf_mask].mean()) if high_conf_count > 0 else 0.0,
        'coverage_rate': float(high_conf_count / len(X_test)),
        'goal_achieved': is_success,
        'optimal_ratio_threshold': float(opt_ratio),
        'ratio_p50': float(np.median(ratios)),
        'fraction_below_threshold': float(fractions_below),
    }


def run_all_timeframes_sweep(ticker="VOO", confidence_threshold=75.0):
    """Run verification across all 7 horizons for a given ticker."""
    print(f"\n{'#'*70}")
    print(f"# FULL SWEEP: {ticker} across all timeframes")
    print(f"{'#'*70}\n")

    summary = {}
    for tf in ALL_TIMEFRAMES:
        result = run_accuracy_verification(tf, confidence_threshold, ticker, n_mc_samples=30)
        if result:
            summary[tf] = result
        print()

    # Print summary table
    print(f"\n{'='*70}")
    print(f"📋 SWEEP SUMMARY — {ticker}")
    print(f"{'='*70}")
    print(f"{'TF':6s} | {'Overall':8s} | {'HiConf':8s} | {'Coverage':9s} | {'Goal':6s} | {'Threshold':10s}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}-+-{'-'*10}")
    for tf, r in summary.items():
        goal = "✅" if r['goal_achieved'] else "❌"
        print(f"{tf:6s} | {r['overall_accuracy']*100:7.2f}% | "
              f"{r['high_conf_accuracy']*100:7.2f}% | "
              f"{r['coverage_rate']*100:8.1f}% | {goal:6s} | "
              f"{r['optimal_ratio_threshold']:.4f}")

    goals_met = sum(1 for r in summary.values() if r['goal_achieved'])
    print(f"\n{'='*70}")
    print(f"  Goals met: {goals_met}/{len(summary)} timeframes")
    print(f"{'='*70}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive model accuracy verification.")
    parser.add_argument("--timeframe",      type=str,   default="1mo")
    parser.add_argument("--threshold",      type=float, default=75.0)
    parser.add_argument("--ticker",         type=str,   default="VOO")
    parser.add_argument("--mc-samples",     type=int,   default=50)
    parser.add_argument("--all-timeframes", action="store_true",
                        help="Sweep all 7 timeframes.")
    args = parser.parse_args()

    if args.all_timeframes:
        run_all_timeframes_sweep(args.ticker, args.threshold)
    else:
        run_accuracy_verification(args.timeframe, args.threshold, args.ticker, args.mc_samples)
