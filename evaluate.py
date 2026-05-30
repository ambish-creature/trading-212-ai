#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           TRADING MODEL EVALUATION & DIAGNOSTICS            ║
║                                                              ║
║  Run this to get a full picture of all trained models,       ║
║  their accuracy, confidence, and what needs improvement.     ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python evaluate.py                  # Full evaluation (all timeframes, VOO)
    python evaluate.py --ticker AAPL    # Evaluate for a specific ticker
    python evaluate.py --all-tickers    # Evaluate across ALL tickers
    python evaluate.py --quick          # Quick mode (fewer MC samples, faster)
    python evaluate.py --timeframe 1mo  # Single timeframe only
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ─── Setup paths ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TIMEFRAME_PROFILES, ASSETS
from src.models.train import LSTMAttention

# ─── Constants ───────────────────────────────────────────────
ALL_TIMEFRAMES = list(TIMEFRAME_PROFILES.keys())
ALL_TICKERS = list(ASSETS.keys())
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ─── ANSI Colors ─────────────────────────────────────────────
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def colorize_accuracy(acc):
    """Color-code accuracy: green ≥80%, yellow ≥60%, red <60%."""
    if acc >= 80:
        return f"{C.GREEN}{C.BOLD}{acc:.1f}%{C.END}"
    elif acc >= 60:
        return f"{C.YELLOW}{acc:.1f}%{C.END}"
    else:
        return f"{C.RED}{acc:.1f}%{C.END}"

def colorize_bool(val):
    return f"{C.GREEN}✓ YES{C.END}" if val else f"{C.RED}✗ NO{C.END}"

def bar_chart(value, max_val=100, width=20, filled_char='█', empty_char='░'):
    """Create a simple text bar chart."""
    ratio = min(value / max_val, 1.0) if max_val > 0 else 0
    filled = int(ratio * width)
    return filled_char * filled + empty_char * (width - filled)

# ─── Device ──────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ─── Model Discovery ────────────────────────────────────────
def discover_all_models():
    """Find all saved models grouped by timeframe."""
    models = defaultdict(list)
    if not MODELS_DIR.exists():
        return models
    for pt_file in sorted(MODELS_DIR.glob("model_*.pt")):
        name = pt_file.stem
        if "_params" in name:
            continue
        parts = name.split("_")
        # model_{timeframe}_{date}_{time}.pt
        # timeframe might be multi-part like "next_day"
        if len(parts) >= 4:
            # Extract timestamp (last two parts)
            ts_str = f"{parts[-2]}_{parts[-1]}"
            # Timeframe is everything between 'model' and the timestamp
            tf = "_".join(parts[1:-2])
            params_file = pt_file.parent / f"{name}_params.json"
            params = {}
            if params_file.exists():
                with open(params_file) as f:
                    params = json.load(f)
            try:
                dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            except ValueError:
                dt = None
            models[tf].append({
                "path": pt_file,
                "params_path": params_file,
                "params": params,
                "timestamp": ts_str,
                "datetime": dt,
                "size_mb": pt_file.stat().st_size / (1024 * 1024),
            })
    return models


def find_latest_model(timeframe):
    """Find the latest model for a given timeframe."""
    pattern = str(MODELS_DIR / f"model_{timeframe}_*.pt")
    candidates = [f for f in sorted(glob.glob(pattern)) if "_params" not in f]
    if not candidates:
        return None, None
    model_path = candidates[-1]
    params_path = model_path.replace(".pt", "_params.json")
    params = {}
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
    return model_path, params


def load_model(model_path, params, device):
    """Load a model from disk."""
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Detect architecture
    is_new_arch = "input_grn.linear1.weight" in state_dict

    # Get input_size from weights
    if is_new_arch:
        input_size = state_dict["input_grn.linear1.weight"].shape[1]
    else:
        input_size = state_dict.get("lstm.weight_ih_l0", list(state_dict.values())[0]).shape[1]

    hidden_size = params.get("hidden_size", 128)
    num_layers = params.get("num_layers", 2)
    dropout = params.get("dropout", 0.2)
    num_heads = params.get("num_heads", 4)

    model = LSTMAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    return model, input_size


def mc_dropout_predict_batch(model, X_tensor, n_mc=30):
    """Run MC Dropout on a batch of inputs. Returns predictions and uncertainties."""
    model.train()  # Enable dropout
    all_mu = []
    all_sigma = []

    with torch.no_grad():
        for _ in range(n_mc):
            mu, log_sigma = model(X_tensor)
            sigma = torch.exp(log_sigma)
            all_mu.append(mu.cpu().numpy().flatten())
            all_sigma.append(sigma.cpu().numpy().flatten())

    all_mu = np.array(all_mu)       # (n_mc, n_samples)
    all_sigma = np.array(all_sigma) # (n_mc, n_samples)

    pred_return = all_mu.mean(axis=0)
    mc_std = all_mu.std(axis=0)
    avg_sigma = all_sigma.mean(axis=0)
    total_unc = np.sqrt(mc_std**2 + avg_sigma**2)
    ratio = mc_std / (avg_sigma + 1e-9)

    # Confidence percentile bands
    lower = np.percentile(all_mu, 10, axis=0)
    upper = np.percentile(all_mu, 90, axis=0)

    return {
        "pred_return": pred_return,
        "mc_std": mc_std,
        "avg_sigma": avg_sigma,
        "total_uncertainty": total_unc,
        "ratio": ratio,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def compute_confidence(ratio, opt_threshold):
    """Convert ratio to confidence percentage."""
    conf = np.zeros_like(ratio)
    for i, r in enumerate(ratio):
        if r <= opt_threshold:
            conf[i] = 100.0 - 25.0 * (r / (opt_threshold + 1e-9))
        elif r <= opt_threshold * 3:
            conf[i] = 75.0 * (1.0 - (r - opt_threshold) / (opt_threshold * 2 + 1e-9))
        else:
            conf[i] = 0.0
    return np.clip(conf, 0, 100)


# ─── Evaluation ──────────────────────────────────────────────

def evaluate_timeframe(timeframe, tickers, device, n_mc=30, verbose=True):
    """Evaluate a single timeframe across given tickers."""
    model_path, params = find_latest_model(timeframe)
    if model_path is None:
        return None

    try:
        model, input_size = load_model(model_path, params, device)
    except Exception as e:
        if verbose:
            print(f"  {C.RED}Error loading model: {e}{C.END}")
        return None

    opt_threshold = params.get("optimal_ratio_threshold", 0.5)
    cal_80 = params.get("calibration_achieved_80pct", False)

    results = {
        "timeframe": timeframe,
        "model_path": model_path,
        "params": params,
        "opt_threshold": opt_threshold,
        "calibration_80pct": cal_80,
        "input_size": input_size,
        "ticker_results": {},
        "all_preds": [],
        "all_targets": [],
        "all_ratios": [],
        "all_confidences": [],
    }

    for ticker in tickers:
        X_path = DATA_DIR / f"{ticker}_X_test_{timeframe}.npy"
        y_path = DATA_DIR / f"{ticker}_y_test_{timeframe}.npy"

        if not X_path.exists() or not y_path.exists():
            continue

        X_test = np.load(X_path)
        y_test = np.load(y_path)

        if len(X_test) == 0:
            continue

        # Check feature dimension match
        if X_test.shape[2] != input_size:
            if verbose:
                print(f"  {C.YELLOW}⚠ {ticker}: feature mismatch (data={X_test.shape[2]}, model={input_size}), skipping{C.END}")
            continue

        X_tensor = torch.FloatTensor(X_test).to(device)

        # Run MC Dropout in batches
        batch_size = 256
        all_preds_t = []
        all_mc_std_t = []
        all_sigma_t = []
        all_ratio_t = []

        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start:start+batch_size]
            mc_out = mc_dropout_predict_batch(model, batch, n_mc)
            all_preds_t.append(mc_out["pred_return"])
            all_mc_std_t.append(mc_out["mc_std"])
            all_sigma_t.append(mc_out["avg_sigma"])
            all_ratio_t.append(mc_out["ratio"])

        preds = np.concatenate(all_preds_t)
        mc_stds = np.concatenate(all_mc_std_t)
        sigmas = np.concatenate(all_sigma_t)
        ratios = np.concatenate(all_ratio_t)
        confidences = compute_confidence(ratios, opt_threshold)

        # Directional accuracy
        pred_signs = np.sign(preds)
        true_signs = np.sign(y_test)
        correct = (pred_signs == true_signs)

        # High-confidence subset (confidence ≥ 75%)
        hc_mask = confidences >= 75.0
        hc_correct = correct[hc_mask] if hc_mask.sum() > 0 else np.array([])

        # Store per-ticker results
        ticker_res = {
            "n_samples": len(y_test),
            "overall_accuracy": correct.mean() * 100 if len(correct) > 0 else 0,
            "hc_accuracy": hc_correct.mean() * 100 if len(hc_correct) > 0 else 0,
            "hc_count": int(hc_mask.sum()),
            "hc_coverage": hc_mask.mean() * 100 if len(hc_mask) > 0 else 0,
            "mean_pred": preds.mean(),
            "mean_actual": y_test.mean(),
            "mean_confidence": confidences.mean(),
            "median_ratio": np.median(ratios),
            "mae": np.mean(np.abs(preds - y_test)),
            "rmse": np.sqrt(np.mean((preds - y_test)**2)),
            "mean_mc_std": mc_stds.mean(),
            "mean_sigma": sigmas.mean(),
            # Directional breakdown
            "bullish_preds": int((pred_signs > 0).sum()),
            "bearish_preds": int((pred_signs < 0).sum()),
            "actual_bullish": int((true_signs > 0).sum()),
            "actual_bearish": int((true_signs < 0).sum()),
        }
        results["ticker_results"][ticker] = ticker_res

        # Aggregate
        results["all_preds"].extend(preds.tolist())
        results["all_targets"].extend(y_test.tolist())
        results["all_ratios"].extend(ratios.tolist())
        results["all_confidences"].extend(confidences.tolist())

    # Compute aggregated metrics
    if results["all_preds"]:
        all_p = np.array(results["all_preds"])
        all_t = np.array(results["all_targets"])
        all_r = np.array(results["all_ratios"])
        all_c = np.array(results["all_confidences"])
        correct_all = (np.sign(all_p) == np.sign(all_t))

        hc_mask = all_c >= 75.0
        hc_correct = correct_all[hc_mask] if hc_mask.sum() > 0 else np.array([])

        results["aggregate"] = {
            "total_samples": len(all_p),
            "overall_accuracy": correct_all.mean() * 100,
            "hc_accuracy": hc_correct.mean() * 100 if len(hc_correct) > 0 else 0,
            "hc_count": int(hc_mask.sum()),
            "hc_coverage": hc_mask.mean() * 100,
            "mae": np.mean(np.abs(all_p - all_t)),
            "rmse": np.sqrt(np.mean((all_p - all_t)**2)),
            "mean_pred": all_p.mean(),
            "std_pred": all_p.std(),
            "mean_actual": all_t.mean(),
            "std_actual": all_t.std(),
            "mean_confidence": all_c.mean(),
            "ratio_p25": np.percentile(all_r, 25),
            "ratio_p50": np.percentile(all_r, 50),
            "ratio_p75": np.percentile(all_r, 75),
            "ratio_p95": np.percentile(all_r, 95),
            # Confidence buckets
            "bucket_90_100": int((all_c >= 90).sum()),
            "bucket_80_90": int(((all_c >= 80) & (all_c < 90)).sum()),
            "bucket_75_80": int(((all_c >= 75) & (all_c < 80)).sum()),
            "bucket_60_75": int(((all_c >= 60) & (all_c < 75)).sum()),
            "bucket_40_60": int(((all_c >= 40) & (all_c < 60)).sum()),
            "bucket_0_40": int((all_c < 40).sum()),
            # Accuracy per bucket
            "acc_90_100": correct_all[all_c >= 90].mean() * 100 if (all_c >= 90).sum() > 0 else None,
            "acc_80_90": correct_all[(all_c >= 80) & (all_c < 90)].mean() * 100 if ((all_c >= 80) & (all_c < 90)).sum() > 0 else None,
            "acc_75_80": correct_all[(all_c >= 75) & (all_c < 80)].mean() * 100 if ((all_c >= 75) & (all_c < 80)).sum() > 0 else None,
            "acc_60_75": correct_all[(all_c >= 60) & (all_c < 75)].mean() * 100 if ((all_c >= 60) & (all_c < 75)).sum() > 0 else None,
            "acc_40_60": correct_all[(all_c >= 40) & (all_c < 60)].mean() * 100 if ((all_c >= 40) & (all_c < 60)).sum() > 0 else None,
            "acc_0_40": correct_all[all_c < 40].mean() * 100 if (all_c < 40).sum() > 0 else None,
        }

    return results


# ─── Pretty Printing ─────────────────────────────────────────

def print_header():
    print(f"\n{C.BOLD}{C.CYAN}{'═'*70}")
    print(f"  🔬  TRADING MODEL EVALUATION & DIAGNOSTICS")
    print(f"{'═'*70}{C.END}")
    print(f"  {C.DIM}Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.END}")
    print(f"  {C.DIM}Device: {get_device()}{C.END}")
    print()


def print_model_inventory():
    """Print overview of all saved models."""
    print(f"\n{C.BOLD}{C.BLUE}{'─'*70}")
    print(f"  📦  MODEL INVENTORY")
    print(f"{'─'*70}{C.END}\n")

    all_models = discover_all_models()
    if not all_models:
        print(f"  {C.RED}No models found in {MODELS_DIR}{C.END}")
        return

    for tf in ALL_TIMEFRAMES:
        models = all_models.get(tf, [])
        if not models:
            print(f"  {C.DIM}{tf:>5s}: No models{C.END}")
            continue

        latest = models[-1]
        dt_str = latest["datetime"].strftime("%Y-%m-%d %H:%M") if latest["datetime"] else "Unknown"
        size = latest["size_mb"]
        p = latest["params"]

        arch_str = f"H={p.get('hidden_size','?')} L={p.get('num_layers','?')} heads={p.get('num_heads','?')}"
        cal_str = colorize_bool(p.get("calibration_achieved_80pct", False))
        thresh = p.get("optimal_ratio_threshold", "?")

        print(f"  {C.BOLD}{tf:>5s}{C.END}: {len(models)} model(s)  │  "
              f"Latest: {dt_str}  │  {size:.1f}MB  │  {arch_str}  │  Cal: {cal_str}  │  θ={thresh}")

    # Also check for next_day models
    nd_models = all_models.get("next_day", [])
    if nd_models:
        latest = nd_models[-1]
        dt_str = latest["datetime"].strftime("%Y-%m-%d %H:%M") if latest["datetime"] else "Unknown"
        print(f"  {C.BOLD}{'n_day':>5s}{C.END}: {len(nd_models)} model(s)  │  "
              f"Latest: {dt_str}  │  {latest['size_mb']:.1f}MB  │  {C.DIM}(legacy){C.END}")
    print()


def print_timeframe_detail(result):
    """Print detailed results for one timeframe."""
    if result is None:
        return

    tf = result["timeframe"]
    p = result["params"]
    agg = result.get("aggregate")
    if not agg:
        print(f"  {C.YELLOW}No test data available for {tf}{C.END}")
        return

    print(f"\n{C.BOLD}{C.CYAN}{'─'*70}")
    print(f"  ⏱  TIMEFRAME: {tf.upper()} ({TIMEFRAME_PROFILES[tf]['target_shift']} trading days)")
    print(f"{'─'*70}{C.END}")

    # Model architecture
    print(f"\n  {C.UNDERLINE}Model Architecture{C.END}")
    print(f"    Hidden Size: {p.get('hidden_size', '?'):>6}   │  Layers:  {p.get('num_layers', '?')}")
    print(f"    Attn Heads:  {p.get('num_heads', '?'):>6}   │  Dropout: {p.get('dropout', '?'):.3f}")
    print(f"    Input Size:  {result['input_size']:>6}   │  Dir Weight: {p.get('direction_weight', '?')}")
    print(f"    Dir Scale:   {p.get('direction_scale', '?'):<10}│  Noise Std:  {p.get('noise_std', '?')}")

    # Calibration
    print(f"\n  {C.UNDERLINE}Calibration{C.END}")
    print(f"    Threshold (θ):   {result['opt_threshold']:.6f}")
    print(f"    Achieved ≥80%:   {colorize_bool(result['calibration_80pct'])}")
    print(f"    Median σ (model):{p.get('median_uncertainty', '?')}")

    # Overall accuracy
    print(f"\n  {C.UNDERLINE}Accuracy Summary ({agg['total_samples']} test samples){C.END}")
    overall = agg["overall_accuracy"]
    hc_acc = agg["hc_accuracy"]
    hc_cov = agg["hc_coverage"]
    print(f"    Overall Direction:   {colorize_accuracy(overall)}  {bar_chart(overall)}")
    print(f"    High-Conf (≥75%):    {colorize_accuracy(hc_acc)}  {bar_chart(hc_acc)}  ({agg['hc_count']}/{agg['total_samples']} = {hc_cov:.1f}% coverage)")

    # Confidence bucket breakdown
    print(f"\n  {C.UNDERLINE}Confidence Buckets{C.END}")
    print(f"    {'Bucket':>12s}  │  {'Count':>6s}  │  {'Accuracy':>10s}  │  Distribution")
    print(f"    {'─'*12}──┼──{'─'*6}──┼──{'─'*10}──┼──{'─'*24}")
    buckets = [
        ("90-100%", agg["bucket_90_100"], agg["acc_90_100"]),
        ("80-90%",  agg["bucket_80_90"],  agg["acc_80_90"]),
        ("75-80%",  agg["bucket_75_80"],  agg["acc_75_80"]),
        ("60-75%",  agg["bucket_60_75"],  agg["acc_60_75"]),
        ("40-60%",  agg["bucket_40_60"],  agg["acc_40_60"]),
        ("0-40%",   agg["bucket_0_40"],   agg["acc_0_40"]),
    ]
    total = agg["total_samples"]
    for label, count, acc in buckets:
        acc_str = colorize_accuracy(acc) if acc is not None else f"{C.DIM}  N/A  {C.END}"
        pct = (count / total * 100) if total > 0 else 0
        bar = bar_chart(pct, 100, 20)
        print(f"    {label:>12s}  │  {count:>6d}  │  {acc_str:>20s}  │  {bar} {pct:.1f}%")

    # Ratio distribution
    print(f"\n  {C.UNDERLINE}Uncertainty Ratio Distribution{C.END}")
    print(f"    P25: {agg['ratio_p25']:.4f}  │  P50: {agg['ratio_p50']:.4f}  │  P75: {agg['ratio_p75']:.4f}  │  P95: {agg['ratio_p95']:.4f}")

    # Prediction quality
    print(f"\n  {C.UNDERLINE}Prediction Quality{C.END}")
    print(f"    Mean Pred:   {agg['mean_pred']:>+8.3f}%  │  Std Pred:   {agg['std_pred']:.3f}%")
    print(f"    Mean Actual: {agg['mean_actual']:>+8.3f}%  │  Std Actual: {agg['std_actual']:.3f}%")
    print(f"    MAE:         {agg['mae']:>8.3f}%  │  RMSE:       {agg['rmse']:.3f}%")
    print(f"    Avg Confidence: {agg['mean_confidence']:.1f}%")


def print_ticker_breakdown(result, top_n=10):
    """Print per-ticker accuracy table for a timeframe."""
    if not result or not result.get("ticker_results"):
        return

    tf = result["timeframe"]
    tickers = result["ticker_results"]

    print(f"\n  {C.UNDERLINE}Per-Ticker Breakdown ({tf}){C.END}")
    print(f"    {'Ticker':>8s}  │  {'N':>5s}  │  {'Overall':>8s}  │  {'HC Acc':>8s}  │  {'HC Cov':>7s}  │  {'MAE':>7s}  │  {'Bias':>8s}")
    print(f"    {'─'*8}──┼──{'─'*5}──┼──{'─'*8}──┼──{'─'*8}──┼──{'─'*7}──┼──{'─'*7}──┼──{'─'*8}")

    # Sort by HC accuracy (or overall if no HC)
    sorted_tickers = sorted(tickers.items(), key=lambda x: x[1].get("hc_accuracy", x[1]["overall_accuracy"]), reverse=True)

    for ticker, tr in sorted_tickers[:top_n]:
        overall = tr["overall_accuracy"]
        hc_acc = tr["hc_accuracy"]
        hc_cov = tr["hc_coverage"]
        mae = tr["mae"]
        bias = tr["mean_pred"] - tr["mean_actual"]
        n = tr["n_samples"]
        print(f"    {ticker:>8s}  │  {n:>5d}  │  {colorize_accuracy(overall):>18s}  │  "
              f"{colorize_accuracy(hc_acc):>18s}  │  {hc_cov:>5.1f}%  │  {mae:>6.2f}%  │  {bias:>+7.2f}%")

    if len(sorted_tickers) > top_n:
        print(f"    {C.DIM}... and {len(sorted_tickers) - top_n} more tickers{C.END}")


def print_grand_summary(all_results):
    """Print the top-level summary table across all timeframes."""
    print(f"\n{C.BOLD}{C.GREEN}{'═'*70}")
    print(f"  📊  GRAND SUMMARY — ALL TIMEFRAMES")
    print(f"{'═'*70}{C.END}\n")

    print(f"  {'TF':>5s}  │ {'Shift':>5s} │ {'Samples':>7s} │ {'Overall':>8s} │ {'HC Acc':>8s} │ {'HC Cov':>7s} │ {'MAE':>7s} │ {'Cal80':>6s} │ Status")
    print(f"  {'─'*5}──┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*12}")

    needs_training = []
    good_models = []

    for tf in ALL_TIMEFRAMES:
        result = all_results.get(tf)
        shift = TIMEFRAME_PROFILES[tf]["target_shift"]
        if result is None or "aggregate" not in result:
            print(f"  {C.BOLD}{tf:>5s}{C.END}  │ {shift:>5d} │ {'---':>7s} │ {'---':>8s} │ {'---':>8s} │ {'---':>7s} │ {'---':>7s} │ {'---':>6s} │ {C.RED}NO MODEL{C.END}")
            needs_training.append(tf)
            continue

        agg = result["aggregate"]
        overall = agg["overall_accuracy"]
        hc_acc = agg["hc_accuracy"]
        hc_cov = agg["hc_coverage"]
        mae = agg["mae"]
        cal = result["calibration_80pct"]
        hc_count = agg["hc_count"]

        # Status assessment
        if hc_acc >= 80 and hc_cov >= 5:
            status = f"{C.GREEN}✓ GOOD{C.END}"
            good_models.append(tf)
        elif hc_acc >= 70:
            status = f"{C.YELLOW}◐ CLOSE{C.END}"
            needs_training.append(tf)
        else:
            status = f"{C.RED}✗ TRAIN{C.END}"
            needs_training.append(tf)

        cal_str = f"{C.GREEN}✓{C.END}" if cal else f"{C.RED}✗{C.END}"

        print(f"  {C.BOLD}{tf:>5s}{C.END}  │ {shift:>5d} │ {agg['total_samples']:>7d} │ "
              f"{colorize_accuracy(overall):>18s} │ {colorize_accuracy(hc_acc):>18s} │ "
              f"{hc_cov:>5.1f}% │ {mae:>6.2f}% │ {cal_str:>15s} │ {status}")

    # Recommendations
    print(f"\n{C.BOLD}{C.YELLOW}{'─'*70}")
    print(f"  💡  RECOMMENDATIONS")
    print(f"{'─'*70}{C.END}")

    if good_models:
        print(f"\n  {C.GREEN}✓ Models meeting target (≥80% HC accuracy):{C.END}")
        for tf in good_models:
            agg = all_results[tf]["aggregate"]
            print(f"    • {tf}: {agg['hc_accuracy']:.1f}% accuracy, {agg['hc_coverage']:.1f}% coverage")

    if needs_training:
        print(f"\n  {C.RED}✗ Models needing improvement:{C.END}")
        for tf in needs_training:
            result = all_results.get(tf)
            if result and "aggregate" in result:
                agg = result["aggregate"]
                p = result["params"]
                gap = 80.0 - agg["hc_accuracy"]
                print(f"    • {tf}: {agg['hc_accuracy']:.1f}% HC accuracy (need +{gap:.1f}pp)")
                print(f"      Direction weight: {p.get('direction_weight', '?')}, "
                      f"Direction scale: {p.get('direction_scale', '?')}, "
                      f"Hidden: {p.get('hidden_size', '?')}")
            else:
                print(f"    • {tf}: No model found — needs training from scratch")

    print()


def print_training_timeline():
    """Print a timeline of all training runs."""
    print(f"\n{C.BOLD}{C.BLUE}{'─'*70}")
    print(f"  📅  TRAINING TIMELINE (Recent)")
    print(f"{'─'*70}{C.END}\n")

    all_models = discover_all_models()
    all_runs = []
    for tf, models in all_models.items():
        for m in models:
            if m["datetime"]:
                all_runs.append((m["datetime"], tf, m["size_mb"], m["params"]))

    all_runs.sort(key=lambda x: x[0], reverse=True)

    for dt, tf, size, params in all_runs[:15]:
        hs = params.get("hidden_size", "?")
        cal = "✓" if params.get("calibration_achieved_80pct", False) else "✗"
        thresh = params.get("optimal_ratio_threshold", "?")
        if isinstance(thresh, float):
            thresh = f"{thresh:.4f}"
        print(f"  {dt.strftime('%Y-%m-%d %H:%M')}  │  {tf:>5s}  │  {size:>5.1f}MB  │  H={hs}  │  Cal={cal}  │  θ={thresh}")

    if len(all_runs) > 15:
        print(f"  {C.DIM}... {len(all_runs) - 15} older runs not shown{C.END}")
    print()


def export_json_report(all_results, output_path):
    """Export results as JSON for programmatic use."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "device": str(get_device()),
        "timeframes": {},
    }
    for tf, result in all_results.items():
        if result and "aggregate" in result:
            entry = {
                "aggregate": result["aggregate"],
                "params": {k: v for k, v in result["params"].items()},
                "opt_threshold": result["opt_threshold"],
                "calibration_80pct": result["calibration_80pct"],
                "ticker_results": result.get("ticker_results", {}),
            }
            # Clean up non-serializable items
            for ticker_data in entry["ticker_results"].values():
                for k, v in ticker_data.items():
                    if isinstance(v, (np.floating, np.integer)):
                        ticker_data[k] = float(v)
            report["timeframes"][tf] = entry

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  {C.DIM}JSON report saved to: {output_path}{C.END}")


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading Model Evaluation & Diagnostics")
    parser.add_argument("--ticker", type=str, default="VOO", help="Ticker to evaluate (default: VOO)")
    parser.add_argument("--all-tickers", action="store_true", help="Evaluate across all tickers")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer MC samples (10 vs 30)")
    parser.add_argument("--timeframe", type=str, default=None, help="Evaluate a single timeframe only")
    parser.add_argument("--no-details", action="store_true", help="Skip per-timeframe detailed output")
    parser.add_argument("--json", type=str, default=None, help="Export results to JSON file")
    parser.add_argument("--mc-samples", type=int, default=30, help="Number of MC Dropout samples (default: 30)")
    args = parser.parse_args()

    print_header()
    print_model_inventory()
    print_training_timeline()

    device = get_device()
    n_mc = 10 if args.quick else args.mc_samples
    tickers = ALL_TICKERS if args.all_tickers else [args.ticker]
    timeframes = [args.timeframe] if args.timeframe else ALL_TIMEFRAMES

    print(f"  {C.BOLD}Evaluating:{C.END} {', '.join(tickers) if len(tickers) <= 5 else f'{len(tickers)} tickers'}")
    print(f"  {C.BOLD}Timeframes:{C.END} {', '.join(timeframes)}")
    print(f"  {C.BOLD}MC Samples:{C.END} {n_mc}")
    print()

    # Check for preprocessed data
    has_data = False
    for tf in timeframes:
        for t in tickers:
            if (DATA_DIR / f"{t}_X_test_{tf}.npy").exists():
                has_data = True
                break
        if has_data:
            break

    if not has_data:
        print(f"\n  {C.RED}{'═'*50}")
        print(f"  ⚠  NO PREPROCESSED TEST DATA FOUND!")
        print(f"  {'═'*50}{C.END}")
        print(f"\n  Run preprocessing first:")
        print(f"  {C.CYAN}cd '{PROJECT_ROOT}'")
        print(f"  source venv/bin/activate")
        print(f"  python -m src.data.preprocess{C.END}\n")
        return

    all_results = {}
    for i, tf in enumerate(timeframes):
        progress = f"[{i+1}/{len(timeframes)}]"
        print(f"  {C.CYAN}⏳ {progress} Evaluating {tf}...{C.END}", end="", flush=True)
        t0 = time.time()

        result = evaluate_timeframe(tf, tickers, device, n_mc)
        elapsed = time.time() - t0
        all_results[tf] = result

        if result and "aggregate" in result:
            agg = result["aggregate"]
            acc = agg["overall_accuracy"]
            hc = agg["hc_accuracy"]
            print(f"\r  {C.GREEN}✓ {progress} {tf:>5s}{C.END}  Overall: {colorize_accuracy(acc)}  "
                  f"HC: {colorize_accuracy(hc)}  ({elapsed:.1f}s)")
        else:
            print(f"\r  {C.RED}✗ {progress} {tf:>5s} — No model or data{C.END}  ({elapsed:.1f}s)")

    # Detailed per-timeframe output
    if not args.no_details:
        for tf in timeframes:
            result = all_results.get(tf)
            if result:
                print_timeframe_detail(result)
                if args.all_tickers:
                    print_ticker_breakdown(result, top_n=15)
                elif len(tickers) > 1:
                    print_ticker_breakdown(result, top_n=len(tickers))

    # Grand summary
    print_grand_summary(all_results)

    # JSON export
    if args.json:
        export_json_report(all_results, args.json)
    else:
        # Always save to a default location
        default_json = PROJECT_ROOT / "evaluation_report.json"
        export_json_report(all_results, default_json)

    print(f"  {C.BOLD}{C.GREEN}Done!{C.END}\n")


if __name__ == "__main__":
    main()
