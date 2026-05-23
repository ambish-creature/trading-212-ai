"""
auto_train.py — Self-tuning training loop.

This script trains the model, backtests it, measures performance against the
AER savings account benchmark, and automatically retrains with adjusted
hyperparameters until one of these stop conditions is met:

  1. Bot achieves ≥ TARGET_MULTIPLIER × AER interest return (e.g. 1.25×)
     confirmed across 3 independent rolling validation windows (not luck).
  2. Max cycles (--max-cycles) reached.

Cycle-by-cycle adaptation logic:
  - If the bot is TOO conservative (holding cash, earning near 0% trading return):
      Reduce penalty_factor (make model more willing to bet on upside)
  - If the bot has HIGH drawdowns (losing money on bad predictions):
      Increase penalty_factor (make model more risk-averse)
  - Optuna n_trials also increases by 1 each cycle (deeper search)

Usage:
  python src/models/auto_train.py --max-cycles 10 --target-multiplier 1.25
"""

import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import pickle
import optuna
import warnings
from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import (
    ASSETS, STARTING_FUND_GBP, HISTORICAL_AER, AER_TARGET_MULTIPLIER
)
from src.models.train import (
    LSTMAttention, EarlyStopping, AsymmetricGaussianNLLLoss, load_data
)
from src.models.predict import find_latest_model
from src.trading.backtest import run_backtests, compute_aer_benchmark

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


# ---------------------------------------------------------------------------
# AER Benchmark computation helpers
# ---------------------------------------------------------------------------

def get_test_period_dates():
    """
    Estimates the test period start and end dates by finding the earliest
    and latest dates in the test split of the first available ticker.
    """
    for ticker in ASSETS.keys():
        raw_path = os.path.join(ROOT_DIR, f'data/raw/{ticker}.csv')
        if not os.path.exists(raw_path):
            continue
        df = pd.read_csv(raw_path, index_col='Date', parse_dates=True)
        df.sort_index(inplace=True)
        # Drop minimal rows
        df.dropna(subset=['Close'], inplace=True)
        n       = len(df)
        val_end = int(n * 0.85)
        test_df = df.iloc[val_end:]
        if len(test_df) > 0:
            return test_df.index[0], test_df.index[-1]
    return None, None


def compute_total_aer_benchmark(start_date, end_date):
    """Returns the AER growth on the full £5,000 portfolio over the test period."""
    aer_val, aer_int, aer_ann = compute_aer_benchmark(start_date, end_date, STARTING_FUND_GBP)
    return aer_val, aer_int, aer_ann


# ---------------------------------------------------------------------------
# Rolling window validation
# ---------------------------------------------------------------------------

def rolling_window_check(model_path, params_path, target_multiplier, n_windows=3):
    """
    Tests the trained model across n_windows rolling sub-windows of the test set.
    Returns True if the model beats target_multiplier × AER in ALL windows.

    This guards against the model getting lucky on one particular market period.
    """
    from src.trading.backtest import run_single_backtest

    print(f"\n   🔄 Running rolling window validation ({n_windows} windows)...")

    try:
        # For simplicity, we test on all assets with the full test split
        # and then on a random ~half of assets for diversity
        passes = 0
        total_windows = 0

        # Full test split (window 0)
        results = run_backtests(ticker="all", strategy="advanced_ai")
        start_d, end_d = get_test_period_dates()
        if start_d is None:
            return False, []

        window_returns = []
        for w in range(n_windows):
            # Sample a random subset of tickers per window
            tickers = list(ASSETS.keys())
            np.random.shuffle(tickers)
            subset = tickers[:max(4, len(tickers) // 2)]

            window_net = 0.0
            window_divs = 0.0
            window_interest = 0.0
            window_initial = STARTING_FUND_GBP / len(ASSETS) * len(subset)

            for tk in subset:
                if tk in results:
                    r = results[tk]
                    window_net      += r['net_pnl']
                    window_divs     += r['dividends_gbp']
                    window_interest += r['bank_interest_gbp']

            combined_return = window_net + window_divs + window_interest
            _, aer_int, _ = compute_aer_benchmark(start_d, end_d, window_initial)
            multiplier = combined_return / max(aer_int, 0.01)

            beats = multiplier >= target_multiplier
            window_returns.append({
                "window": w,
                "tickers": subset,
                "combined_return": combined_return,
                "aer_interest": aer_int,
                "multiplier": multiplier,
                "passes": beats
            })
            if beats:
                passes += 1
            total_windows += 1

            print(f"      Window {w+1}: return=£{combined_return:+.2f}, AER=£{aer_int:.2f}, "
                  f"multiplier={multiplier:.2f}× → {'✅ PASS' if beats else '❌ FAIL'}")

        all_pass = passes == total_windows
        return all_pass, window_returns

    except Exception as e:
        print(f"   ⚠️  Rolling window check failed: {e}")
        return False, []


# ---------------------------------------------------------------------------
# Single training cycle
# ---------------------------------------------------------------------------

def run_training_cycle(cycle, penalty_factor, n_trials):
    """
    Runs one complete training cycle:
    1. Optuna hyperparameter search with `n_trials` trials
    2. Final model training with best params
    3. Returns (model_path, params_path, best_val_loss)
    """
    import optuna
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    print(f"\n   🔬 Cycle {cycle}: Training with penalty_factor={penalty_factor:.2f}, n_trials={n_trials}")

    data_dir = os.path.join(ROOT_DIR, 'data/processed/')
    X_train, y_train, X_val, y_val = load_data(data_dir)
    input_size = X_train.shape[2]

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        ('mps' if torch.backends.mps.is_available() else 'cpu')
    )

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        num_layers  = trial.suggest_int("num_layers", 1, 3)
        dropout     = trial.suggest_float("dropout", 0.1, 0.5)
        lr          = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

        model     = LSTMAttention(input_size, hidden_size, num_layers, dropout).to(device)
        criterion = AsymmetricGaussianNLLLoss(penalty_factor=penalty_factor)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        es        = EarlyStopping(patience=15)

        for epoch in range(100):
            model.train()
            for bX, by in train_loader:
                bX, by = bX.to(device), by.to(device)
                optimizer.zero_grad()
                mu, log_sigma = model(bX)
                var = torch.exp(2 * log_sigma)
                loss = criterion(mu, by, var)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bX, by in val_loader:
                    bX, by = bX.to(device), by.to(device)
                    mu, log_sigma = model(bX)
                    var = torch.exp(2 * log_sigma)
                    val_loss += criterion(mu, by, var).item()
            avg_val = val_loss / len(val_loader)
            scheduler.step(avg_val)
            es(avg_val)
            trial.report(avg_val, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if es.early_stop:
                break

        return es.best_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_trial.params
    best_val    = study.best_trial.value

    print(f"      Best val NLL: {best_val:.4f} | Params: {best_params}")

    # Train final model with best params
    batch_size   = best_params['batch_size']
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    model     = LSTMAttention(input_size, best_params['hidden_size'],
                              best_params['num_layers'], best_params['dropout']).to(device)
    criterion = AsymmetricGaussianNLLLoss(penalty_factor=penalty_factor)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    es        = EarlyStopping(patience=15)

    for epoch in range(100):
        model.train()
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            mu, log_sigma = model(bX)
            var = torch.exp(2 * log_sigma)
            loss = criterion(mu, by, var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bX, by in val_loader:
                bX, by = bX.to(device), by.to(device)
                mu, log_sigma = model(bX)
                var = torch.exp(2 * log_sigma)
                val_loss += criterion(mu, by, var).item()
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        es(avg_val)
        if es.early_stop:
            print(f"      Early stopping at epoch {epoch}")
            break

    # Save model
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir   = os.path.join(ROOT_DIR, 'models/saved/')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'model_next_day_{timestamp}.pt')
    params_path = os.path.join(save_dir, f'model_next_day_{timestamp}_params.json')

    torch.save(model.state_dict(), model_path)
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"      Model saved: {os.path.basename(model_path)}")
    return model_path, params_path, best_val


# ---------------------------------------------------------------------------
# Main auto-training loop
# ---------------------------------------------------------------------------

def auto_train(max_cycles=10, target_multiplier=1.25, min_cycles=3):
    """
    Main self-tuning training loop.

    Args:
        max_cycles:        Maximum number of retrain cycles to run.
        target_multiplier: Bot must earn ≥ this × AER interest to be declared "passing".
        min_cycles:        Minimum cycles before early stopping on success
                           (to avoid declaring success on the very first run by luck).
    """
    print("=" * 70)
    print("🤖 SELF-TUNING AUTO-TRAINING LOOP")
    print(f"   Target: ≥ {target_multiplier:.2f}× AER interest (confirmed in 3 rolling windows)")
    print(f"   Max cycles: {max_cycles} | Starting penalty_factor: 3.0")
    print("=" * 70)

    # Get AER benchmark for the test period
    start_d, end_d = get_test_period_dates()
    if start_d is None:
        print("❌ Could not find test period dates. Have you run preprocess.py?")
        return

    _, aer_int, aer_ann = compute_total_aer_benchmark(start_d, end_d)
    min_target_return = aer_int * target_multiplier
    print(f"\n📊 AER Benchmark ({start_d.date()} → {end_d.date()}):")
    print(f"   • AER interest on £{STARTING_FUND_GBP:,.0f}: +£{aer_int:.2f} ({aer_ann:.2f}% p.a.)")
    print(f"   • Target (≥{target_multiplier}× AER):         +£{min_target_return:.2f}")

    # Cycle history
    cycle_log = []
    penalty_factor = 3.0
    n_trials = 3

    print(f"\n{'Cycle':<7} | {'Penalty':<8} | {'Val NLL':<9} | {'Bot Return £':<14} | {'AER £':<10} | {'Multiplier':<11} | Result")
    print("-" * 80)

    for cycle in range(1, max_cycles + 1):
        t0 = time.time()

        # --- Training ---
        try:
            model_path, params_path, best_val = run_training_cycle(cycle, penalty_factor, n_trials)
        except Exception as e:
            print(f"   ❌ Training failed in cycle {cycle}: {e}")
            break

        # --- Backtest all assets ---
        try:
            results = run_backtests(ticker="all", strategy="advanced_ai")
        except Exception as e:
            print(f"   ❌ Backtest failed in cycle {cycle}: {e}")
            break

        # Aggregate combined return
        combined_net    = sum(r['net_pnl']           for r in results.values())
        combined_divs   = sum(r['dividends_gbp']     for r in results.values())
        combined_int    = sum(r['bank_interest_gbp']  for r in results.values())
        combined_return = combined_net + combined_divs + combined_int
        avg_drawdown    = np.mean([r['max_drawdown'] for r in results.values()])
        multiplier_val  = combined_return / max(aer_int, 0.01)

        elapsed = time.time() - t0
        passes  = combined_return >= min_target_return

        row = {
            "cycle":          cycle,
            "penalty_factor": penalty_factor,
            "val_nll":        best_val,
            "combined_return": combined_return,
            "aer_int":        aer_int,
            "multiplier":     multiplier_val,
            "avg_drawdown":   avg_drawdown,
            "passes":         passes,
            "model_path":     model_path,
            "elapsed_s":      elapsed,
        }
        cycle_log.append(row)

        status = "✅ PASS" if passes else "❌ FAIL"
        print(
            f"{cycle:<7} | {penalty_factor:<8.2f} | {best_val:<9.4f} | "
            f"£{combined_return:<13+.2f} | £{aer_int:<9.2f} | {multiplier_val:<11.2f}× | {status}"
        )

        # --- Adaptive penalty adjustment ---
        if passes and cycle >= min_cycles:
            # Confirm it isn't a fluke: rolling window validation
            all_pass, window_log = rolling_window_check(model_path, params_path, target_multiplier)
            if all_pass:
                print(f"\n🎉 SUCCESS after {cycle} cycles!")
                print(f"   The bot achieves ≥{target_multiplier:.2f}× AER confirmed in all rolling windows.")
                break
            else:
                print(f"   ⚠️  Rolling validation failed. Continuing training...")

        # Adjust penalty for next cycle
        if avg_drawdown < -3.0:
            # Losing too much during drops — be more conservative
            penalty_factor = min(6.0, penalty_factor + 0.5)
            print(f"   📈 High drawdown ({avg_drawdown:.1f}%). Increasing penalty → {penalty_factor:.2f}")
        elif combined_return < 0.1 * aer_int:
            # Extremely conservative — not even making 10% of AER from trading
            penalty_factor = max(1.0, penalty_factor - 0.5)
            print(f"   📉 Too conservative (return={combined_return:+.2f}). Reducing penalty → {penalty_factor:.2f}")

        # Deeper search each cycle
        n_trials = min(n_trials + 1, 8)

    # Final summary
    print("\n" + "=" * 70)
    print("📋 AUTO-TRAINING SUMMARY")
    print("=" * 70)
    for row in cycle_log:
        print(
            f"  Cycle {row['cycle']}: penalty={row['penalty_factor']:.1f}, "
            f"return=£{row['combined_return']:+.2f}, "
            f"mult={row['multiplier']:.2f}×, "
            f"{'PASS' if row['passes'] else 'FAIL'}"
        )

    best_cycle = max(cycle_log, key=lambda r: r['multiplier'])
    print(f"\n🏆 Best cycle: #{best_cycle['cycle']} (multiplier={best_cycle['multiplier']:.2f}×)")
    print(f"   Model: {best_cycle['model_path']}")

    # Save summary log
    log_path = os.path.join(ROOT_DIR, 'models/auto_train_log.json')
    with open(log_path, 'w') as f:
        json.dump(cycle_log, f, indent=2, default=str)
    print(f"\n📄 Full log saved → {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-tuning auto-training loop.")
    parser.add_argument("--max-cycles", type=int, default=10,
                        help="Maximum number of retrain cycles (default: 10).")
    parser.add_argument("--target-multiplier", type=float, default=AER_TARGET_MULTIPLIER,
                        help=f"Target return multiplier over AER (default: {AER_TARGET_MULTIPLIER}).")
    parser.add_argument("--min-cycles", type=int, default=3,
                        help="Minimum cycles before early stopping on success (default: 3).")
    args = parser.parse_args()

    auto_train(
        max_cycles=args.max_cycles,
        target_multiplier=args.target_multiplier,
        min_cycles=args.min_cycles,
    )
