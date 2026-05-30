import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import ACTIVE_TIMEFRAME

# ---------------------------------------------------------------------------
# 1. Model Architecture — Improved LSTMAttention with Multi-Head Attention,
#    Residual Connections, and Layer Normalization
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention over LSTM output sequence.
    Allows the model to attend to multiple different positions simultaneously
    with different learned representations.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        B, T, H = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, H)  # (B, T, H)
        return self.out_proj(context)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network for non-linear feature selection.
    Inspired by Temporal Fusion Transformer. Allows the model to skip
    irrelevant features via learned gates.
    """
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.gate_linear = nn.Linear(input_size, hidden_size)
        self.gate_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Skip connection (projection if sizes differ)
        self.skip = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        # Primary path: ELU activation for non-linearity
        h = self.dropout(F.elu(self.linear1(x)))
        h = self.linear2(h)

        # Gating: sigmoid gate controls how much of h passes through
        gate = torch.sigmoid(self.gate_linear(x))
        gated = gate * h

        # Residual + layer norm
        out = self.gate_norm(gated + self.skip(x))
        return out


class LSTMAttention(nn.Module):
    """
    Improved LSTM + Multi-Head Attention architecture with:
    - Bidirectional LSTM for richer sequence encoding
    - Multi-head self-attention for global temporal dependencies
    - Gated Residual Networks for feature selection
    - Layer Normalization for training stability
    - Residual connections to prevent vanishing gradients
    - Dual output heads: mean (mu) and log-uncertainty (log_sigma)
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_heads=4):
        super(LSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection via GRN for initial feature selection
        self.input_grn = GatedResidualNetwork(input_size, hidden_size, dropout)

        # LSTM backbone (bidirectional for richer context)
        # Note: bidirectional doubles hidden size, so we project back
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        # Project bidirectional output back to hidden_size
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.lstm_dropout = nn.Dropout(dropout)

        # Multi-head self-attention
        # Ensure num_heads divides hidden_size
        safe_num_heads = num_heads
        while hidden_size % safe_num_heads != 0 and safe_num_heads > 1:
            safe_num_heads -= 1
        self.attention = MultiHeadSelfAttention(hidden_size, safe_num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # GRN for post-attention feature selection
        self.post_attn_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)

        # Global pooling: combine CLS-style (last step) + mean pooling
        # Final representation = concat of last-step + mean-pooled
        self.final_grn = GatedResidualNetwork(hidden_size * 2, hidden_size, dropout)

        # Head 1: Predicts the mean return (mu)
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        # Head 2: Predicts the log-uncertainty (log_sigma)
        self.fc_log_sigma = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        B, T, _ = x.shape

        # Input feature selection via GRN
        x_proj = self.input_grn(x)  # (B, T, hidden_size)

        # LSTM encoding (bidirectional)
        lstm_out, _ = self.lstm(x_proj)  # (B, T, hidden_size*2)
        lstm_out = self.lstm_proj(lstm_out)  # (B, T, hidden_size)
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)

        # Multi-head self-attention with residual
        attn_out = self.attention(lstm_out)  # (B, T, hidden_size)
        attn_out = self.attn_norm(attn_out + lstm_out)  # residual

        # Post-attention GRN
        attn_out = self.post_attn_grn(attn_out)  # (B, T, hidden_size)

        # Pooling: last timestep + mean pooling → concat
        last_step = attn_out[:, -1, :]         # (B, hidden_size)
        mean_pool = attn_out.mean(dim=1)        # (B, hidden_size)
        combined = torch.cat([last_step, mean_pool], dim=-1)  # (B, hidden_size*2)

        # Final feature mixing via GRN
        final = self.final_grn(combined)  # (B, hidden_size)

        # Dual output heads
        mu = self.fc_mu(final).squeeze(-1)           # (B,)
        log_sigma = self.fc_log_sigma(final).squeeze(-1)  # (B,)

        return mu, log_sigma


# ---------------------------------------------------------------------------
# 2. Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=25, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    def restore_best_weights(self, model):
        """Restore model to the best validation weights found during training."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# ---------------------------------------------------------------------------
# 2.5 Gaussian NLL Loss (Unbiased, penalty_factor=1.0)
# ---------------------------------------------------------------------------

class AsymmetricGaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss.
    With penalty_factor=1.0 (default), this is the standard unbiased NLL.
    
    Includes a Directional Alignment Penalty (DAP) to force correct directional 
    predictions (sign of mu matches sign of target), directly optimizing directional accuracy.
    """
    def __init__(self, penalty_factor=1.0, direction_weight=0.5, direction_scale=1.0, eps=1e-6):
        super(AsymmetricGaussianNLLLoss, self).__init__()
        self.penalty_factor = penalty_factor
        self.direction_weight = direction_weight
        self.direction_scale = direction_scale
        self.eps = eps

    def forward(self, mu, target, var):
        # 1. Base Gaussian NLL Loss (for magnitude and aleatoric uncertainty prediction)
        base_loss = 0.5 * (torch.log(var + self.eps) + (target - mu)**2 / (var + self.eps))
        
        # 2. Penalty factor for false positives (conservatism)
        if self.penalty_factor != 1.0:
            false_positive = (mu > 0.0) & (target < 0.0)
            reg_loss = torch.where(false_positive, base_loss * self.penalty_factor, base_loss)
        else:
            reg_loss = base_loss
            
        # 3. Directional Alignment Penalty (DAP)
        dap_loss = F.softplus(-mu * target * self.direction_scale)

        
        # Combined multi-task loss
        loss = reg_loss + self.direction_weight * dap_loss
        return torch.mean(loss)


# ---------------------------------------------------------------------------
# 3. Noise Augmentation
# ---------------------------------------------------------------------------

def augment_with_noise(X_batch, noise_std=0.01):
    """
    Adds small Gaussian noise to training inputs for regularization.
    This prevents overfitting and improves generalization.
    Only applied during training, not validation/test.
    """
    noise = torch.randn_like(X_batch) * noise_std
    return X_batch + noise


# ---------------------------------------------------------------------------
# 4. Training Loop & Optuna Objective
# ---------------------------------------------------------------------------

GLOBAL_TIMEFRAME = ACTIVE_TIMEFRAME


def load_data(data_dir, timeframe, device=None):
    """Loads processed numpy arrays and creates DataLoaders."""
    X_train = torch.tensor(np.load(os.path.join(data_dir, f'X_train_{timeframe}.npy')), dtype=torch.float32)
    y_train = torch.tensor(np.load(os.path.join(data_dir, f'y_train_{timeframe}.npy')), dtype=torch.float32)
    X_val   = torch.tensor(np.load(os.path.join(data_dir, f'X_val_{timeframe}.npy')),   dtype=torch.float32)
    y_val   = torch.tensor(np.load(os.path.join(data_dir, f'y_val_{timeframe}.npy')),   dtype=torch.float32)
    if device is not None:
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)
    return X_train, y_train, X_val, y_val


def objective(trial):
    """Optuna objective function for hyperparameter tuning."""

    # Hyperparameter Search Space (expanded)
    hidden_size      = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers       = trial.suggest_int("num_layers", 1, 3)
    dropout          = trial.suggest_float("dropout", 0.1, 0.4)
    lr               = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    batch_size       = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay     = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    noise_std        = trial.suggest_float("noise_std", 0.001, 0.05, log=True)
    num_heads        = trial.suggest_categorical("num_heads", [2, 4, 8])
    direction_weight = trial.suggest_float("direction_weight", 0.4, 2.0)
    direction_scale  = trial.suggest_float("direction_scale", 1.0, 5.0)

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    data_dir   = os.path.join(os.path.dirname(__file__), '../../data/processed/')
    
    # Speedup: Load dataset directly into GPU memory to bypass CPU bottleneck!
    X_train, y_train, X_val, y_val = load_data(data_dir, GLOBAL_TIMEFRAME, device)
    input_size = X_train.shape[2]

    # Data loaders are instant because memory is already on device
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    model     = LSTMAttention(input_size, hidden_size, num_layers, dropout, num_heads).to(device)
    criterion = AsymmetricGaussianNLLLoss(penalty_factor=1.0, direction_weight=direction_weight, direction_scale=direction_scale)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # CosineAnnealingWarmRestarts: better exploration, avoids local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    early_stopping = EarlyStopping(patience=15)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = augment_with_noise(batch_X, noise_std)
            optimizer.zero_grad()
            mu, log_sigma = model(batch_X)
            variance = torch.exp(2 * log_sigma)
            loss = criterion(mu, batch_y, variance)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                mu, log_sigma = model(batch_X)
                variance = torch.exp(2 * log_sigma)
                val_loss += criterion(mu, batch_y, variance).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(epoch + 1)
        early_stopping(avg_val_loss)

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if early_stopping.early_stop:
            break

    return early_stopping.best_loss


def compute_optimal_ratio_threshold(model, val_loader, device, n_mc_samples=30):
    """
    Computes the optimal ratio threshold for selective classification.

    The key insight: we want the SMALLEST ratio threshold r* such that:
      - Predictions where mc_std/model_sigma <= r* have directional accuracy >= 80%
      - At least 5% of samples satisfy this (to avoid cherry-picking)

    This is fundamentally different from finding the LARGEST threshold that fits,
    which was the previous bug.
    """
    model.train()  # Enable MC Dropout
    val_preds, val_mc_stds, val_model_sigmas, val_targets = [], [], [], []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            mus, sigmas = [], []
            for _ in range(n_mc_samples):
                mu_s, log_sig_s = model(batch_X)
                mus.append(mu_s.cpu().numpy())
                sigmas.append(torch.exp(log_sig_s).cpu().numpy())

            mus_arr    = np.array(mus)  # (n_mc, batch)
            sigmas_arr = np.array(sigmas)

            val_preds.extend(np.mean(mus_arr, axis=0).tolist())
            val_mc_stds.extend(np.std(mus_arr, axis=0).tolist())
            val_model_sigmas.extend(np.mean(sigmas_arr, axis=0).tolist())
            val_targets.extend(batch_y.numpy().tolist())

    val_preds        = np.array(val_preds)
    val_mc_stds      = np.array(val_mc_stds)
    val_model_sigmas = np.array(val_model_sigmas)
    val_targets      = np.array(val_targets)

    val_ratios      = val_mc_stds / (val_model_sigmas + 1e-9)
    val_pred_signs  = np.sign(val_preds)
    val_target_signs = np.sign(val_targets)
    correct          = (val_pred_signs == val_target_signs)

    print(f"   📊 Ratio distribution: min={val_ratios.min():.4f} max={val_ratios.max():.4f} "
          f"mean={val_ratios.mean():.4f} median={np.median(val_ratios):.4f}")

    # Compute overall (unfiltered) accuracy
    overall_acc = correct.mean()
    print(f"   📊 Overall unfiltered accuracy: {overall_acc*100:.2f}%")

    # FIXED: Sweep from small to large — find the MINIMUM threshold
    # where (ratio <= threshold) samples achieve >= 80% accuracy.
    # Use percentile-based steps adapted to the actual distribution.
    max_ratio = np.percentile(val_ratios, 95)  # 95th percentile as upper bound
    sweep_values = np.linspace(val_ratios.min() + 1e-6, max_ratio, 200)

    optimal_r = None
    best_coverage = 0.0
    found = False

    # Strategy: find LARGEST r such that samples with ratio <= r have >= 80% accuracy
    # AND enough samples (>= 5% of total)
    min_samples = max(20, int(len(val_targets) * 0.05))

    for r in sweep_values[::-1]:
        mask = val_ratios <= r
        count = np.sum(mask)
        if count >= min_samples:
            acc = correct[mask].mean()
            if acc >= 0.80:
                optimal_r = float(r)
                best_coverage = count / len(val_targets)
                found = True
                print(f"   ✅ Found threshold: {optimal_r:.4f} "
                      f"(Acc={acc*100:.2f}%, Coverage={best_coverage*100:.2f}%)")
                break  # Stop at the LARGEST r that achieves 80%

    if not found:
        # No threshold achieves 80% — find the one with the highest accuracy
        best_acc = 0.0
        for r in sweep_values:
            mask = val_ratios <= r
            count = np.sum(mask)
            if count >= min_samples:
                acc = correct[mask].mean()
                if acc > best_acc:
                    best_acc = acc
                    optimal_r = float(r)
                    best_coverage = count / len(val_targets)

        best_r_str = f"{optimal_r:.4f}" if optimal_r is not None else "N/A"
        print(f"   ⚠️  Could not achieve 80% accuracy. Best: {best_r_str} "
              f"(Acc={best_acc*100:.2f}%, Coverage={best_coverage*100:.2f}%)")

    return optimal_r or float(np.percentile(val_ratios, 30)), found


def train_and_save_final_model(best_params, num_heads=4):
    """Trains the model with the best parameters and saves a timestamped backup."""
    print("\n--- Training Final Model with Best Parameters ---")
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/processed/')
    # Speedup: Load dataset directly into GPU memory to bypass CPU bottleneck!
    X_train, y_train, X_val, y_val = load_data(data_dir, GLOBAL_TIMEFRAME, device)
    input_size = X_train.shape[2]

    batch_size   = best_params['batch_size']
    noise_std    = best_params.get('noise_std', 0.01)
    used_heads   = best_params.get('num_heads', num_heads)

    # Data loaders are instant because memory is already on device
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    model = LSTMAttention(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        num_heads=used_heads
    ).to(device)

    dw = best_params.get('direction_weight', 0.5)
    ds = best_params.get('direction_scale', 1.0)
    criterion = AsymmetricGaussianNLLLoss(penalty_factor=1.0, direction_weight=dw, direction_scale=ds)
    wd        = best_params.get('weight_decay', 1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=wd)

    # CosineAnnealingWarmRestarts for better training dynamics
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Early stopping with longer patience for final model
    early_stopping = EarlyStopping(patience=30)

    # Mixed Precision Training (AMP) for maximum GPU speedup
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)

    epochs = 200
    interrupted = False
    
    print("🚀 Speed Optimization Active: CUDA Mixed Precision (AMP) & Pin Memory.")
    print("🛑 Ctrl+C Intercept Active: You can stop training anytime, best model weights will be saved gracefully!")
    
    try:
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_X = augment_with_noise(batch_X, noise_std)
                optimizer.zero_grad()
                
                # Autocast forward pass under AMP
                with torch.cuda.amp.autocast(enabled=is_cuda):
                    mu, log_sigma = model(batch_X)
                    variance = torch.exp(2 * log_sigma)
                    loss = criterion(mu, batch_y, variance)
                
                # Scaled backprop
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    with torch.cuda.amp.autocast(enabled=is_cuda):
                        mu, log_sigma = model(batch_X)
                        variance = torch.exp(2 * log_sigma)
                        val_loss += criterion(mu, batch_y, variance).item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(epoch + 1)
            early_stopping(avg_val_loss, model)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}. Restoring best weights.")
                early_stopping.restore_best_weights(model)
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Training gracefully interrupted by user (Ctrl+C). Saving the best model weights obtained so far...")
        early_stopping.restore_best_weights(model)
        interrupted = True

    # Compute median uncertainty
    model.eval()
    val_uncertainties = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device)
            with torch.cuda.amp.autocast(enabled=is_cuda):
                _, log_sigma = model(batch_X)
                sigma = torch.exp(log_sigma)
            val_uncertainties.extend(sigma.cpu().numpy().tolist())
    median_u = float(np.median(val_uncertainties))
    best_params['median_uncertainty'] = median_u
    print(f"📊 Computed Validation Median Uncertainty: {median_u:.4f}")

    # FIXED Optimal Ratio Threshold Auto-Calibration
    print("📐 Auto-Calibrating Optimal Ratio Threshold on Validation Set...")
    optimal_r, calibration_found = compute_optimal_ratio_threshold(model, val_loader, device)
    best_params['optimal_ratio_threshold'] = optimal_r
    best_params['calibration_achieved_80pct'] = calibration_found
    best_params['training_interrupted'] = interrupted

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if interrupted:
        timestamp += "_interrupted"
    save_dir  = os.path.join(os.path.dirname(__file__), '../../models/saved/')
    os.makedirs(save_dir, exist_ok=True)

    model_path  = os.path.join(save_dir, f'model_{GLOBAL_TIMEFRAME}_{timestamp}.pt')
    params_path = os.path.join(save_dir, f'model_{GLOBAL_TIMEFRAME}_{timestamp}_params.json')

    torch.save(model.state_dict(), model_path)
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"\n✅ Final model saved: {model_path}")
    print(f"✅ Parameters saved:  {params_path}")
    return model_path, params_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Improved LSTMAttention Trainer v2.0.")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Specific timeframe/horizon to train (e.g. 1mo, 3mo).")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of Optuna trials (default: 20). Set to 0 to train directly.")
    args = parser.parse_args()

    if args.timeframe:
        GLOBAL_TIMEFRAME = args.timeframe

    if args.trials == 0:
        print(f"Bypassing Optuna. Training with robust defaults for timeframe: {GLOBAL_TIMEFRAME}...")
        default_params = {
            "hidden_size": 128,
            "num_layers":  2,
            "dropout":     0.2,
            "lr":          2e-4,
            "batch_size":  64,
            "weight_decay": 1e-5,
            "noise_std":   0.01,
            "num_heads":   4,
            "direction_weight": 0.5,
            "direction_scale":  1.0,
        }
        train_and_save_final_model(default_params, num_heads=4)
    else:
        print(f"Starting Optuna Hyperparameter Tuning ({args.trials} trials) for timeframe: {GLOBAL_TIMEFRAME}...")
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
        study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        
        # Parallel Execution: Run multiple trials in parallel on the GPU to utilize RTX 4080 compute power!
        n_jobs = 4 if torch.cuda.is_available() else 1
        print(f"🚀 Speedup Active: Running {n_jobs} trials concurrently.")
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False, n_jobs=n_jobs)

        print("\nBest Trial:")
        trial = study.best_trial
        print(f"  Value (Validation Gaussian NLL): {trial.value:.6f}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        train_and_save_final_model(trial.params, num_heads=trial.params.get('num_heads', 4))
