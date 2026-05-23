import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import ACTIVE_TIMEFRAME

# ---------------------------------------------------------------------------
# 1. Model Architecture
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Attention weight computation
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        weights = self.attention(lstm_out) # (batch_size, seq_length, 1)
        weights = self.softmax(weights)
        
        # Multiply weights by lstm_out to get context vector
        context = torch.sum(weights * lstm_out, dim=1) # (batch_size, hidden_size)
        return context

class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_size)
        
        # Head 1: Predicts the mean return (mu)
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Head 2: Predicts the log-uncertainty (log_sigma)
        # Separate head so the model learns WHEN it is uncertain
        self.fc_log_sigma = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Pass sequence through attention
        attn_out = self.attention(lstm_out)
        
        # Two outputs: predicted return and learned uncertainty
        mu = self.fc_mu(attn_out).squeeze(-1)
        log_sigma = self.fc_log_sigma(attn_out).squeeze(-1)
        
        return mu, log_sigma

# ---------------------------------------------------------------------------
# 2. Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ---------------------------------------------------------------------------
# 3. Training Loop & Optuna Objective
# ---------------------------------------------------------------------------

def load_data(data_dir):
    """Loads processed numpy arrays and creates DataLoaders."""
    X_train = torch.tensor(np.load(os.path.join(data_dir, 'X_train.npy')), dtype=torch.float32)
    y_train = torch.tensor(np.load(os.path.join(data_dir, 'y_train.npy')), dtype=torch.float32)
    X_val = torch.tensor(np.load(os.path.join(data_dir, 'X_val.npy')), dtype=torch.float32)
    y_val = torch.tensor(np.load(os.path.join(data_dir, 'y_val.npy')), dtype=torch.float32)
    
    return X_train, y_train, X_val, y_val

def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    
    # Define Hyperparameter Search Space
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Load Data
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/processed/')
    X_train, y_train, X_val, y_val = load_data(data_dir)
    input_size = X_train.shape[2]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # Initialize Model, Loss (Gaussian NLL), and Optimizer
    # Gaussian NLL naturally punishes confident wrong answers heavily:
    #   loss = 0.5 * [log(sigma^2) + (target - mu)^2 / sigma^2]
    model = LSTMAttention(input_size, hidden_size, num_layers, dropout).to(device)
    criterion = nn.GaussianNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning Rate Scheduler (Reduce LR on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=15)
    
    epochs = 100
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            mu, log_sigma = model(batch_X)
            variance = torch.exp(2 * log_sigma)  # sigma^2 = exp(2 * log_sigma)
            loss = criterion(mu, batch_y, variance)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                mu, log_sigma = model(batch_X)
                variance = torch.exp(2 * log_sigma)
                loss = criterion(mu, batch_y, variance)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler and check early stopping
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        
        # Report intermediate values to Optuna for pruning bad trials
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if early_stopping.early_stop:
            break
            
    return early_stopping.best_loss

def train_and_save_final_model(best_params):
    """Trains the model on the optimal parameters and saves a timestamped backup."""
    print("\n--- Training Final Model with Best Parameters ---")
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/processed/')
    X_train, y_train, X_val, y_val = load_data(data_dir)
    input_size = X_train.shape[2]
    
    batch_size = best_params['batch_size']
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    model = LSTMAttention(
        input_size=input_size, 
        hidden_size=best_params['hidden_size'], 
        num_layers=best_params['num_layers'], 
        dropout=best_params['dropout']
    ).to(device)
    
    criterion = nn.GaussianNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)
    
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            mu, log_sigma = model(batch_X)
            variance = torch.exp(2 * log_sigma)
            loss = criterion(mu, batch_y, variance)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                mu, log_sigma = model(batch_X)
                variance = torch.exp(2 * log_sigma)
                val_loss += criterion(mu, batch_y, variance).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    # Save the model and its parameters to create a backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), '../../models/saved/')
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f'model_{ACTIVE_TIMEFRAME}_{timestamp}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Final model successfully backed up to: {model_path}")
    
    params_path = os.path.join(save_dir, f'model_{ACTIVE_TIMEFRAME}_{timestamp}_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"✅ Hyperparameters backed up to: {params_path}")

if __name__ == "__main__":
    print("Starting Hyperparameter Tuning with Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    
    print("\nBest Trial:")
    trial = study.best_trial
    print(f"  Value (Validation Gaussian NLL): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Automatically train and backup the best model
    train_and_save_final_model(trial.params)
