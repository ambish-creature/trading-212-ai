# 🌲 How to Train & Analyze the Trading 212 AI Model

This guide teaches you exactly how to fetch fresh data, preprocess features, trigger the Deep Learning model training by hand, and analyze the outputs to assess model performance.

---

## 🛠️ Step-by-Step Execution Commands

To execute a full self-training pipeline manually, run these commands in sequence from your project root directory (ensure your virtual environment is active):

### Step 1: Fetch Fresh Historical Data
This downloads a full, fresh 10-year historical dataset of daily bars and corporate actions (dividends) for all 12 portfolio assets via Yahoo Finance:
```bash
source venv/bin/activate
python src/data/fetch.py
```
*Outputs*: Saved raw CSVs for all assets under `data/raw/` (e.g., `SPY.csv`, `VWRL.L.csv`, `TSLA.csv`).

### Step 2: Preprocess Features & Align Targets
This computes technical indicators, handles dividend scaling, adds category one-hot flags, fits a global StandardScaler, and stacks/shuffles sequences into training, validation, and test splits:
```bash
python src/data/preprocess.py
```
*Outputs*: Saves processed training arrays (`X_train.npy`, `y_train.npy`), validation arrays (`X_val.npy`, `y_val.npy`), individual test npy arrays, and the global `scaler.pkl` under `data/processed/`.

### Step 3: Run Model Retraining & Optuna Tuning
This triggers the hyperparameter tuning search (running 3 automated search trials) and trains the final LSTM-Attention model on the optimal parameters using our custom **Asymmetric Loss Function** (to natively avoid crashes):
```bash
python src/models/train.py
```
*Outputs*: Saves final `.pt` weights and configuration `.json` parameters under `models/saved/` (e.g., `model_next_day_20260523_163517.pt` and `model_next_day_20260523_163517_params.json`).

---

## 📊 How to Analyze the Training Outputs

When running `python src/models/train.py`, the console will output three main sections of feedback. Here is how to read them:

### 1. Optuna Hyperparameter Trials
During the optimization phase, you will see lines like this:
```text
[I 2026-05-23 18:02:23,764] Trial 0 finished with value: 0.8796 and parameters: {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.45, 'lr': 0.0003, 'batch_size': 64}. Best is trial 0 with value: 0.8796.
```
* **`value: 0.8796`**: This is the validation **Asymmetric Gaussian Negative Log-Likelihood (NLL)**. 
  - *Lower is better*. A smaller score indicates that the model's return predictions are highly accurate and its uncertainty estimations are mathematically sound.
  - If a trial fails or overfits, Optuna's pruning mechanism immediately cuts it short to save execution time.

### 2. Early Stopping
During the final model training phase, you will see a printout indicating when training halted:
```text
Early stopping triggered at epoch 30
```
* **Why it triggers**: The final training loop runs for a maximum of 100 epochs, but it monitors validation loss. If the validation loss fails to improve for 15 consecutive epochs, the script terminates training.
* **Why this is good**: It acts as a safety shield, preventing the LSTM from "memorizing" historical details (overfitting) so it can successfully generalize to unseen market dynamics in live trading.

### 3. Asymmetric Loss Behavior
The training utilizes a custom **`AsymmetricGaussianNLLLoss`**.
* **False Positive Penalty**: When the model predicts a price rise (`mu > 0`) but the stock actually drops (`target < 0`), the loss is multiplied by **`3.0x`**.
* **Model Response**: The model's return prediction ($\mu$) will naturally shift downwards under high market volatility. You will notice that the model outputs conservative, negative return forecasts during correction periods, safely keeping your portfolio in cash (false negatives) to avoid catastrophic drops.

---

## 📈 Running and Analyzing Backtests

After training a new model, you can instantly evaluate how it trades historically (Feb 2025 – May 2026) using the **`advanced_ai`** fractional exit strategy:
```bash
python src/trading/backtest.py --ticker all --strategy advanced_ai
```

### Deciphering the Backtest Table
The terminal will display a complete comparative simulation table:
```text
==========================================================================================
TICKER     | CATEGORY   | FINAL VALUE  | NET PROFIT      | MAX DRAWDOWN   | DIVIDENDS   | TRADES
------------------------------------------------------------------------------------------
SPY        | ETF        | $5,083.90    | $+83.90 (+1.68%) | -1.72        % | $6.96       | 196 B / 208 S
TSLA       | Tech       | $5,132.90    | $+132.90 (+2.66%) | -0.79        % | $0.00       | 13 B / 19 S
...
==========================================================================================
🏆 COMBINED MULTI-ASSET PORTFOLIO RESULTS:
   • Total Starting Portfolio:  $60,000.00
   • Total Portfolio Net PnL:   $+623.55 (+1.04%)
   • Total Dividends Collected: $48.90
==========================================================================================
```

#### Metrics Dictionary:
1. **`NET PROFIT`**: Total gain or loss generated by the AI strategy. Look for a positive combined percentage (e.g. `+1.04%`).
2. **`MAX DRAWDOWN`**: The largest peak-to-trough drop in portfolio value during the backtest. 
   - *Analysis*: If drawdowns remain small (e.g. `-1.26%` or `-0.79%`), it proves the model is successfully avoiding holding assets through major drops.
3. **`DIVIDENDS`**: Total cash payouts received from company distributions during holding periods. This is added directly to your cash profit.
4. **`TRADES`**: Total buy (`B`) and sell (`S`) operations executed.
   - *Analysis*: In the `advanced_ai` strategy, you will notice more sells (`S`) than buys (`B`) due to fractional sells (partial profit-taking or time-delayed risk trimming).
