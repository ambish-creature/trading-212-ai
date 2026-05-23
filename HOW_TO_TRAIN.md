# 🌲 How to Train & Analyze the Trading 212 AI Model

A complete guide to running the full data pipeline, training, backtesting,
and the self-tuning auto-training loop. All commands run from the project root.

---

## ⚙️ Setup

```bash
# Activate your virtual environment
source venv/bin/activate

# Install / upgrade all dependencies (do this once after pulling changes)
pip install -r requirements.txt
```

---

## 🛠️ Step-by-Step Command Pipeline

Run these commands in order:

```
fetch.py  →  fetch_macro.py  →  fetch_fundamentals.py  →  fetch_sentiment.py
    │                 │                   │                        │
    ▼                 ▼                   ▼                        ▼
data/raw/       data/macro/       data/fundamentals/      data/sentiment/
    │                                                             │
    └──────────────── preprocess.py ─────────────────────────────┘
                           │
                           ▼
                    data/processed/
                           │
                     train.py / auto_train.py
                           │
                    models/saved/*.pt
                           │
                      backtest.py
```

### Step 1 — Fetch OHLCV + GBP/USD FX data
```bash
python src/data/fetch.py
```
Downloads 10 years of daily price + dividend data for all 12 assets, plus
the daily GBP/USD exchange rate (used to convert USD assets to GBP).

*Output*: `data/raw/<TICKER>.csv`, `data/raw/GBPUSD.csv`

### Step 2 — Fetch Macroeconomic data
```bash
python src/data/fetch_macro.py
```
Downloads from FRED (St. Louis Fed free API, no key needed):
- US Federal Funds Rate, 10-Year Treasury yield, CPI, GDP growth, Unemployment
- Crude oil (WTI) and Gold prices via Yahoo Finance

*Output*: `data/macro/*.csv`, `data/macro/macro_combined.csv`

### Step 3 — Fetch Fundamental ratios
```bash
python src/data/fetch_fundamentals.py
```
Downloads per-ticker P/E, P/B, ROE, D/E, EPS, Revenue Growth, Dividend Yield
etc. from Yahoo Finance. ETFs have fewer ratios (filled with 0 for neutrality).

*Output*: `data/fundamentals/<TICKER>_fundamentals.csv`

### Step 4 — Fetch Sentiment data
```bash
python src/data/fetch_sentiment.py
```
Downloads analyst Buy/Hold/Sell ratings → converts to 1–5 score,
scores recent news headlines with VADER sentiment, fetches institutional %.

*Output*: `data/sentiment/<TICKER>_sentiment.csv`

### Step 5 — Preprocess & build feature matrix
```bash
python src/data/preprocess.py
```
Combines all data sources into a unified feature matrix (~43 features per day):
- OHLCV + Technical: 19 features (SMA, EMA, RSI, MACD, BB, ATR, etc.)
- Macroeconomic: 7 features (rates, CPI, GDP, oil, gold)
- FX Rate: 1 (GBP/USD)
- Fundamentals: 10 (PE, PB, ROE, etc.)
- Sentiment: 3 (analyst score, news sentiment, institutional %)
- Category one-hot: 3 (ETF / Tech / Consumer)

Fits a single global StandardScaler **only on training data** (no leakage).

*Output*: `data/processed/X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`,
`<TICKER>_X_test.npy`, `<TICKER>_y_test.npy`, `scaler.pkl`, `feature_cols.pkl`

### Step 6a — Manual training run
```bash
python src/models/train.py
```
Runs 3 Optuna trials → trains final LSTM-Attention model with asymmetric loss.

*Output*: `models/saved/model_next_day_<timestamp>.pt` + `_params.json`

### Step 6b — Self-tuning auto-training (RECOMMENDED)
```bash
python src/models/auto_train.py --max-cycles 10 --target-multiplier 1.25
```
Automatically runs up to 10 cycles of: train → backtest → compare vs AER.
Stops early when the bot beats 1.25× the savings account interest,
confirmed in 3 rolling validation windows (not just lucky on one period).

Options:
- `--max-cycles N` — maximum number of retrain cycles (default: 10)
- `--target-multiplier X` — target return multiple over AER (default: 1.25)
- `--min-cycles N` — minimum cycles before early-stop check (default: 3)

### Step 7 — Run backtest
```bash
# All assets
python src/trading/backtest.py --ticker all --strategy advanced_ai

# Single asset
python src/trading/backtest.py --ticker TSLA --strategy advanced_ai
```

---

## 📊 Reading Backtest Output

```text
TICKER     | CAT      | FINAL £    | NET PnL              | DRAWDOWN   | DIVS     | INTEREST   | AER BENCH | TRADES
SPY        | ETF      | £427.45    | £+10.12 (+2.43%)     | -0.82 %    | £1.23    | £3.41      | £424.50   | 15B / 23S
...
════════════════════════════════════════════════════════
🏆 COMBINED PORTFOLIO RESULTS (Starting: £5,000.00 GBP)
   • Total Net PnL (Trading): £+84.52 (+1.69%)
   • Total Dividends:         £9.87
   • Total Bank Interest:     £41.23
   • All-in Return:           £+135.62
   ─────────────────────────────────────────────────
   📊 AER Benchmark:          £5,234.10 (savings account return)
   📊 AER Interest:           +£234.10 (4.81% p.a. avg)
   ✅ Bot BEATS the AER benchmark
   🎉 Return is 1.58× the savings account interest!
```

### Key Metrics:
| Metric | What it means |
|--------|--------------|
| `FINAL £` | What this asset's allocation grew to |
| `NET PnL` | Pure trading profit/loss (excluding interest & dividends) |
| `DRAWDOWN` | Worst peak-to-trough drop. 0.00% = bot held cash through crash |
| `DIVS` | Dividend income collected while holding positions |
| `INTEREST` | Bank AER interest earned on free cash each day |
| `AER BENCH` | What £ you'd have if you'd put the money in a savings account |
| `Multiplier` | Combined return ÷ AER interest. Target: ≥ 1.25× |

---

## 🔧 Fine-Tuning Risk vs Reward

In `src/models/train.py`, line ~168:
```python
criterion = AsymmetricGaussianNLLLoss(penalty_factor=3.0)
```
| `penalty_factor` | Behaviour |
|:---:|---|
| `1.0` | Symmetric — model treats upside and downside equally |
| `2.0` | Moderately conservative |
| `3.0` | **Default** — 3× harder penalty for calling a rise on a falling market |
| `5.0+` | Very cautious — will mostly sit in cash during uncertainty |

The `auto_train.py` script automatically adjusts this between 1.0 and 6.0 based on:
- High drawdowns → increase penalty (be safer)
- Near-zero trading returns → decrease penalty (be more active)

---

## 🏦 Cash Reserve Rules

The bot always keeps **20% of current portfolio value** in cash (emergency reserve).
This ensures there is always money to buy into sudden market crashes.

Max spend per single trade = `15% × (free_cash - reserve_floor) × confidence_score`

Example with £5,000 portfolio:
- Reserve floor: £1,000 (20%)
- Available above reserve: £4,000
- Max per trade: £600 × confidence (e.g. 75% → £450 spent)
