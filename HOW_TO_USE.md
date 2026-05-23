# 🤖 How to Use the AI Finance Advisor

A step-by-step guide to getting personalised stock advice from the AI,
including how to pick stocks, adjust the prediction window, and interpret the output.

---

## 🚀 Quick Start (30 seconds)

```bash
# Activate your environment
source venv/bin/activate

# Get advice on any stock — just type the ticker code
python src/models/advisor.py --ticker AAPL
```

That's it. The AI will fetch live data, run its model, and print its recommendation.

---

## 📋 Full Command Reference

```
python src/models/advisor.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--ticker` | string | *(required)* | Stock/ETF code (see list below) |
| `--horizon` | integer | `5` | How many trading days ahead to predict |
| `--date` | YYYY-MM-DD | today | Reference date (see §3 below) |
| `--holding` | float | `0` | Number of shares you currently hold |
| `--avg-cost` | float | — | Your average cost per share |
| `--no-interactive` | flag | off | Skip the "do you hold this?" prompt |

---

## 1️⃣ Choosing a Stock (Ticker Code)

Pass the **exact ticker code** used on Yahoo Finance / Trading 212.

### Examples by Category

| Stock / ETF | Ticker | Notes |
|------------|--------|-------|
| Apple | `AAPL` | |
| Microsoft | `MSFT` | |
| Tesla | `TSLA` | |
| Google (Alphabet) | `GOOGL` | |
| Meta | `META` | |
| ASML | `ASML` | |
| Samsung | `005930.KS` | Korean Stock Exchange |
| Nvidia | `NVDA` | |
| Amazon | `AMZN` | |
| McDonald's | `MCD` | |
| Costco | `COST` | |
| KFC parent (Yum! Brands) | `YUM` | |
| S&P 500 ETF | `SPY` or `VOO` | |
| Vanguard FTSE All-World | `VWRL.L` | London-listed (GBP) |
| iShares Growth Portfolio | `IWY` | |
| Global X AI ETF | `AIQ` | |
| UK Gilts | `IGLT.L` | London-listed (GBP) |

> **Tip**: The AI was trained on 12 specific assets. For stocks **outside** this set
> (e.g. AAPL, NVDA, AMZN), it will still work — but use the *Unknown* category weights.
> The prediction will be less calibrated than for the trained assets.

---

## 2️⃣ Setting the Prediction Horizon

The `--horizon` argument controls how many **trading days later** the AI predicts.

```bash
# Predict 1 day ahead (next day)
python src/models/advisor.py --ticker AAPL --horizon 1

# Predict 5 days ahead (default, ~1 calendar week)
python src/models/advisor.py --ticker AAPL --horizon 5

# Predict 10 days ahead (~2 calendar weeks)
python src/models/advisor.py --ticker TSLA --horizon 10

# Predict 21 days ahead (~1 calendar month)
python src/models/advisor.py --ticker SPY --horizon 21
```

> **How it works**: The model is trained to predict the 1-day return. For longer horizons,
> it compounds the daily return: `(1 + r_daily)^N - 1`.
> This is a simplification — longer horizons have **much higher uncertainty**.

| Horizon | Recommended use |
|---------|----------------|
| 1–3 days | Short-term trade, swing trade, day-adjacent |
| 5 days | Standard (1 week) — highest confidence |
| 10–15 days | Medium-term view — treat as directional only |
| 21+ days | Long-term trend — very wide uncertainty band |

---

## 3️⃣ Changing the Reference Date

By default, the AI looks at data up to **today** and predicts N days from now.
You can change the reference date with `--date YYYY-MM-DD`.

### Use Case A: Predict from Today (default)
```bash
python src/models/advisor.py --ticker AAPL --horizon 5
```
→ Uses data up to today. Predicts 5 trading days from today.

### Use Case B: Simulate past advice (historical analysis)
```bash
# "What would the AI have advised on 1 March 2025, looking 5 days forward?"
python src/models/advisor.py --ticker TSLA --date 2025-03-01 --horizon 5
```
→ Uses data up to 2025-03-01. Predicts what the price would be on ~2025-03-08.
Great for backtesting your instincts or understanding past decisions.

### Use Case C: Use a specific past date with a different horizon
```bash
# "Look at data from 1 month ago, and predict 6 days after that"
python src/models/advisor.py --ticker MSFT --date 2026-04-23 --horizon 6
```

> **Note**: You cannot set a future reference date (no future data exists).
> For future predictions, use today as the reference date and increase `--horizon`.

---

## 4️⃣ Getting Personalised Advice (with your holdings)

If you already own shares, the advisor adjusts its recommendation based on your
unrealised profit/loss.

### Option A: Interactive (the advisor asks you)
```bash
python src/models/advisor.py --ticker GOOGL --horizon 5
# The advisor will ask:
#   Shares held (e.g. 10.5, or 0): 25
#   Average cost per share (e.g. 150.00): 142.50
```

### Option B: Pass directly on the command line
```bash
# You hold 25 shares of GOOGL, bought at £142.50 average
python src/models/advisor.py --ticker GOOGL --holding 25 --avg-cost 142.50

# You hold 0.5 shares of TSLA (fractional), bought at $200.00
python src/models/advisor.py --ticker TSLA --holding 0.5 --avg-cost 200.00
```

### Option C: Non-interactive mode (for scripts / automation)
```bash
python src/models/advisor.py --ticker SPY --holding 10 --avg-cost 480.00 --no-interactive
```

---

## 5️⃣ Understanding the Output

Here is a complete annotated example output:

```
══════════════════════════════════════════════════════════════
  🤖 AI PERSONAL FINANCE ADVISOR — GOOGL
  📌 Alphabet Inc. | Communication Services
══════════════════════════════════════════════════════════════
  📅 Reference Date:      2026-05-23
  📅 Prediction Horizon:  5 trading day(s) later
  💰 Current Price:       $178.42
  🎯 AI Target Price:     $182.61  (+2.35%)       ← predicted price N days later
  📊 80% Price Range:     $176.14 — $185.33       ← where price is LIKELY to land
  🧠 AI Confidence:       71%  (💪 STRONG)        ← how sure the AI is
  📈 52-Week High/Low:    $208.70 / $140.53
  👨‍💼 Analyst Consensus:   $205.00 (mean target)
  🎁 Dividend Yield:      0.48% p.a.
  ⚡ Beta (Volatility):   1.05  [Medium risk vs market]
──────────────────────────────────────────────────────────────
  📦 YOUR HOLDING:
      25.0000 shares @ $142.50 avg cost
      Current Value:   $4,460.50
      Unrealised P&L:  🟢 +$895.50 (+25.28%)
──────────────────────────────────────────────────────────────

  📋 RECOMMENDATION:

  📈 Signal: BUY / BUY MORE  (STRONG — 71% confidence)

  ✅ You are already holding GOOGL. The AI suggests adding more.
     ➡  Consider buying an additional ~35% of your current holding
        (8.7500 shares ≈ $1,561.18)
     ➡  Timing: Buy within the next 1-2 trading days

  📤 PROFIT-TAKING PLAN (if price rises):
     • Sell ~30% of TOTAL holding when price hits $182.61
       (that's 10.1625 shares ≈ $1,855.26)
     • Sell another 20% after 2 trading days regardless of price
     • Hold remaining position until $185.33 (upper range)

──────────────────────────────────────────────────────────────
  ⚠️  RISK NOTES:
  • Predicted return is equivalent to ~118.5% p.a. (vs bank AER 4.75%).
  • AI accuracy is not guaranteed. Always diversify.
══════════════════════════════════════════════════════════════
  ⚡ This advice is AI-generated. Not financial advice.
     Always do your own research before investing.
══════════════════════════════════════════════════════════════
```

### Decoding each part:

| Section | What it means |
|---------|--------------|
| **AI Target Price** | The price the AI predicts N days from now |
| **80% Price Range** | The range where the price is likely to land (not guaranteed) |
| **AI Confidence** | How certain the model is. Below 45% = don't act on it |
| **52-Week High/Low** | Context: is the stock near its peak or trough? |
| **Analyst Consensus** | What human analysts predict (good cross-reference) |
| **Beta** | Volatility vs market. >1.5 = high risk, <0.8 = low risk |
| **Unrealised P&L** | Your current profit/loss on the position |

---

## 6️⃣ Action Signals Explained

### 📈 BUY / BUY MORE
AI predicts a rise **with ≥45% confidence**.

| Confidence | What it means | Suggested position size |
|-----------|---------------|------------------------|
| ≥70% (STRONG) | Model is very sure | ~15% of available cash |
| 55–69% (MODERATE) | Reasonable conviction | ~10% of available cash |
| 45–54% (WEAK) | Low conviction | ~7% of available cash |

**If you already hold**: AI suggests adding 30–50% more to your position,
with a specific take-profit at the target price and a time-based exit.

### 📉 SELL / DO NOT BUY
AI predicts a drop **with ≥45% confidence**.

| Situation | Advice |
|-----------|--------|
| Holding at profit (>5%) | Sell 60–70% now, hold rest |
| Holding at loss (< -5%) | Cut 50% now, cut rest if drops further |
| Near breakeven | Reduce 50% to protect capital |
| Not holding | Don't open a position, wait for bullish signal |

### ⏸️ HOLD / WAIT
Confidence is below 45%, or the return is near 0%.
The AI is unsure — don't add, don't sell. Re-check in 2-3 days.

---

## 7️⃣ The Timing Rules (When to Buy/Sell)

The advisor uses a **staged** approach — never go all-in or all-out at once:

```
📈 Bullish scenario (you don't hold):
  Day 0 (Today)   → Buy Tranche 1 (50% of planned size)
  Day 2            → Buy Tranche 2 (50%) if price holds or dips
  When target hit  → Sell 30% (lock in profit)
  Day N            → Sell remaining if target not reached

📉 Bearish scenario (you hold with profit):
  Today            → Sell 60-70% immediately
  Day N            → Sell remainder if still bearish signal
```

---

## 8️⃣ Batch: Scan Multiple Stocks at Once

You can write a simple shell script to scan multiple tickers:

```bash
# scan_stocks.sh
for TICKER in AAPL MSFT TSLA NVDA GOOGL AMZN; do
  echo "─────────────────────"
  python src/models/advisor.py --ticker $TICKER --horizon 5 --no-interactive
done
```

Run it:
```bash
chmod +x scan_stocks.sh
./scan_stocks.sh
```

Or a Python loop:
```python
from src.models.advisor import run_advisor

tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
for ticker in tickers:
    pred, advice = run_advisor(
        ticker=ticker,
        horizon_days=5,
        interactive=False
    )
    print(f"{ticker}: {pred['predicted_return']:+.2f}% ({pred['confidence']:.0f}% conf)")
```

---

## 9️⃣ Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `No saved models found` | Run `python src/models/train.py` first |
| `Scaler not found` | Run `python src/data/preprocess.py` first |
| `Not enough data for ticker` | The stock may be too new or delisted. Try a different date. |
| `KeyError: 'Category_Unknown'` | Run preprocess.py to regenerate `feature_cols.pkl` |
| Very low confidence (<30%) | Market is very uncertain. Don't trade on this signal. |

---

## 🔑 Key Principles

1. **Never bet everything on one signal.** The AI is a tool, not an oracle.
2. **Confidence < 45% = don't trade.** The signal is too weak.
3. **Always set a stop-loss** at ~2.5% below your entry price.
4. **Diversify.** The advisor is designed for a portfolio of 10-15 stocks, not single bets.
5. **Re-run regularly.** Markets change daily — run the advisor again after 2-3 days.
6. **The 20% cash rule**: Keep at least 20% of your portfolio in cash for crashes.
