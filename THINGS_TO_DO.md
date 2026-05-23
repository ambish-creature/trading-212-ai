# 🌲 Model & System Roadmap: Completed & Next Steps

This file tracks the development progress, completed features, and future optimization ideas for our Trading 212 AI bot.

---

## 🏆 Completed Enhancements

We have successfully implemented and verified several advanced Deep Learning and system features:

* **[x] Multi-Asset Scale & Generalization**: Replaced the single-stock baseline with a diverse 12-asset portfolio across three sectors (ETFs, Tech, Consumer) to dramatically mitigate overfitting and capture broader market trends.
* **[x] Conditional Inputs (Categorical Learning)**: Implemented a unified **Conditional LSTM Model** that dynamically learns distinct behaviors for different asset classes by feeding a 3-dimensional one-hot encoded vector representing the asset's category (`ETF`, `Tech`, `Consumer`).
* **[x] Attention Mechanism**: Equipped the LSTM with an active **Attention Block** that automatically learns to focus on the most important historical trading days in the input sequence, improving prediction accuracy.
* **[x] Real-time Dividend Payout & Tracking**: Integrated historical dividends into data fetching/preprocessing as a feature, and added daily dividend payout tracking (`qty * dividend_per_share` added to available cash) in both the backtest simulation and active live trading engine.
* **[x] Robust Regularization & Early Stopping**: Added dropout layers (configured up to 50%), gradient clipping, dynamic learning rate plateau scheduler, and an automated **Early Stopping** class to halt training when validation NLL loss stops improving.
* **[x] Unified Data Scaling**: Preprocessed all assets under a single global `StandardScaler` fitted on stacked training splits, ensuring robust and uniform indicator scaling while preserving one-hot categorical flags.
* **[x] 24/7 Production Scheduler**: Added an autonomous `orchestrator.py` that schedules Nasdaq-Open execution loops (9:35 AM EST, Mon–Fri) and weekly model retraining (12:00 AM Saturdays).

---

## 🚀 Future Roadmap & Model Improvement Ideas

* **[ ] Bidirectional & Hybrid Architectures**: Evaluate bidirectional LSTM (`Bi-LSTM`) layers and hybrids that combine attention with multilayer perceptron (MLP) heads. This can capture both forward and backward temporal dependencies.
* **[ ] Robust Time-Series Cross-Validation**: Implement k-fold cross-validation using rolling time windows (walk-forward validation) to obtain highly reliable, unbiased historical performance estimates.
* **[ ] Feature Data Augmentation**: Inject small amounts of Gaussian noise into technical indicators during sequence construction to act as a regularization technique and encourage the model to generalize better to highly volatile market periods.
* **[ ] Sentiment & Macroeconomic Indicators**: Expand features to incorporate market-wide fear/greed indices, dynamic interest rates, inflation indicators, or news sentiment scores to capture macroeconomic drivers.
* **[ ] Temporal Fusion Transformers (TFT)**: If sequence model performance plateaus, evaluate advanced transformer-based structures that naturally handle static metadata (like asset categories) alongside multi-horizon time-series data.