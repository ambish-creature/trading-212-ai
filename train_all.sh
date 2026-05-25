#!/usr/bin/env bash
# train_all.sh — Retrain all 7 horizon models with v2.0 architecture
# 
# Usage:
#   bash train_all.sh           # default: 15 Optuna trials per horizon
#   bash train_all.sh 20        # 20 Optuna trials per horizon
#   bash train_all.sh 0         # skip Optuna, use default params
#
# Total estimated time (MPS acceleration):
#   15 trials: ~4–6 hours for all 7 horizons
#   30 trials: ~8–12 hours for all 7 horizons

set -e

TRIALS=${1:-15}
TIMEFRAMES=("1mo" "2mo" "3mo" "6mo" "9mo" "1yr" "2yr")

echo "================================================================="
echo "🏋️  Trading Bot v2.0 — Full Retrain Pipeline"
echo "   Optuna trials per horizon: $TRIALS"
echo "   Total horizons: ${#TIMEFRAMES[@]}"
echo "================================================================="
echo ""

# Step 1: Preprocess all timeframes
echo "📊 Step 1/3: Preprocessing all timeframes..."
for TF in "${TIMEFRAMES[@]}"; do
    echo "   Preprocessing $TF..."
    venv/bin/python src/data/preprocess.py --timeframe "$TF" 2>&1 | tail -3
done
echo "✅ Preprocessing complete."
echo ""

# Step 2: Train all models
echo "🔬 Step 2/3: Training models..."
for TF in "${TIMEFRAMES[@]}"; do
    echo ""
    echo "-------------------------------------------"
    echo "  Training timeframe: $TF (trials=$TRIALS)"
    echo "-------------------------------------------"
    venv/bin/python src/models/train.py --timeframe "$TF" --trials "$TRIALS"
done
echo ""
echo "✅ Training complete."
echo ""

# Step 3: Calibrate all models
echo "📐 Step 3/3: Calibrating confidence thresholds..."
venv/bin/python scratch/calibrate.py --mc-samples 50
echo ""
echo "✅ Calibration complete."
echo ""

# Step 4: Verify accuracy
echo "📈 Verification: Running accuracy sweep..."
venv/bin/python scratch/verify_accuracy.py --all-timeframes --ticker VOO
echo ""

echo "================================================================="
echo "🎉 Full pipeline complete!"
echo "   To use the advisor:"
echo "     venv/bin/python src/models/advisor.py --ticker VOO --horizon 1mo"
echo "================================================================="
