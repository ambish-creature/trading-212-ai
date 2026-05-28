#!/usr/bin/env bash
# train_short.sh — Retrain short horizons (1mo, 2mo, 3mo, 6mo) with improved direction weighting
set -e

TRIALS=${1:-15}
TIMEFRAMES=("1mo" "2mo" "3mo" "6mo")

echo "================================================================="
echo "🏋️  Trading Bot v2.0 — Short Horizon Retrain Pipeline"
echo "   Optuna trials per horizon: $TRIALS"
echo "   Total horizons: ${#TIMEFRAMES[@]}"
echo "================================================================="
echo ""

# Step 1: Preprocess timeframes
echo "📊 Step 1/3: Preprocessing..."
for TF in "${TIMEFRAMES[@]}"; do
    echo "   Preprocessing $TF..."
    venv/bin/python src/data/preprocess.py --timeframe "$TF" --seed 42 2>&1 | tail -3
done
echo "✅ Preprocessing complete."
echo ""

# Step 2: Train models
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

# Step 3: Calibrate models
echo "📐 Step 3/3: Calibrating confidence thresholds..."
for TF in "${TIMEFRAMES[@]}"; do
    venv/bin/python scratch/calibrate.py --timeframe "$TF" --mc-samples 50
done
echo ""
echo "✅ Calibration complete."
echo ""

# Step 4: Verify accuracy
echo "📈 Verification: Running accuracy sweep..."
venv/bin/python scratch/verify_accuracy.py --all-timeframes --ticker VOO
echo ""
