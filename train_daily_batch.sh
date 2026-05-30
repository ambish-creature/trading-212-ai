#!/usr/bin/env bash
# train_daily_batch.sh — Train daily horizons (1d to 28d) in optimal parallel batches to max out resources without crashing
set -e

TRIALS=${1:-30}
DAILY_HORIZONS=(
    "1d" "2d" "3d" "4d" "5d" "6d" "7d" "8d" "9d" "10d" 
    "11d" "12d" "13d" "14d" "15d" "16d" "17d" "18d" "19d" "20d" 
    "21d" "22d" "23d" "24d" "25d" "26d" "27d" "28d"
)

BATCH_SIZE=4

echo "================================================================="
# Keep design premium and high-tech
echo "⚡ Trading Bot Daily Horizons Batch Train Pipeline"
echo "   Optuna trials per horizon: $TRIALS"
echo "   Total horizons: ${#DAILY_HORIZONS[@]}"
echo "   Batch size: $BATCH_SIZE (saturates RTX 4080 & 16 vCPUs)"
echo "================================================================="
echo ""

# Loop through in batches of BATCH_SIZE
for ((i=0; i<${#DAILY_HORIZONS[@]}; i+=BATCH_SIZE)); do
    BATCH=("${DAILY_HORIZONS[@]:i:BATCH_SIZE}")
    echo "🚀 Starting Batch: ${BATCH[*]}"
    
    # Step 1: Preprocess the batch
    echo "📊 Preprocessing batch..."
    for TF in "${BATCH[@]}"; do
        python src/data/preprocess.py --timeframe "$TF" --seed 42 2>&1 | tail -3 &
    done
    wait
    echo "✅ Preprocessing complete for batch."
    
    # Step 2: Train the batch in parallel
    echo "🔬 Training models in parallel..."
    for TF in "${BATCH[@]}"; do
        python src/models/train.py --timeframe "$TF" --trials "$TRIALS" > "train_${TF}.log" 2>&1 &
        echo "   -> Started training $TF (logging to train_${TF}.log)"
    done
    wait
    echo "✅ Training complete for batch."
    
    # Step 3: Calibrate the batch
    echo "📐 Calibrating models in batch..."
    for TF in "${BATCH[@]}"; do
        python scratch/calibrate.py --timeframe "$TF" --mc-samples 50 &
    done
    wait
    echo "✅ Calibration complete for batch."
    echo "---------------------------------------------------------"
done

echo "🎉 ALL DAILY HORIZON MODELS TRAINED AND CALIBRATED SUCCESSFULLY!"
