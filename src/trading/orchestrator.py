import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def run_script(script_path, args=[]):
    """Utility to run a python script as a subprocess and wait for completion."""
    cmd = [sys.executable, script_path] + args
    print(f"\n⚙️  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Success!")
        print(result.stdout[-500:]) # Print last 500 characters of stdout to avoid clutter
        return True
    else:
        print("❌ Failed!")
        print("STDOUT:")
        print(result.stdout[-1000:])
        print("STDERR:")
        print(result.stderr)
        return False

def execute_daily_pipeline():
    """Runs data fetch, preprocessing, and active trading execution."""
    print(f"\n⏱️  Executing Daily Trading Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    fetch_path = os.path.join(root_dir, 'src/data/fetch.py')
    preprocess_path = os.path.join(root_dir, 'src/data/preprocess.py')
    loop_path = os.path.join(root_dir, 'src/trading/loop.py')
    
    print("Step 1: Downloading latest daily bar data...")
    if not run_script(fetch_path):
        return False
        
    print("Step 2: Re-preprocessing indicators & split data...")
    if not run_script(preprocess_path):
        return False
        
    print("Step 3: Evaluating active trading signals and placing demo orders...")
    if not run_script(loop_path):
        return False
        
    return True

def execute_weekly_training_pipeline():
    """Fetches full historical data, preprocesses, and retrains LSTM model weights."""
    print(f"\n⏱️  Executing Weekly Model Self-Training Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    fetch_path = os.path.join(root_dir, 'src/data/fetch.py')
    preprocess_path = os.path.join(root_dir, 'src/data/preprocess.py')
    train_path = os.path.join(root_dir, 'src/models/train.py')
    
    print("Step 1: Downloading full fresh 10-year historical dataset...")
    if not run_script(fetch_path):
        return False
        
    print("Step 2: Preprocessing full dataset...")
    if not run_script(preprocess_path):
        return False
        
    print("Step 3: Triggering automated hyperparameter tuning and model retraining...")
    # This will train on the optimal settings, save a new timestamped model, and update active files
    if not run_script(train_path):
        return False
        
    print("\n🎉 Self-training successfully completed! Model weights updated automatically.")
    return True

def start_orchestrator(test_mode=False):
    print("=" * 60)
    print(f"🌲 AUTOMONOUS TRADING BOT ORCHESTRATOR STARTED")
    print(f"   Mode: {'HIGH-SPEED TEST MODE' if test_mode else '24/7 PRODUCTION MODE'}")
    print("=" * 60)
    
    if test_mode:
        print("\n⚡ Running high-speed scheduling dry-run...")
        print("1. Running Daily trading pipeline...")
        execute_daily_pipeline()
        
        print("\n2. Running Weekly self-training pipeline...")
        execute_weekly_training_pipeline()
        
        print("\n✅ High-speed verification completed successfully!")
        return

    # Production Loop (24/7)
    print("\n⏰ Orchestrator entered active scheduling loop.")
    print("   Daily Trading: Nasdaq Open (9:35 AM EST, Monday - Friday)")
    print("   Weekly Training: Saturdays at 12:00 AM EST")
    
    last_trade_date = None
    last_train_week = None
    
    while True:
        now = datetime.now()
        current_date_str = now.strftime('%Y-%m-%d')
        current_week_str = now.strftime('%Y-%U') # Year-WeekNumber
        
        # Check if it is a weekday (Monday=0, Friday=4)
        is_weekday = now.weekday() < 5
        
        # 1. Trigger Daily Trading Pipeline
        # We run at 9:35 AM EST (14:35 UTC or matching local time, adjust according to local timezone)
        # Here we configure it to run at 9:35 AM local time for simplicity, or 14:35 UTC.
        # Let's target 09:35 local system time.
        if is_weekday and now.hour == 9 and now.minute == 35 and last_trade_date != current_date_str:
            execute_daily_pipeline()
            last_trade_date = current_date_str
            
        # 2. Trigger Weekly Self-Training Pipeline
        # Run on Saturday (weekday = 5) at 12:00 AM (midnight)
        if now.weekday() == 5 and now.hour == 0 and now.minute == 0 and last_train_week != current_week_str:
            execute_weekly_training_pipeline()
            last_train_week = current_week_str
            
        # Check every 30 seconds
        time.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="24/7 Autonomous Trading Bot Orchestrator.")
    parser.add_argument("--test-mode", action="store_true", help="Run a high-speed dry-run verification of all orchestrator jobs.")
    args = parser.parse_args()
    
    start_orchestrator(test_mode=args.test_mode)
