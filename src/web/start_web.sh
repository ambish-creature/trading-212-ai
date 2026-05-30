#!/bin/bash
# start_web.sh — Launch FastAPI server in virtual environment on the Pi

# Resolve directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../../" && pwd )"

cd "$PROJECT_DIR" || exit 1

# Activate venv
if [ -d "venv" ]; then
    echo "⚡ Activating Python Virtual Environment..."
    source venv/bin/activate
else
    echo "❌ Error: Virtual environment 'venv' not found in $PROJECT_DIR."
    exit 1
fi

# Set host/port
HOST="0.0.0.0"
PORT=8000

echo "🚀 Launching Trading AI Web Dashboard on $HOST:$PORT..."
exec python -m uvicorn src.web.app:app --host "$HOST" --port "$PORT"
