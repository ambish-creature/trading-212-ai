import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

# Add the project root to python path for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import ASSETS, TIMEFRAME_PROFILES, ACTIVE_TIMEFRAME
from src.models.advisor import run_advisor

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingServer")

app = FastAPI(
    title="Trading 212 AI Stock Prediction Dashboard",
    description="Deep Learning Multi-Horizon Stock Advisor with Monte Carlo Dropout uncertainty calibration.",
    version="3.0.0"
)

# Enable CORS for local testing and Cloudflare flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and Templates directories
web_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(web_dir, "static")
templates_dir = os.path.join(web_dir, "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates engine
templates = Jinja2Templates(directory=templates_dir)

# --- Compute queue lock for Pi CPU protection ---
# PyTorch MC Dropout uses 100 passes per prediction. To prevent Pi resource exhaustion,
# we force all predictions to execute in a single-file FIFO queue.
prediction_lock = asyncio.Lock()
waiting_count = 0  # Number of requests currently waiting in the queue

def get_passcode_configured() -> Optional[str]:
    """Returns the passcode if configured in the environment."""
    return os.getenv("TRADING_DASHBOARD_PASSCODE")

@app.get("/", response_class=HTMLResponse)
async def home_route(request: Request):
    """Renders the main glassmorphic dashboard."""
    passcode_required = get_passcode_configured() is not None
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"passcode_required": passcode_required}
    )

@app.get("/api/tickers")
async def api_get_tickers():
    """Returns the list of supported assets configured in the AI bot."""
    formatted_assets = []
    for ticker, category in ASSETS.items():
        formatted_assets.append({
            "ticker": ticker,
            "category": category
        })
    return {
        "success": True,
        "tickers": formatted_assets
    }

@app.get("/api/queue")
async def api_get_queue_status():
    """Returns the current state of the request queue."""
    return {
        "waiting": waiting_count,
        "active": 1 if prediction_lock.locked() else 0
    }

@app.get("/api/analyze")
async def api_analyze_ticker(
    ticker: str = Query(..., min_length=1, max_length=15, description="Stock ticker (e.g. AAPL)"),
    horizon: str = Query("1mo", description="Horizon timeframe (e.g. 1mo, 2mo, 3mo, 6mo, 1yr, 2yr)"),
    holding: float = Query(0.0, description="Shares currently held"),
    avg_cost: Optional[float] = Query(None, description="Average cost paid per share"),
    passcode: Optional[str] = Query(None, description="Access passcode if configured")
):
    """
    Executes model inference and advice pipeline for a given stock ticker.
    This endpoint features strict single-thread FIFO serialization to protect Pi system health.
    """
    global waiting_count
    
    # 1. Passcode Check
    expected_passcode = get_passcode_configured()
    if expected_passcode and passcode != expected_passcode:
        logger.warning(f"Unauthorized access attempt for ticker {ticker} with passcode: '{passcode}'")
        raise HTTPException(status_code=401, detail="Invalid access passcode. Please check your credentials.")

    ticker_upper = ticker.strip().upper()
    logger.info(f"New prediction request received for {ticker_upper} ({horizon}) | Queue size: {waiting_count}")

    # Validate timeframe horizon
    if horizon not in TIMEFRAME_PROFILES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid horizon. Must be one of: {', '.join(TIMEFRAME_PROFILES.keys())}"
        )

    # 2. Wait in queue
    waiting_count += 1
    try:
        async with prediction_lock:
            # We are no longer waiting, we are running!
            waiting_count = max(0, waiting_count - 1)
            logger.info(f"Acquired prediction lock. Running inference for {ticker_upper} ({horizon})...")
            
            # Since PyTorch model load and inference is CPU-heavy synchronous blocking code,
            # run it in FastAPI's offloaded threadpool to keep the server responsive.
            pred_scaled, advice = await run_in_threadpool(
                run_advisor,
                ticker=ticker_upper,
                horizon_days=horizon,
                reference_date=None,
                holding_shares=holding,
                avg_cost=avg_cost if holding > 0 else None,
                ask_holdings=False
            )
            
            logger.info(f"Successfully finished inference for {ticker_upper}. Release lock.")
            
            # Extract info safely
            info = pred_scaled.get("info", {})
            
            return {
                "success": True,
                "ticker": ticker_upper,
                "horizon": horizon,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "current_price": float(pred_scaled.get("current_price", 0.0) or 0.0),
                    "predicted_return": float(pred_scaled.get("predicted_return", 0.0)),
                    "confidence": float(pred_scaled.get("confidence", 0.0)),
                    "lower_bound": float(pred_scaled.get("lower_bound", 0.0)),
                    "upper_bound": float(pred_scaled.get("upper_bound", 0.0)),
                    "ratio": float(pred_scaled.get("ratio", 0.0)),
                    "optimal_ratio_threshold": float(pred_scaled.get("optimal_ratio_threshold", 0.15)),
                    "mc_std": float(pred_scaled.get("mc_std", 0.0)),
                    "model_sigma": float(pred_scaled.get("model_sigma", 0.0)),
                    "total_uncertainty": float(pred_scaled.get("total_uncertainty", 0.0))
                },
                "info": {
                    "name": info.get("name", ticker_upper),
                    "sector": info.get("sector", "N/A"),
                    "currency": info.get("currency", "USD"),
                    "pe_ratio": info.get("pe_ratio"),
                    "market_cap": info.get("market_cap"),
                    "dividend_yield": float(info.get("dividend_yield", 0.0) or 0.0) * 100,
                    "beta": info.get("beta"),
                    "high_52w": info.get("52w_high"),
                    "low_52w": info.get("52w_low"),
                    "analyst_target": info.get("analyst_target")
                },
                "advice": advice
            }
            
    except asyncio.CancelledError:
        # Decrement queue size if request was aborted/cancelled while waiting
        waiting_count = max(0, waiting_count - 1)
        logger.warning(f"Request for {ticker_upper} was cancelled by the client.")
        raise
    except Exception as e:
        logger.error(f"Error executing prediction for {ticker_upper}: {e}", exc_info=True)
        # Ensure we always keep our queue clean
        raise HTTPException(
            status_code=500, 
            detail=f"Inference Engine failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
