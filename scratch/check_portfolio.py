import os
import requests
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

api_key = os.getenv("TRADING212_API_KEY")
api_secret = os.getenv("TRADING212_API_SECRET")

print("🔑 Loaded T212 API Credentials:")
print(f"   API Key:    {api_key[:10]}...{api_key[-5:] if api_key else ''}")
print(f"   API Secret: {api_secret[:10]}...{api_secret[-5:] if api_secret else ''}")

# Headers for Trading 212 Basic Auth
# Note: Trading 212 API uses standard Authorization headers
headers = {"Authorization": api_key}

# Live URL
base_url = "https://live.trading212.com/api/v0"

print("\n🚀 Connecting to Trading 212 Live API...")

# 1. Fetch Account Summary
try:
    summary_url = f"{base_url}/equity/account/summary"
    response = requests.get(summary_url, headers=headers)
    print(f"🌐 Summary status: {response.status_code}")
    if response.status_code == 200:
        summary = response.json()
        print("📊 Live Account Summary:")
        print(f"   Total Value: {summary.get('total', 0.0):.2f} GBP")
        print(f"   Free Cash:   {summary.get('free', 0.0):.2f} GBP")
        print(f"   Blocked:     {summary.get('blocked', 0.0):.2f} GBP")
        print(f"   Invested:    {summary.get('ppl', 0.0):.2f} GBP")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Fetch Portfolio Positions
try:
    portfolio_url = f"{base_url}/equity/portfolio"
    response = requests.get(portfolio_url, headers=headers)
    print(f"\n🌐 Portfolio status: {response.status_code}")
    if response.status_code == 200:
        positions = response.json()
        print(f"📦 Found {len(positions)} held positions:")
        for idx, pos in enumerate(positions, 1):
            print(f"   {idx}. Ticker: {pos.get('ticker')} | Quantity: {pos.get('quantity')} | Value: {pos.get('currentPrice', 0.0) * pos.get('quantity', 0.0):.2f} GBP")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")
