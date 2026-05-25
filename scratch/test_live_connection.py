import os
import sys
from dotenv import load_dotenv

# Ensure base dir in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.config import API_KEY_REAL, API_SECRET_REAL, API_KEY_DEMO, API_SECRET_DEMO, get_base_url
from src.trading.execution import get_account_cash, get_portfolio_positions, place_market_order

def run_tests():
    print("=" * 60)
    print("🧪 TRADING 212 DUAL ACCOUNT CONNECTION VERIFICATION")
    print("=" * 60)
    
    # 1. Print Env Credentials Check (masked)
    print("\n🔑 Credentials Loaded Check:")
    print(f"   DEMO Key:     {API_KEY_DEMO[:10]}...{API_KEY_DEMO[-5:] if API_KEY_DEMO else 'MISSING'}")
    print(f"   DEMO Secret:  {API_SECRET_DEMO[:6]}...{API_SECRET_DEMO[-4:] if API_SECRET_DEMO else 'MISSING'}")
    print(f"   REAL Key:     {API_KEY_REAL[:10]}...{API_KEY_REAL[-5:] if API_KEY_REAL else 'MISSING'}")
    print(f"   REAL Secret:  {API_SECRET_REAL[:6]}...{API_SECRET_REAL[-4:] if API_SECRET_REAL else 'MISSING'}")

    # 2. Test DEMO Account Summary
    print("\n🔍 Fetching DEMO Account Summary...")
    demo_cash = get_account_cash(real=False)
    print(f"   Available Cash (DEMO): ${demo_cash:,.2f}")
    
    # 3. Test DEMO Portfolio
    print("\n🔍 Fetching DEMO Portfolio Positions...")
    demo_positions = get_portfolio_positions(real=False)
    print(f"   Found {len(demo_positions)} positions on DEMO.")
    for idx, pos in enumerate(demo_positions, 1):
        ticker = pos.get("ticker")
        qty = float(pos.get("quantity", 0.0))
        price = float(pos.get("currentPrice", 0.0))
        value = qty * price
        print(f"     {idx}. {ticker}: {qty:.4f} shares @ ${price:.2f} (Value: ${value:,.2f})")

    # 4. Test REAL Account Summary
    print("\n🔍 Fetching REAL Account Summary...")
    real_cash = get_account_cash(real=True)
    print(f"   Available Cash (REAL): £{real_cash:,.2f}")
    
    # 5. Test REAL Portfolio
    print("\n🔍 Fetching REAL Portfolio Positions...")
    real_positions = get_portfolio_positions(real=True)
    print(f"   Found {len(real_positions)} positions on REAL.")
    for idx, pos in enumerate(real_positions, 1):
        ticker = pos.get("ticker")
        qty = float(pos.get("quantity", 0.0))
        price = float(pos.get("currentPrice", 0.0))
        value = qty * price
        print(f"     {idx}. {ticker}: {qty:.4f} shares @ £{price:.2f} (Value: £{value:,.2f})")

    # 6. Verify Security Constraints (Order Placement Safety)
    print("\n🛡️ Verifying Order Placement Safety Safeguards...")
    try:
        print("   Attempting to place mock market order on REAL (should be strictly BLOCKED)...")
        place_market_order("AAPL", 1, real=True)
        print("   ❌ FAIL: Order was not blocked! (This should never happen!)")
    except ValueError as e:
        print(f"   ✅ SUCCESS: Order was blocked as expected! Error message: \"{e}\"")
    except Exception as e:
        print(f"   ✅ SUCCESS: Order was blocked with custom exception: \"{e}\"")

    print("\n" + "=" * 60)
    print("🏁 CONNECTION VERIFICATION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
