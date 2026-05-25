import requests
import base64
from ..config import API_KEY_REAL, API_SECRET_REAL, API_KEY_DEMO, API_SECRET_DEMO, get_base_url

def get_auth_header(real=False):
    key = API_KEY_REAL if real else API_KEY_DEMO
    secret = API_SECRET_REAL if real else API_SECRET_DEMO
    
    # Fallback to general API_KEY if others not defined
    if not key or not secret:
        from ..config import API_KEY, API_SECRET
        key = key or API_KEY
        secret = secret or API_SECRET
        
    credentials = f"{key}:{secret}"
    encoded = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
    return {"Authorization": f"Basic {encoded}"}

def get_portfolio_positions(real=False):
    """
    Fetches the currently held positions from Trading 212.
    Returns a list of dictionaries, or an empty list on failure.
    """
    url = f"{get_base_url(real=real)}/equity/portfolio"
    headers = get_auth_header(real=real)
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        account_type = "REAL" if real else "DEMO"
        print(f"Failed to fetch {account_type} portfolio: {response.status_code} {response.text}")
        return []

def get_account_cash(real=False):
    """
    Fetches available cash from Trading 212.
    Returns free cash amount, or 0.0 on failure.
    """
    url = f"{get_base_url(real=real)}/equity/account/summary"
    headers = get_auth_header(real=real)
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Parse nested cash -> availableToTrade
        if "cash" in data and isinstance(data["cash"], dict):
            return float(data["cash"].get("availableToTrade", 0.0))
        return float(data.get("free", data.get("totalValue", 0.0)))
    else:
        account_type = "REAL" if real else "DEMO"
        print(f"Failed to fetch {account_type} account summary: {response.status_code} {response.text}")
        return 0.0

def place_market_order(ticker, quantity, real=False):
    """
    Places a market order via Trading 212 API.
    Quantity > 0 for BUY, Quantity < 0 for SELL.
    """
    if real:
        raise ValueError("CRITICAL: Market orders are strictly BLOCKED on the REAL live account to ensure safety.")
        
    url = f"{get_base_url(real=real)}/equity/orders/market"
    headers = get_auth_header(real=real)
    headers["Content-Type"] = "application/json"
    
    payload = {
        "ticker": ticker,
        "quantity": quantity
    }
    
    print(f"Placing market order for {quantity} shares of {ticker} on DEMO...")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        print("Order placed successfully:", response.json())
    else:
        print("Failed to place order:", response.status_code, response.text)

if __name__ == "__main__":
    # Test connection
    print("Testing connection to Trading 212 DEMO...")
    headers_demo = get_auth_header(real=False)
    response_demo = requests.get(f"{get_base_url(real=False)}/equity/account/summary", headers=headers_demo)
    print("DEMO Account Summary:", response_demo.json() if response_demo.status_code == 200 else response_demo.text)

    print("\nTesting connection to Trading 212 REAL...")
    headers_real = get_auth_header(real=True)
    response_real = requests.get(f"{get_base_url(real=True)}/equity/account/summary", headers=headers_real)
    print("REAL Account Summary:", response_real.json() if response_real.status_code == 200 else response_real.text)

