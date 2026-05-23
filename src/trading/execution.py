import requests
import base64
from ..config import API_KEY, API_SECRET, get_base_url

def get_auth_header():
    credentials = f"{API_KEY}:{API_SECRET}"
    encoded = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
    return {"Authorization": f"Basic {encoded}"}

def get_portfolio_positions():
    """
    Fetches the currently held positions from Trading 212.
    Returns a list of dictionaries, or an empty list on failure.
    """
    url = f"{get_base_url()}/equity/portfolio"
    headers = get_auth_header()
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch portfolio: {response.status_code} {response.text}")
        return []

def get_account_cash():
    """
    Fetches available cash from Trading 212.
    Returns free cash amount, or 0.0 on failure.
    """
    url = f"{get_base_url()}/equity/account/summary"
    headers = get_auth_header()
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Parse nested cash -> availableToTrade
        if "cash" in data and isinstance(data["cash"], dict):
            return float(data["cash"].get("availableToTrade", 0.0))
        return float(data.get("free", data.get("totalValue", 0.0)))
    else:
        print(f"Failed to fetch account summary: {response.status_code} {response.text}")
        return 0.0

def place_market_order(ticker, quantity):
    """
    Places a market order via Trading 212 API.
    Quantity > 0 for BUY, Quantity < 0 for SELL.
    """
    url = f"{get_base_url()}/equity/orders/market"
    headers = get_auth_header()
    headers["Content-Type"] = "application/json"
    
    payload = {
        "ticker": ticker,
        "quantity": quantity
    }
    
    print(f"Placing market order for {quantity} shares of {ticker}...")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        print("Order placed successfully:", response.json())
    else:
        print("Failed to place order:", response.status_code, response.text)

if __name__ == "__main__":
    # Test connection
    print("Testing connection to Trading 212...")
    headers = get_auth_header()
    response = requests.get(f"{get_base_url()}/equity/account/summary", headers=headers)
    print("Account Summary:", response.json() if response.status_code == 200 else response.text)
