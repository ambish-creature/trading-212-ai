import requests
import base64
from ..config import API_KEY, API_SECRET, get_base_url

def get_auth_header():
    credentials = f"{API_KEY}:{API_SECRET}"
    encoded = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
    return {"Authorization": f"Basic {encoded}"}

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
