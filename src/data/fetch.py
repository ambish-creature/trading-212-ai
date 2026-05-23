import yfinance as yf
import pandas as pd
import os

def fetch_historical_data(ticker, start_date, end_date, output_path):
    """
    Fetches historical data using yfinance and saves it to a CSV.
    Trading 212 doesn't have an endpoint for rich historical candlesticks.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    fetch_historical_data("AAPL", "2020-01-01", "2023-01-01", "../data/raw/AAPL.csv")
