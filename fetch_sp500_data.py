#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_sp500_tickers():
    """Fetch S&P 500 component tickers from Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Use requests to get the page content with a user agent header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, verify=False)
    response.raise_for_status()

    # Use pandas to read the HTML table from the response content
    tables = pd.read_html(response.text)
    # The first table is usually the S&P 500 components
    sp500_table = tables[0]

    tickers = sp500_table['Symbol'].tolist()

    # Clean up ticker symbols (remove any extra characters)
    tickers = [ticker.replace('.', '-') for ticker in tickers]

    return tickers

def main():
    # Get S&P 500 tickers
    print("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers")

    # Calculate date range: last 15 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)

    print(f"Fetching data from {start_date.date()} to {end_date.date()}")

    # Fetch data for all tickers at once (more efficient)
    print("Downloading all data...")
    data = yf.download(tickers, start=start_date, end=end_date, threads=True)

    # Extract Close prices (yfinance now returns adjusted prices as 'Close')
    # Columns are MultiIndex: (Price, Ticker)
    close_data = data['Close'].copy()

    print(f"Fetched data shape: {close_data.shape}")

    # Save as Parquet
    output_file = 'sp500_adjusted_close.parquet'
    close_data.to_parquet(output_file)
    print(f"Data saved to {output_file}")

    # Also save a CSV for quick inspection (optional)
    close_data.to_csv('sp500_adjusted_close.csv')
    print("Data also saved as sp500_adjusted_close.csv")

if __name__ == "__main__":
    main()