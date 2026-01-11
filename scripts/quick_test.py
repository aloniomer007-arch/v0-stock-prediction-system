"""
Quick test to prove Python execution works with real data
"""
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

print("=" * 60)
print("QUICK TEST - Downloading REAL stock data")
print("=" * 60)

# Download just 1 year of Apple stock
print("\nDownloading AAPL data from Yahoo Finance...")
ticker = yf.Ticker("AAPL")
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                    end=end_date.strftime('%Y-%m-%d'))

print(f"\nRESULTS:")
print(f"  Rows downloaded: {len(df)}")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  Latest close price: ${df['Close'].iloc[-1]:.2f}")
print(f"  Average daily return: {df['Close'].pct_change().mean():.4f}")
print(f"  Volatility (std): {df['Close'].pct_change().std():.4f}")

print("\n" + "=" * 60)
print("✓ Real data successfully downloaded and analyzed")
print("=" * 60)
