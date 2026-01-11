"""
Utility to validate if a ticker exists and has sufficient data.
"""

import yfinance as yf
from datetime import datetime, timedelta
import sys
import json

def validate_ticker(ticker: str) -> dict:
    """
    Check if ticker exists and has recent data.
    Returns dict with status and info.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Try to fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty or len(hist) < 10:
            return {
                'valid': False,
                'error': 'Ticker not found or insufficient data',
                'ticker': ticker.upper()
            }
        
        # Get stock info
        info = stock.info
        
        return {
            'valid': True,
            'ticker': ticker.upper(),
            'name': info.get('longName', 'Unknown'),
            'exchange': info.get('exchange', 'Unknown'),
            'sector': info.get('sector', 'Unknown'),
            'last_price': float(hist['Close'].iloc[-1]),
            'last_date': hist.index[-1].strftime('%Y-%m-%d'),
            'data_points': len(hist)
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'ticker': ticker.upper()
        }

if __name__ == '__main__':
    # If called from API, read from stdin
    if not sys.stdin.isatty():
        try:
            input_data = json.loads(sys.stdin.read())
            ticker = input_data['ticker']
            result = validate_ticker(ticker)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({'valid': False, 'error': str(e)}))
    else:
        # Test validation
        test_tickers = ['AAPL', 'MSFT', 'ILIKEDOGS', 'FAKE123']
        
        for ticker in test_tickers:
            result = validate_ticker(ticker)
            print(f"\n{ticker}: {result}")
