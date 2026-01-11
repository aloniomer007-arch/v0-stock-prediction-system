"""
Data ingestion and preprocessing for stock prediction system.
Handles 20 years of historical data with price, fundamentals, and macro indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Optional
import json

class DataIngestion:
    """Ingest and preprocess stock data from multiple sources."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_price_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.
        Returns adjusted prices with corporate actions applied.
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            # Rename columns to match schema
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'splits'
            })
            
            # Add adjusted close (yfinance already adjusts Close)
            df['adj_close'] = df['close']
            df['ticker'] = ticker
            df['date'] = df.index
            
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"[v0] Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_fundamentals(self, ticker: str) -> Dict:
        """
        Fetch fundamental data (quarterly/annual financials).
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals = {
                'market_cap': info.get('marketCap', np.nan),
                'pe_ratio': info.get('trailingPE', np.nan),
                'pb_ratio': info.get('priceToBook', np.nan),
                'revenue': info.get('totalRevenue', np.nan),
                'eps': info.get('trailingEps', np.nan),
                'roe': info.get('returnOnEquity', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
            }
            return fundamentals
        except Exception as e:
            print(f"[v0] Error fetching fundamentals for {ticker}: {e}")
            return {}
    
    def fetch_macro_indicators(self) -> pd.DataFrame:
        """
        Fetch macro indicators: S&P500, VIX, interest rates.
        """
        try:
            # Fetch S&P 500
            sp500 = yf.Ticker("^GSPC")
            sp_df = sp500.history(start=self.start_date, end=self.end_date)
            sp_df = sp_df[['Close']].rename(columns={'Close': 'sp500'})
            
            # Fetch VIX
            vix = yf.Ticker("^VIX")
            vix_df = vix.history(start=self.start_date, end=self.end_date)
            vix_df = vix_df[['Close']].rename(columns={'Close': 'vix'})
            
            # Combine
            macro = sp_df.join(vix_df, how='outer')
            macro['date'] = macro.index
            macro = macro.reset_index(drop=True)
            
            # Forward fill missing values
            macro = macro.fillna(method='ffill')
            
            return macro
        except Exception as e:
            print(f"[v0] Error fetching macro indicators: {e}")
            return pd.DataFrame()
    
    def ingest_all(self) -> pd.DataFrame:
        """
        Ingest all data for all tickers and merge with macro indicators.
        """
        all_data = []
        
        print(f"[v0] Fetching macro indicators...")
        macro_df = self.fetch_macro_indicators()
        
        for ticker in self.tickers:
            print(f"[v0] Fetching data for {ticker}...")
            price_df = self.fetch_price_data(ticker)
            
            if price_df.empty:
                continue
            
            # Fetch fundamentals
            fundamentals = self.fetch_fundamentals(ticker)
            for key, value in fundamentals.items():
                price_df[key] = value
            
            # Merge with macro indicators
            if not macro_df.empty:
                price_df['date'] = pd.to_datetime(price_df['date'])
                macro_df['date'] = pd.to_datetime(macro_df['date'])
                price_df = pd.merge(price_df, macro_df, on='date', how='left')
            
            all_data.append(price_df)
        
        # Combine all tickers
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by ticker and date
        combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        print(f"[v0] Ingested {len(combined_df)} rows for {len(self.tickers)} tickers")
        
        return combined_df


def main():
    """Main ingestion pipeline."""
    # Sample tickers (S&P 500 constituents)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
    
    # 20 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*20)
    
    ingestion = DataIngestion(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    df = ingestion.ingest_all()
    
    # Save to parquet
    output_path = 'data/raw_stock_data.parquet'
    df.to_parquet(output_path, index=False)
    print(f"[v0] Saved data to {output_path}")
    print(f"[v0] Shape: {df.shape}")
    print(f"[v0] Columns: {df.columns.tolist()}")
    print(f"[v0] Date range: {df['date'].min()} to {df['date'].max()}")

if __name__ == '__main__':
    main()
