"""
Feature engineering with exact mathematical formulas as specified.
Implements technical indicators, returns, volatility measures, and derived features.
"""

import pandas as pd
import numpy as np
from typing import Tuple

class FeatureEngineer:
    """
    Feature engineering for stock prediction.
    All formulas implemented exactly as specified in requirements.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def compute_returns(self) -> pd.DataFrame:
        """
        Compute simple and log returns.
        
        Simple return: R_t = (P_t - P_{t-1}) / P_{t-1}
        Log return: r_t = ln(P_t / P_{t-1})
        """
        df = self.df.copy()
        
        # Group by ticker to compute returns within each ticker
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            prices = df.loc[mask, 'adj_close'].values
            
            # Simple returns
            simple_returns = np.diff(prices) / prices[:-1]
            df.loc[mask, 'return_1d'] = np.concatenate([[np.nan], simple_returns])
            
            # Log returns
            log_returns = np.log(prices[1:] / prices[:-1])
            df.loc[mask, 'log_return_1d'] = np.concatenate([[np.nan], log_returns])
        
        return df
    
    def compute_sma(self, window: int) -> pd.DataFrame:
        """
        Simple Moving Average.
        
        SMA_{t,N} = (1/N) * sum_{i=0}^{N-1} P_{t-i}
        """
        df = self.df.copy()
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            df.loc[mask, f'sma_{window}'] = df.loc[mask, 'adj_close'].rolling(window=window).mean()
        
        return df
    
    def compute_ema(self, window: int) -> pd.DataFrame:
        """
        Exponential Moving Average.
        
        alpha = 2 / (N + 1)
        EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}
        
        Initialize with SMA of first N points.
        """
        df = self.df.copy()
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            df.loc[mask, f'ema_{window}'] = df.loc[mask, 'adj_close'].ewm(span=window, adjust=False).mean()
        
        return df
    
    def compute_volatility(self, window: int = 90) -> pd.DataFrame:
        """
        Rolling volatility of log returns.
        
        sigma_{t,N} = sqrt((1/(N-1)) * sum_{i=0}^{N-1} (r_{t-i} - r_bar)^2)
        """
        df = self.df.copy()
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            df.loc[mask, f'vol_{window}'] = df.loc[mask, 'log_return_1d'].rolling(window=window).std()
        
        return df
    
    def compute_atr(self, window: int = 14) -> pd.DataFrame:
        """
        Average True Range.
        
        TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
        ATR_{t,N} = (1/N) * sum_{i=0}^{N-1} TR_{t-i}
        """
        df = self.df.copy()
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_df = df[mask].copy()
            
            high = ticker_df['high'].values
            low = ticker_df['low'].values
            close = ticker_df['close'].values
            
            # True Range calculation
            tr = np.zeros(len(ticker_df))
            for i in range(1, len(ticker_df)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
            
            # ATR as simple moving average of TR
            atr = pd.Series(tr).rolling(window=window).mean().values
            df.loc[mask, f'atr_{window}'] = atr
        
        return df
    
    def compute_macd(self) -> pd.DataFrame:
        """
        MACD = EMA_12 - EMA_26
        Signal = EMA_9(MACD)
        """
        df = self.df.copy()
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            
            ema_12 = df.loc[mask, 'adj_close'].ewm(span=12, adjust=False).mean()
            ema_26 = df.loc[mask, 'adj_close'].ewm(span=26, adjust=False).mean()
            
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            
            df.loc[mask, 'macd'] = macd.values
            df.loc[mask, 'macd_signal'] = signal.values
        
        return df
    
    def compute_rsi(self, window: int = 14) -> pd.DataFrame:
        """
        RSI (Wilder's method).
        
        RSI_t = 100 - 100 / (1 + avgGain_t / avgLoss_t)
        
        avgGain and avgLoss are exponential moving averages with alpha = 1/N.
        """
        df = self.df.copy()
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            prices = df.loc[mask, 'adj_close'].values
            
            # Price changes
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Initialize with SMA
            avg_gain = np.mean(gains[:window]) if len(gains) >= window else 0
            avg_loss = np.mean(losses[:window]) if len(losses) >= window else 0
            
            rsi_values = [np.nan] * window
            
            # Wilder's smoothing
            alpha = 1.0 / window
            for i in range(window, len(gains)):
                avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - 100 / (1 + rs)
                
                rsi_values.append(rsi)
            
            rsi_values.append(np.nan)  # Account for diff reducing length by 1
            df.loc[mask, f'rsi_{window}'] = rsi_values
        
        return df
    
    def compute_targets(self) -> pd.DataFrame:
        """
        Compute prediction targets:
        - y_1d: next-day return
        - y_7d: next-week return (5 trading days)
        - y_30d: next-month return (21 trading days)
        """
        df = self.df.copy()
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            log_returns = df.loc[mask, 'log_return_1d'].values
            
            # Next-day return
            y_1d = np.roll(log_returns, -1)
            y_1d[-1] = np.nan
            df.loc[mask, 'target_1d'] = y_1d
            
            # Next-week return (sum of 5 days)
            y_7d = np.full(len(log_returns), np.nan)
            for i in range(len(log_returns) - 5):
                y_7d[i] = np.sum(log_returns[i+1:i+6])
            df.loc[mask, 'target_7d'] = y_7d
            
            # Next-month return (sum of 21 days)
            y_30d = np.full(len(log_returns), np.nan)
            for i in range(len(log_returns) - 21):
                y_30d[i] = np.sum(log_returns[i+1:i+22])
            df.loc[mask, 'target_30d'] = y_30d
        
        return df
    
    def add_calendar_features(self) -> pd.DataFrame:
        """Add calendar-based features."""
        df = self.df.copy()
        
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Run all feature engineering steps.
        """
        print("[v0] Computing returns...")
        self.df = self.compute_returns()
        
        print("[v0] Computing moving averages...")
        self.df = self.compute_sma(5)
        self.df = self.compute_sma(20)
        self.df = self.compute_sma(50)
        self.df = self.compute_ema(12)
        self.df = self.compute_ema(26)
        
        print("[v0] Computing volatility...")
        self.df = self.compute_volatility(90)
        
        print("[v0] Computing ATR...")
        self.df = self.compute_atr(14)
        
        print("[v0] Computing MACD...")
        self.df = self.compute_macd()
        
        print("[v0] Computing RSI...")
        self.df = self.compute_rsi(14)
        
        print("[v0] Computing targets...")
        self.df = self.compute_targets()
        
        print("[v0] Adding calendar features...")
        self.df = self.add_calendar_features()
        
        print(f"[v0] Feature engineering complete. Shape: {self.df.shape}")
        
        return self.df


def main():
    """Run feature engineering on raw data."""
    # Load raw data
    df = pd.read_parquet('data/raw_stock_data.parquet')
    print(f"[v0] Loaded {len(df)} rows")
    
    # Engineer features
    engineer = FeatureEngineer(df)
    df_features = engineer.engineer_all_features()
    
    # Save engineered features
    output_path = 'data/features.parquet'
    df_features.to_parquet(output_path, index=False)
    print(f"[v0] Saved features to {output_path}")
    print(f"[v0] Columns: {df_features.columns.tolist()}")

if __name__ == '__main__':
    main()
