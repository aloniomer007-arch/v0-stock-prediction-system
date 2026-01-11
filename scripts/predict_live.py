"""
Live prediction using trained model.
Reads from stdin and writes prediction to stdout as JSON.
"""

import sys
import json
import yfinance as yf
import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_latest_features(ticker: str):
    """Fetch latest data and compute features."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get last 90 days to compute indicators
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty or len(df) < 60:
            return None
        
        # Compute same features as training
        prices = df['Close'].values
        returns = np.diff(prices) / prices[:-1]
        
        features = {
            'return_1d': returns[-1] if len(returns) > 0 else 0,
            'sma_5': df['Close'].rolling(5).mean().iloc[-1],
            'sma_20': df['Close'].rolling(20).mean().iloc[-1],
            'sma_50': df['Close'].rolling(50).mean().iloc[-1],
            'volatility_20': pd.Series(returns).rolling(20).std().iloc[-1],
            'Volume': df['Volume'].iloc[-1]
        }
        
        # RSI
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = pd.Series(gains[-14:]).mean()
        avg_loss = pd.Series(losses[-14:]).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        return features, df['Close'].iloc[-1], df.index[-1]
    
    except Exception as e:
        print(json.dumps({'error': f'Failed to fetch data: {str(e)}'}), file=sys.stderr)
        return None

def main():
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    ticker = input_data['ticker']
    horizon = input_data['horizon']
    
    # Load trained model
    try:
        model = lgb.Booster(model_file='data/trained_model.txt')
    except:
        print(json.dumps({'error': 'Model not found. Please train first.'}))
        return
    
    # Get latest features
    result = get_latest_features(ticker)
    if result is None:
        print(json.dumps({'error': 'Failed to fetch latest data'}))
        return
    
    features, last_price, last_date = result
    
    # Make prediction
    feature_order = ['return_1d', 'sma_5', 'sma_20', 'sma_50', 'volatility_20', 'rsi_14', 'Volume']
    X = np.array([[features[f] for f in feature_order]])
    
    pred_return = model.predict(X)[0]
    
    # Convert to price prediction
    pred_price = last_price * (1 + pred_return)
    
    # Rough confidence intervals (using typical volatility)
    std = features['volatility_20']
    lower = pred_return - 1.96 * std
    upper = pred_return + 1.96 * std
    
    output = {
        'ticker': ticker,
        'as_of_date': last_date.strftime('%Y-%m-%d'),
        'current_price': float(last_price),
        'horizon': horizon,
        'predicted_return': float(pred_return),
        'predicted_price': float(pred_price),
        'lower_bound': float(lower),
        'upper_bound': float(upper),
        'confidence': 0.95,
        'model_version': 'v1.0-lightgbm-real',
        'features_used': features,
        'timestamp': datetime.now().isoformat()
    }
    
    print(json.dumps(output))

if __name__ == '__main__':
    main()
