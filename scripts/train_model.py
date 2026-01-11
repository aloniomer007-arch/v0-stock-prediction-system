"""
SIMPLIFIED REAL TRAINING - Optimized for v0 execution
Trains on 30 major stocks with 5 years of data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STOCK PREDICTION MODEL - REAL TRAINING")
print("="*80)

# Major stocks for training (diverse sectors)
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
    'JPM', 'BAC', 'WFC', 'V', 'MA',            # Finance
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO',        # Healthcare
    'XOM', 'CVX', 'COP',                        # Energy
    'WMT', 'HD', 'PG', 'KO', 'MCD',            # Consumer
    'BA', 'CAT', 'GE', 'UPS',                  # Industrial
    'DIS', 'NFLX', 'CMCSA'                     # Media
]

print(f"\n[1/5] Downloading 5 years of data for {len(TICKERS)} stocks...")

# Use yfinance for real data
try:
    import yfinance as yf
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    
    all_data = []
    for i, ticker in enumerate(TICKERS, 1):
        print(f"  [{i}/{len(TICKERS)}] Fetching {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                              end=end_date.strftime('%Y-%m-%d'))
            
            if len(df) > 100:
                df['ticker'] = ticker
                df['date'] = df.index
                all_data.append(df)
        except:
            print(f"    WARNING: Skipped {ticker}")
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n  SUCCESS: {len(df):,} rows downloaded")
    
except ImportError:
    print("  WARNING: yfinance not available, using synthetic data for demo")
    # Generate synthetic data for demo
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
    all_data = []
    
    for ticker in TICKERS[:10]:
        prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02))
        df_ticker = pd.DataFrame({
            'Close': prices,
            'Open': prices * (1 + np.random.randn(len(dates)) * 0.01),
            'High': prices * (1 + abs(np.random.randn(len(dates))) * 0.01),
            'Low': prices * (1 - abs(np.random.randn(len(dates))) * 0.01),
            'Volume': np.random.randint(1e6, 1e8, len(dates)),
            'ticker': ticker,
            'date': dates
        })
        all_data.append(df_ticker)
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"  DEMO MODE: Generated {len(df):,} synthetic rows")

# [2/5] Feature Engineering
print(f"\n[2/5] Creating technical features...")

df = df.sort_values(['ticker', 'date']).copy()

for ticker in df['ticker'].unique():
    mask = df['ticker'] == ticker
    prices = df.loc[mask, 'Close'].values
    
    # Returns
    returns = np.diff(prices) / prices[:-1]
    df.loc[mask, 'return_1d'] = np.concatenate([[np.nan], returns])
    
    # Moving averages
    df.loc[mask, 'sma_5'] = df.loc[mask, 'Close'].rolling(5).mean()
    df.loc[mask, 'sma_20'] = df.loc[mask, 'Close'].rolling(20).mean()
    
    # EMA
    df.loc[mask, 'ema_12'] = df.loc[mask, 'Close'].ewm(span=12).mean()
    
    # Volatility
    df.loc[mask, 'volatility'] = df.loc[mask, 'return_1d'].rolling(10).std()
    
    # RSI
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    avg_gain = pd.Series(gains).rolling(14).mean().values
    avg_loss = pd.Series(losses).rolling(14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    df.loc[mask, 'rsi'] = np.concatenate([[np.nan], rsi])
    
    # Target
    df.loc[mask, 'target'] = np.concatenate([returns[1:], [np.nan]])

feature_cols = ['return_1d', 'sma_5', 'sma_20', 'ema_12', 'volatility', 'rsi', 'Volume']
df_clean = df.dropna(subset=feature_cols + ['target'])
print(f"  Created {len(feature_cols)} features, {len(df_clean):,} clean rows")

# [3/5] Train/Test Split
print(f"\n[3/5] Splitting data...")
df_clean = df_clean.sort_values('date')
split_idx = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

X_train = train_df[feature_cols].values
y_train = train_df['target'].values
X_test = test_df[feature_cols].values
y_test = test_df['target'].values

print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

# [4/5] Train Model
print(f"\n[4/5] Training model...")

try:
    import lightgbm as lgb
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)
    y_pred = model.predict(X_test)
    model_type = "LightGBM"
    
    # Save model
    model.save_model('data/model.txt')
    
except ImportError:
    # Fallback to sklearn
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_type = "Sklearn GradientBoosting"
    
    # Save model
    import pickle
    with open('data/model.pkl', 'wb') as f:
        pickle.dump(model, f)

print(f"  Model trained: {model_type}")

# [5/5] Evaluate
print(f"\n[5/5] Evaluating...")

rmse = np.sqrt(np.mean((y_test - y_pred)**2))
mae = np.mean(np.abs(y_test - y_pred))

# Direction accuracy (KEY METRIC)
direction_correct = np.sum(np.sign(y_test) == np.sign(y_pred))
direction_accuracy = (direction_correct / len(y_test)) * 100

baseline_direction = (np.sum(y_test > 0) / len(y_test)) * 100
baseline_rmse = np.sqrt(np.mean(y_test**2))

print("\n" + "="*80)
print("TRAINING RESULTS")
print("="*80)
print(f"\nOUR MODEL:")
print(f"  Direction Accuracy: {direction_accuracy:.2f}% ← KEY METRIC FOR TRADING")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"\nBASELINE (always predict 0):")
print(f"  Direction Accuracy: {baseline_direction:.2f}%")
print(f"  RMSE: {baseline_rmse:.6f}")
print(f"\nIMPROVEMENT:")
print(f"  Direction Lift: {direction_accuracy - baseline_direction:+.2f}%")
print(f"  RMSE Reduction: {(1 - rmse/baseline_rmse)*100:.2f}%")
print("\nINTERPRETATION:")
print(f"  50% = Random guess")
print(f"  53%+ = Good for stocks")
print(f"  55%+ = Very good")
print(f"  YOUR MODEL: {direction_accuracy:.2f}%")
print("="*80)

# Save results
results = {
    'direction_accuracy': float(direction_accuracy),
    'rmse': float(rmse),
    'mae': float(mae),
    'baseline_direction': float(baseline_direction),
    'improvement': float(direction_accuracy - baseline_direction),
    'model_type': model_type,
    'test_samples': int(len(y_test)),
    'tickers': len(df_clean['ticker'].unique()),
    'timestamp': datetime.now().isoformat()
}

with open('data/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Model saved to data/")
print(f"✓ Results saved to data/results.json")
print("\nTraining complete! Refresh the UI to see results.\n")
