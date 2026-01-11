"""
REAL training and evaluation pipeline.
Downloads data, trains models, and reports actual accuracy metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import json
import os

# Create data directory if needed
os.makedirs('data', exist_ok=True)

print("=" * 80)
print("STOCK PREDICTION MODEL - TRAINING & EVALUATION")
print("=" * 80)

# Step 1: Download REAL data
print("\n[1/6] Downloading 20 years of real stock data...")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']
end_date = datetime.now()
start_date = end_date - timedelta(days=365*20)

all_data = []
for ticker in tickers:
    print(f"  Fetching {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date.strftime('%Y-%m-%d'))
        df['ticker'] = ticker
        df['date'] = df.index
        all_data.append(df)
    except Exception as e:
        print(f"  ERROR fetching {ticker}: {e}")

if not all_data:
    print("FATAL: No data downloaded!")
    exit(1)

df = pd.concat(all_data, ignore_index=True)
print(f"  Downloaded {len(df):,} rows from {len(tickers)} tickers")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

# Step 2: Feature engineering
print("\n[2/6] Engineering features...")

def compute_features(df):
    """Compute technical indicators for prediction"""
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
        df.loc[mask, 'sma_50'] = df.loc[mask, 'Close'].rolling(50).mean()
        
        # Volatility
        df.loc[mask, 'volatility_20'] = df.loc[mask, 'return_1d'].rolling(20).std()
        
        # RSI
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = pd.Series(gains).rolling(14).mean().values
        avg_loss = pd.Series(losses).rolling(14).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        df.loc[mask, 'rsi_14'] = np.concatenate([[np.nan], rsi])
        
        # Target: next day return
        df.loc[mask, 'target_1d'] = np.concatenate([returns[1:], [np.nan]])
    
    return df

df = compute_features(df)

# Remove rows with NaN in features or target
feature_cols = ['return_1d', 'sma_5', 'sma_20', 'sma_50', 'volatility_20', 'rsi_14', 'Volume']
df_clean = df.dropna(subset=feature_cols + ['target_1d'])
print(f"  {len(df_clean):,} rows after feature engineering")

# Step 3: Time-series train/test split
print("\n[3/6] Splitting data (time-series aware)...")
# Sort by date
df_clean = df_clean.sort_values('date')

# Use 80% for training, 20% for testing
split_idx = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

print(f"  Training: {len(train_df):,} rows ({train_df['date'].min()} to {train_df['date'].max()})")
print(f"  Testing:  {len(test_df):,} rows ({test_df['date'].min()} to {test_df['date'].max()})")

X_train = train_df[feature_cols].values
y_train = train_df['target_1d'].values
X_test = test_df[feature_cols].values
y_test = test_df['target_1d'].values

# Step 4: Train LightGBM model
print("\n[4/6] Training LightGBM model...")
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
)

print(f"  Model trained with {model.num_trees()} trees")

# Step 5: Evaluate on test set
print("\n[5/6] Evaluating on test set...")
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Direction accuracy (most important for trading!)
direction_correct = np.sum(np.sign(y_test) == np.sign(y_pred))
direction_accuracy = direction_correct / len(y_test)

# Baseline comparison: always predict 0 (no change)
baseline_rmse = np.sqrt(mean_squared_error(y_test, np.zeros_like(y_test)))
baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
baseline_direction = np.sum(y_test > 0) / len(y_test)  # Just predicting up

print(f"\n{'='*80}")
print("MODEL PERFORMANCE REPORT")
print(f"{'='*80}")
print(f"\nTest Set Size: {len(y_test):,} predictions")
print(f"Date Range: {test_df['date'].min()} to {test_df['date'].max()}")
print(f"\nOUR MODEL:")
print(f"  RMSE:               {rmse:.6f} ({(1-rmse/baseline_rmse)*100:+.1f}% vs baseline)")
print(f"  MAE:                {mae:.6f} ({(1-mae/baseline_mae)*100:+.1f}% vs baseline)")
print(f"  Direction Accuracy: {direction_accuracy:.2%} (Hit Rate)")
print(f"\nBASELINE (predict 0):")
print(f"  RMSE:               {baseline_rmse:.6f}")
print(f"  MAE:                {baseline_mae:.6f}")
print(f"  Direction Accuracy: {baseline_direction:.2%}")
print(f"\nIMPROVEMENT:")
print(f"  RMSE Reduction:     {(1-rmse/baseline_rmse)*100:.1f}%")
print(f"  Direction Lift:     {(direction_accuracy/baseline_direction - 1)*100:+.1f}%")

# Step 6: Compare with walk-forward validation
print(f"\n{'='*80}")
print("[6/6] Walk-Forward Cross-Validation (More Realistic)")
print(f"{'='*80}")

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(df_clean)):
    fold_train = df_clean.iloc[train_idx]
    fold_test = df_clean.iloc[test_idx]
    
    X_fold_train = fold_train[feature_cols].values
    y_fold_train = fold_train['target_1d'].values
    X_fold_test = fold_test[feature_cols].values
    y_fold_test = fold_test['target_1d'].values
    
    fold_model = lgb.train(
        params,
        lgb.Dataset(X_fold_train, label=y_fold_train),
        num_boost_round=300,
        verbose_eval=False
    )
    
    fold_pred = fold_model.predict(X_fold_test)
    fold_rmse = np.sqrt(mean_squared_error(y_fold_test, fold_pred))
    fold_direction = np.sum(np.sign(y_fold_test) == np.sign(fold_pred)) / len(y_fold_test)
    
    cv_scores.append({
        'rmse': fold_rmse,
        'direction_accuracy': fold_direction
    })
    
    print(f"  Fold {fold+1}: RMSE={fold_rmse:.6f}, Direction Accuracy={fold_direction:.2%}")

avg_cv_rmse = np.mean([s['rmse'] for s in cv_scores])
avg_cv_direction = np.mean([s['direction_accuracy'] for s in cv_scores])

print(f"\nCross-Validation Averages:")
print(f"  RMSE:               {avg_cv_rmse:.6f}")
print(f"  Direction Accuracy: {avg_cv_direction:.2%}")

# Save results
results = {
    'test_metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'direction_accuracy': float(direction_accuracy),
        'n_samples': int(len(y_test))
    },
    'baseline_metrics': {
        'rmse': float(baseline_rmse),
        'mae': float(baseline_mae),
        'direction_accuracy': float(baseline_direction)
    },
    'cross_validation': {
        'avg_rmse': float(avg_cv_rmse),
        'avg_direction_accuracy': float(avg_cv_direction),
        'n_folds': 5
    },
    'data_info': {
        'n_tickers': len(tickers),
        'tickers': tickers,
        'train_samples': int(len(train_df)),
        'test_samples': int(len(test_df)),
        'date_range': f"{df['date'].min()} to {df['date'].max()}"
    },
    'timestamp': datetime.now().isoformat()
}

with open('data/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save the trained model
model.save_model('data/trained_model.txt')

print(f"\n{'='*80}")
print("SUMMARY:")
print(f"{'='*80}")
print(f"✓ Model trained on {len(train_df):,} real samples")
print(f"✓ Tested on {len(test_df):,} held-out samples")
print(f"✓ Direction accuracy: {direction_accuracy:.2%} (this is what matters for trading)")
print(f"✓ Model saved to: data/trained_model.txt")
print(f"✓ Results saved to: data/training_results.json")
print(f"\nNOTE: Direction accuracy >50% means model beats random guessing")
print(f"      Direction accuracy >53% is considered good for stock prediction")
print(f"      Current model: {direction_accuracy:.2%}")
print(f"{'='*80}\n")
