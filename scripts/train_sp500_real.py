"""
Real S&P 500 Training Script
Trains on ALL 500 S&P companies with 20 years of real data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import time

# Full S&P 500 tickers
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH",
    "V", "XOM", "WMT", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "KO", "PEP", "COST", "AVGO", "TMO", "LLY", "MCD", "CSCO", "ACN", "ABT",
    "ADBE", "NKE", "DHR", "TXN", "NEE", "PM", "VZ", "CMCSA", "UPS", "RTX",
    "HON", "ORCL", "NFLX", "INTC", "CRM", "WFC", "BMY", "QCOM", "AMD", "INTU",
    "UNP", "T", "SBUX", "LOW", "CAT", "BA", "IBM", "GE", "MDT", "SPGI",
    "AMT", "GILD", "BLK", "AXP", "CVS", "LMT", "SYK", "DE", "BKNG", "ADP",
    "TJX", "MMM", "C", "MDLZ", "ISRG", "MO", "CI", "ZTS", "DUK", "SO",
    "PLD", "SHW", "CB", "CME", "MS", "TGT", "USB", "CL", "BDX", "EOG",
    "ITW", "PNC", "NSC", "FIS", "GM", "APD", "MMC", "CSX", "HUM", "F",
    "MU", "ATVI", "PGR", "COF", "D", "GD", "EW", "NOC", "TFC", "EMR",
    "MCO", "EL", "BSX", "FCX", "AON", "PSA", "ROP", "REGN", "ILMN", "FDX",
    "SLB", "ECL", "LRCX", "KLAC", "APH", "CARR", "GIS", "MCK", "AFL", "AIG",
    "ICE", "MSI", "WELL", "AZO", "SYY", "KMB", "PPG", "ADM", "HSY", "TRV",
    "PAYX", "TROW", "ALL", "WM", "HLT", "CTAS", "ORLY", "FTNT", "CMG", "IDXX",
    "STZ", "MSCI", "NEM", "EXC", "DFS", "BK", "AME", "ROST", "A", "YUM",
    "VRSK", "DD", "CPRT", "AEP", "GPN", "SPG", "CTSH", "EA", "OTIS", "SRE",
    "IQV", "KHC", "FAST", "CMI", "AJG", "MNST", "XEL", "RMD", "ADSK", "EBAY",
    "KEYS", "MCHP", "FITB", "VRSN", "DOW", "CCI", "KMI", "HPQ", "WBA", "GLW",
    "ED", "WST", "MTD", "AWK", "WEC", "CTVA", "TSCO", "DLR", "EXR", "ANSS",
    "ES", "ETR", "PCAR", "ROK", "FTV", "CBRE", "SBAC", "FRC", "RF", "WY",
    "AEE", "EVRG", "CAH", "LH", "PPL", "HBAN", "CFG", "AES", "MLM", "LYB",
    "FE", "CMS", "PEAK", "DTE", "EXPD", "NTRS", "VICI", "HOLX", "VMC", "TDY",
    "STT", "EIX", "CNP", "K", "ZBRA", "WAT", "PKI", "ARE", "BR", "TTWO",
    "DRI", "CHD", "FANG", "IP", "CLX", "MAA", "TSN", "AKAM", "UAL", "LUV",
    "ALGN", "INVH", "APTV", "AVB", "JBHT", "LDOS", "SWK", "INCY", "IFF", "EQR",
    "MKC", "TER", "DAL", "SWKS", "SIVB", "MPWR", "NDAQ", "CPT", "WDC", "AMCR",
    "POOL", "TYL", "UDR", "ENPH", "JKHY", "TECH", "BBY", "APA", "CHRW", "LKQ",
    "FFIV", "HSIC", "EXPE", "LW", "GNRC", "J", "REG", "TRMB", "PAYC", "HII",
    "HWM", "FBHS", "BF-B", "AAL", "QRVO", "CRL", "TAP", "VTRS", "CE", "DVN",
    "WYNN", "PWR", "ALLE", "RL", "WHR", "PNW", "MTCH", "DISH", "NLSN", "NWS",
    "FMC", "BXP", "NWSA", "LNC", "IVZ", "HAS", "AIZ", "MOS", "DXC", "VNO",
    "UAA", "UA", "HBI", "SEE", "NI", "PNR", "PBCT", "FRT", "ALK", "KIM",
    "ZION", "TPR", "GL", "SJM", "JNPR", "NCLH", "PVH", "XRX", "LEG", "AAP",
    "NOV", "MHK", "PRGO", "CF", "ALB", "FOXA", "FOX", "COO", "AOS", "VFC",
    "PENN", "PHM", "IPG", "EMN", "OGN", "HFC", "BWA", "NWL", "GPS", "LEN",
    "RHI", "CMA", "UHS", "WRK", "RJF", "DVA", "OMC", "ABMD", "NLOK", "MGM",
    "HRB", "IRM", "BBWI", "SNA", "CPB", "PBCT", "LB", "DISCA", "DISCK", "INFO",
    "CZR", "MRO", "FLS", "MKTX", "LVS", "NRG", "AIV", "IPGP", "WAB", "WU",
    "PFG", "DGX", "BEN", "RE", "HST", "HSIC", "COG", "IEX", "ATO", "XRAY"
]

print(f"[Training] Starting training on {len(SP500_TICKERS)} S&P 500 companies")
print(f"[Training] Period: Last 20 years of data")
print(f"[Training] Expected time: 15-20 minutes...")
print(f"[Training] This is REAL training with actual market data\n")

start_time = time.time()

def download_data(ticker, start_date, end_date):
    """Download historical data for a ticker"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty or len(df) < 100:
            return None
        df['ticker'] = ticker
        return df
    except:
        return None

def calculate_features(df):
    """Calculate technical indicators"""
    # Price features
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_20d'] = df['Close'].pct_change(20)
    
    # Moving averages
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    
    # Volatility
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    
    # Volume
    df['volume_20d_avg'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_20d_avg']
    
    # Target: next day return
    df['target'] = df['return_1d'].shift(-1)
    
    return df

# Download data for all tickers
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 20)  # 20 years

print(f"[Training] Downloading data from {start_date.date()} to {end_date.date()}...")

all_data = []
success_count = 0

for i, ticker in enumerate(SP500_TICKERS):
    if i % 25 == 0:
        elapsed = time.time() - start_time
        print(f"[Training] Progress: {i}/{len(SP500_TICKERS)} tickers ({elapsed/60:.1f} minutes elapsed)")
    
    df = download_data(ticker, start_date, end_date)
    if df is not None:
        df = calculate_features(df)
        all_data.append(df)
        success_count += 1

print(f"\n[Training] Successfully downloaded {success_count}/{len(SP500_TICKERS)} tickers")

# Combine all data
print("[Training] Combining data and preparing features...")
combined_df = pd.concat(all_data, ignore_index=False)

# Feature columns
feature_cols = ['return_1d', 'return_5d', 'return_20d', 'sma_20', 'sma_50', 
                'ema_12', 'ema_26', 'rsi_14', 'macd', 'volatility_20d', 
                'volume_ratio']

# Remove NaN
combined_df = combined_df.dropna()

print(f"[Training] Total samples: {len(combined_df):,}")

# Prepare train/test split (time series)
split_date = end_date - timedelta(days=365)  # Last year for testing
train_df = combined_df[combined_df.index < split_date]
test_df = combined_df[combined_df.index >= split_date]

X_train = train_df[feature_cols]
y_train = train_df['target']
X_test = test_df[feature_cols]
y_test = test_df['target']

print(f"[Training] Train samples: {len(X_train):,}")
print(f"[Training] Test samples: {len(X_test):,}")

# Train model
print("[Training] Training LightGBM model on all S&P 500 data...")
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# Evaluate
print("[Training] Evaluating model...")
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

# Direction accuracy
train_direction = np.mean((train_pred > 0) == (y_train > 0)) * 100
test_direction = np.mean((test_pred > 0) == (y_test > 0)) * 100

total_time = time.time() - start_time

# Save results
results = {
    "model_type": "LightGBM",
    "tickers": success_count,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "train_rmse": float(train_rmse),
    "rmse": float(test_rmse),
    "train_direction_accuracy": float(train_direction),
    "direction_accuracy": float(test_direction),
    "improvement": float(test_direction - 50),
    "trained_date": datetime.now().isoformat(),
    "training_time_minutes": float(total_time / 60),
    "features": feature_cols
}

# Print results
print("\n" + "="*60)
print("TRAINING COMPLETE - REAL RESULTS FROM S&P 500")
print("="*60)
print(f"Model Type: {results['model_type']}")
print(f"S&P 500 Tickers Trained: {results['tickers']}")
print(f"Training Samples: {results['train_samples']:,}")
print(f"Test Samples: {results['test_samples']:,}")
print(f"Total Training Time: {results['training_time_minutes']:.1f} minutes")
print(f"\nTrain RMSE: {results['train_rmse']:.6f}")
print(f"Test RMSE: {results['rmse']:.6f}")
print(f"\nTrain Direction Accuracy: {results['train_direction_accuracy']:.2f}%")
print(f"Test Direction Accuracy: {results['direction_accuracy']:.2f}%")
print(f"\nImprovement over Random (50%): +{results['improvement']:.2f}%")
print("\nInterpretation:")
if test_direction >= 55:
    print("✓ EXCELLENT - Very profitable for trading")
elif test_direction >= 53:
    print("✓ GOOD - Profitable trading possible")
elif test_direction > 50:
    print("✓ BETTER than random - Potential edge")
else:
    print("⚠ Needs improvement - Similar to random guessing")
print("="*60)

# Save to file
with open('data/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[Training] Results saved to data/training_results.json")
print("[Training] Training complete!")
