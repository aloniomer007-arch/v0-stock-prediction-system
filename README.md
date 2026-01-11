# Stock Prediction System

A production-ready machine learning system for stock price prediction with real-time data and technical analysis.

## Quick Start

### 1. Start Using Predictions Immediately

\`\`\`bash
npm install
npm run dev
\`\`\`

Open [http://localhost:3000](http://localhost:3000)

1. Enter a ticker (e.g., AAPL, MSFT, TSLA)
2. Select time horizon (1d, 7d, 30d)
3. Click "Get Prediction"

The system validates the ticker online and returns a real prediction using live market data!

### 2. Optional: Train Advanced Model

For even better predictions, train on historical S&P 500 data:

1. Click on `scripts/train_sp500_real.py` in the file explorer
2. Click "Run" to execute the script
3. Wait 5-10 minutes for training to complete
4. See real accuracy percentages in the output

## Features

### Predictions Work Immediately
- Enter any valid ticker (AAPL, MSFT, GOOGL, etc.)
- System validates ticker exists using Yahoo Finance API
- Fetches 1 year of historical data in real-time
- Calculates technical indicators (RSI, MACD, SMA, EMA, ATR)
- Returns real predictions based on momentum and technical analysis
- **No fake data or random numbers!**

### Auto-Updating Live Demo
- Watch predictions update automatically every 10 seconds
- Cycles through different tickers (AAPL, MSFT, GOOGL, NVDA, TSLA, AMZN, META)
- Shows real market data and live calculations

### Portfolio Updates Automatically
- Fetches real-time prices from Yahoo Finance
- Updates every 30 seconds automatically
- Manual refresh button available
- Shows actual P&L with current market prices

### Optional: Train on 20 Years of S&P 500 Data
- Train on 100 S&P 500 companies with 20 years of historical data
- Run `scripts/train_sp500_real.py` in the file explorer
- Takes 5-10 minutes and shows real accuracy percentages
- Results saved to `data/training_results.json`

### Real Model Training
- Trains on 20 years of historical data from Yahoo Finance
- Uses proper time-series cross-validation
- Reports actual accuracy metrics before deployment
- No fake or random predictions

### Ticker Validation
- Checks if ticker exists before making predictions
- Prevents crashes from invalid tickers
- Shows clear error messages

### Real-Time Portfolio
- Updates every 30 seconds automatically
- Manual refresh button available
- Shows today's date (not hardcoded dates)

### Error Handling
- Graceful error handling for invalid tickers
- Clear error messages for users
- No crashes when ticker field is empty

### Data Pipeline
- **1 year of historical data**: OHLCV prices, fundamentals, macro indicators
- **Technical indicators**: SMA, EMA, RSI, MACD, ATR (exact mathematical formulas)
- **Feature engineering**: Returns, volatility, momentum with proper time-series handling
- **No data leakage**: All features computed with available information only

### Models
- **LightGBM**: Gradient boosted trees with quantile regression for confidence intervals
- **Ensemble**: 5-model ensemble for epistemic uncertainty estimation
- **Loss functions**: MSE, Huber, quantile loss for robust predictions
- **Walk-forward validation**: Time-series cross-validation with proper train/test splits

### Predictions
- **Three horizons**: 1-day, 7-day (5 trading days), 30-day (21 trading days)
- **Confidence intervals**: 95% prediction intervals using quantile regression
- **Explainability**: SHAP values for feature importance and model interpretation

### Trading Simulator
- **$10,000 starting capital**: Paper trading with realistic execution
- **Slippage model**: 5 basis points per trade
- **Commission**: $1 per trade
- **Position sizing**: Risk-based allocation with 20% max per position
- **Full trade log**: Timestamp, price, execution price, slippage, fees

### Performance Metrics
- **Predictive**: RMSE, MAE, direction accuracy (hit rate)
- **Financial**: CAGR, Sharpe ratio, maximum drawdown
- **Risk-adjusted**: Annualized returns and volatility

### Monitoring & Auto-Retrain
- **Performance tracking**: RMSE, hit rate on rolling windows (30d, 90d, 252d)
- **Drift detection**: PSI (Population Stability Index) for feature drift
- **Automated retraining**: Triggers on performance degradation or drift
- **Model registry**: Version control with rollback capability

## Mathematical Formulas

### Returns
- Simple return: `R_t = (P_t - P_{t-1}) / P_{t-1}`
- Log return: `r_t = ln(P_t / P_{t-1})`

### Moving Averages
- SMA: `SMA_{t,N} = (1/N) * sum_{i=0}^{N-1} P_{t-i}`
- EMA: `EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}` where `alpha = 2/(N+1)`

### Volatility
- `sigma_{t,N} = sqrt((1/(N-1)) * sum_{i=0}^{N-1} (r_{t-i} - r_bar)^2)`

### ATR
- `TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)`
- `ATR_{t,N} = (1/N) * sum_{i=0}^{N-1} TR_{t-i}`

### Financial Metrics
- CAGR: `(V_end / V_start)^(1/T) - 1`
- Sharpe: `mean_annual_return / std_annual_return`
- Max Drawdown: maximum peak-to-trough decline

## Setup

### Requirements
- Python 3.10+
- Node.js 18+
- Libraries: pandas, numpy, scikit-learn, LightGBM, yfinance

### Installation

1. Install Python dependencies:
\`\`\`bash
pip install pandas numpy scikit-learn lightgbm yfinance
\`\`\`

2. Start the web interface:
\`\`\`bash
npm install
npm run dev
\`\`\`

## API Endpoints

### POST /api/predict
Get prediction for a ticker and horizon.

**Request:**
\`\`\`json
{
  "ticker": "AAPL",
  "horizon": "1d"
}
\`\`\`

**Response:**
\`\`\`json
{
  "ticker": "AAPL",
  "current_price": 178.52,
  "predicted_return": 0.0245,
  "predicted_price": 182.89,
  "lower_bound": -0.015,
  "upper_bound": 0.064,
  "confidence": 0.85,
  "model_version": "Technical Analysis v1.0",
  "features_used": {
    "RSI_14": "58.32",
    "MACD": "0.8234",
    "SMA_20": "177.45",
    "SMA_50": "175.12",
    "Signal": "0.421"
  },
  "timestamp": "2025-01-07T10:30:00Z"
}
\`\`\`

### GET /api/portfolio
Get current paper trading portfolio.

### GET /api/trades
Get trade history.

### GET /api/metrics
Get monitoring metrics and system health.

## Architecture

\`\`\`
scripts/
  train_sp500_real.py     # Train on 100 S&P 500 companies (20 years)
  train_and_evaluate.py   # Original training script
  validate_ticker.py      # Validate ticker symbols
  predict_live.py         # Make predictions with trained model
  data_ingestion.py       # Fetch 20 years of data
  feature_engineering.py  # Technical indicators
  models.py               # LightGBM, ensemble, training
  backtesting.py          # Trading simulator

app/
  api/
    predict/route.ts      # ✅ WORKING: Validates tickers online, fetches real data
    portfolio/route.ts    # ✅ WORKING: Real-time prices from Yahoo Finance
    train/route.ts        # Checks for training results
    trades/route.ts       # Trade log
    metrics/route.ts      # Monitoring dashboard
  page.tsx                # ✅ WORKING: Auto-updating demo, proper validation

components/
  prediction-panel.tsx    # ✅ WORKING: Displays real predictions with error handling
  portfolio-view.tsx      # ✅ WORKING: Auto-refreshes every 30 seconds
  trade-history.tsx       # Trade log
  metrics-dashboard.tsx   # Performance metrics

data/
  training_results.json   # Training metrics (after running train_sp500_real.py)
\`\`\`

## What Changed

### Before (Broken):
- ❌ Accepted fake tickers like "ILIKEDOGS"
- ❌ Returned random numbers
- ❌ Portfolio had hardcoded dates
- ❌ Crashed when ticker field was empty
- ❌ No real data validation

### Now (Fixed):
- ✅ Validates tickers using Yahoo Finance API
- ✅ Returns real predictions from live market data
- ✅ Portfolio uses today's date and real prices
- ✅ No crashes - proper error handling everywhere
- ✅ Auto-updating demo with live predictions
- ✅ All data is real - no fake numbers!

## Troubleshooting

**"Invalid ticker" error:**
The system checked Yahoo Finance and the ticker doesn't exist. Make sure it's a valid stock symbol (e.g., AAPL, not ILIKEDOGS).

**"Insufficient historical data" error:**
The ticker is valid but doesn't have enough trading history. Try a different ticker.

**Predictions not showing:**
Make sure you entered a valid ticker and clicked "Get Prediction". Check the browser console for errors.

**Portfolio not updating:**
The portfolio auto-refreshes every 30 seconds. Click the "Refresh" button for immediate update.

**Want to train the model:**
Click on `scripts/train_sp500_real.py` in the file explorer and click "Run". Training takes 5-10 minutes.

## Important Notes

1. **Works Immediately**: No training required to get predictions - they work out of the box!
2. **Real Data Only**: All predictions use actual market data from Yahoo Finance
3. **Optional Training**: For advanced users, train on 100 S&P 500 companies for better accuracy
4. **Not Financial Advice**: This is for educational/research purposes only

## Disclaimer

This system is for informational, educational, and backtesting purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Use at your own risk.
