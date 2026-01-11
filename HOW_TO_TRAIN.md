# How to Train the Stock Prediction Model

## The Problem

The training needs to run Python scripts that download 20 years of data for 100-500 companies. This takes significant time and computational resources, and **cannot run directly in the browser**.

## Solution: Train Locally

### Step 1: Download the Project

Click the three dots in the top right of v0 and select "Download ZIP" or push to GitHub.

### Step 2: Install Python Dependencies

\`\`\`bash
pip install pandas numpy scikit-learn lightgbm yfinance
\`\`\`

### Step 3: Run the Training Script

\`\`\`bash
python3 scripts/train_sp500.py
\`\`\`

This will:
- Download 20 years of historical data for 100+ S&P 500 companies
- Train a LightGBM model with proper validation
- Save results to `data/training_results.json`
- Print real accuracy metrics

### Step 4: Deploy with Trained Model

Once training completes, you'll have:
- `data/model.pkl` - Trained model
- `data/training_results.json` - Accuracy metrics

Deploy the project to Vercel with these files included, and the predictions will use the real trained model.

## What the Training Does

- **Downloads Real Data**: 20 years of OHLCV data + fundamentals from Yahoo Finance
- **Feature Engineering**: Calculates technical indicators (SMA, EMA, RSI, MACD, ATR)
- **Model Training**: LightGBM with walk-forward validation
- **Real Metrics**: Direction accuracy, RMSE, and other performance metrics

## Expected Results

- **Direction Accuracy**: 52-58% (anything above 50% beats random guessing)
- **Training Time**: 10-30 minutes depending on number of companies
- **Data Points**: Typically 500,000+ samples across all companies

## Alternative: Use Pre-trained Model

For testing purposes, you can use a pre-trained model by creating dummy training results in `data/training_results.json`:

\`\`\`json
{
  "direction_accuracy": 55.2,
  "rmse": 0.0234,
  "training_samples": 450000,
  "test_samples": 112500,
  "n_tickers": 100,
  "n_features": 45
}
\`\`\`

But remember: **real predictions require a real trained model**.
