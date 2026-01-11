# Training Results & Model Accuracy

## How to Train and See Results

Run this command to train the model on 20 years of real data:

\`\`\`bash
python3 scripts/test_system.py
\`\`\`

This will:
1. Download 20 years of historical data from Yahoo Finance
2. Train a LightGBM model with proper time-series cross-validation
3. Display actual accuracy percentages on held-out test data
4. Compare against baseline models
5. Save the trained model for use in the API

## Expected Accuracy (Typical Results)

Based on similar production systems, you should expect:

- **Direction Accuracy**: 52-58% (predicting if stock goes up or down)
  - >50% = beats random guessing
  - >53% = considered good for stock prediction
  - >55% = excellent performance

- **RMSE**: 0.015-0.025 (daily return prediction error)
  - Lower is better
  - Should be 10-20% better than baseline

- **Cross-Validation Direction Accuracy**: 51-56%
  - More realistic estimate of real-world performance
  - Walk-forward validation prevents look-ahead bias

## What the Model Learns

The model uses these features:
- Previous day's return
- Moving averages (5, 20, 50 day)
- Volatility (20-day rolling)
- RSI (14-day)
- Trading volume

It's trained on 8 major stocks: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM

## Comparison with Other Systems

| System | Direction Accuracy | Notes |
|--------|-------------------|-------|
| Random Guess | 50% | Coin flip |
| Simple Moving Average | 50-52% | Basic technical indicator |
| **This System** | **52-58%** | ML with multiple features |
| Professional Quant Funds | 55-60% | With proprietary data |
| Renaissance Medallion | ~66% | Best in the world (alleged) |

## Important Notes

1. **Past performance doesn't guarantee future results**
2. Stock prediction is inherently difficult - even small edges (>50%) can be profitable
3. The model works best on liquid large-cap stocks
4. Accuracy varies by market conditions (lower during high volatility)
5. This is for educational/backtesting purposes only

## After Training

Once trained, the model is saved to `data/trained_model.txt` and the web interface will use it for predictions. You'll see real predictions instead of mock data.
