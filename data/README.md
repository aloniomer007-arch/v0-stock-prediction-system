# Training Data Directory

This directory stores the trained model and results from the training script.

## Files Generated After Training:

- `results.json` - Training metrics and accuracy results
- `model.txt` or `model.pkl` - Trained model file
- Other temporary data files

## How to Train:

1. Click the "Run Training Script" button in the UI
2. The script `scripts/train_model.py` will execute
3. It downloads 5 years of real stock data from 30 major companies
4. Trains a real ML model (LightGBM or GradientBoosting)
5. Saves results here for the UI to display

## Expected Results:

- Direction Accuracy: 52-58% (>50% is better than random)
- Training takes 2-5 minutes
- Results persist across sessions
