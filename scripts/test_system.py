"""
Test the entire system end-to-end:
1. Train the model on real data
2. Show accuracy metrics
3. Test predictions on new tickers
"""

import subprocess
import sys
import json

print("="*80)
print("STOCK PREDICTION SYSTEM - END-TO-END TEST")
print("="*80)

print("\n[STEP 1] Training model on 20 years of real data...")
print("This will take several minutes. Please wait...\n")

# Run training
result = subprocess.run([sys.executable, 'scripts/train_and_evaluate.py'], 
                       capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("Warnings/Errors:", result.stderr)

# Check if training succeeded
try:
    with open('data/training_results.json', 'r') as f:
        training_results = json.load(f)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - ACCURACY REPORT")
    print("="*80)
    
    test_metrics = training_results['test_metrics']
    baseline = training_results['baseline_metrics']
    cv = training_results['cross_validation']
    
    print(f"\nTest Set Performance:")
    print(f"  Direction Accuracy: {test_metrics['direction_accuracy']:.2%}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  Samples Tested: {test_metrics['n_samples']:,}")
    
    print(f"\nCross-Validation (Walk-Forward):")
    print(f"  Avg Direction Accuracy: {cv['avg_direction_accuracy']:.2%}")
    print(f"  Avg RMSE: {cv['avg_rmse']:.6f}")
    
    print(f"\nImprovement vs Baseline:")
    improvement = (test_metrics['direction_accuracy'] / baseline['direction_accuracy'] - 1) * 100
    print(f"  Direction Lift: {improvement:+.1f}%")
    
    print(f"\n{'='*80}")
    print(f"VERDICT: {'✓ GOOD' if test_metrics['direction_accuracy'] > 0.53 else '✗ NEEDS IMPROVEMENT'}")
    print(f"{'='*80}")
    print(f"\nNote: Direction accuracy >50% = beats random guess")
    print(f"      Direction accuracy >53% = considered good for stock prediction")
    print(f"      Current model: {test_metrics['direction_accuracy']:.2%}")
    
except FileNotFoundError:
    print("\n❌ ERROR: Training failed or results not saved")
    sys.exit(1)

print("\n" + "="*80)
print("[STEP 2] Testing predictions on new ticker...")
print("="*80)

# Test on a ticker not in training
test_ticker = 'AAPL'
print(f"\nTesting prediction for {test_ticker}...")

prediction_input = json.dumps({'ticker': test_ticker, 'horizon': '1d'})
proc = subprocess.Popen(
    [sys.executable, 'scripts/predict_live.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

stdout, stderr = proc.communicate(input=prediction_input.encode())

try:
    prediction = json.loads(stdout.decode())
    
    if 'error' in prediction:
        print(f"❌ Prediction failed: {prediction['error']}")
    else:
        print(f"\n✓ Prediction successful!")
        print(f"  Current Price: ${prediction['current_price']:.2f}")
        print(f"  Predicted Return: {prediction['predicted_return']*100:+.2f}%")
        print(f"  Predicted Price: ${prediction['predicted_price']:.2f}")
        print(f"  95% CI: [{prediction['lower_bound']*100:.2f}%, {prediction['upper_bound']*100:.2f}%]")
        
except json.JSONDecodeError:
    print(f"❌ Failed to parse prediction")
    print(f"Output: {stdout.decode()}")
    print(f"Errors: {stderr.decode()}")

print("\n" + "="*80)
print("SYSTEM TEST COMPLETE")
print("="*80)
print("\nYou can now use the web interface to make predictions!")
print("The model has been trained and is ready to use.\n")
