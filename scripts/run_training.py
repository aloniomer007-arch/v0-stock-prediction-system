import subprocess
import sys

# Run the training script
print("Starting model training on 20 years of real stock data...")
print("This will take a few minutes...\n")

result = subprocess.run([sys.executable, 'scripts/train_and_evaluate.py'], 
                       capture_output=True, 
                       text=True)

print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)
