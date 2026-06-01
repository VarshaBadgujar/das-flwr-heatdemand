"""Quick hyperparameter sensitivity test on 20 buildings."""
import subprocess
import re
import time

def run_and_extract(cmd, label):
    """Run command and extract t+1 MAE."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "t+1" in line:
            parts = line.strip().split()
            mae = parts[1] if len(parts) > 1 else "N/A"
            r2 = parts[3] if len(parts) > 3 else "N/A"
            print(f"  {label:<40} MAE={mae}  R²={r2}")
            return
    print(f"  {label:<40} FAILED")

print("=" * 70)
print("HYPERPARAMETER SENSITIVITY ANALYSIS (20 buildings, t+1)")
print("=" * 70)

# Baseline: current settings
print("\n--- FL Local Epochs ---")
for epochs in [1, 3, 5, 10]:
    run_and_extract(
        f"python -u pipeline/run_fl.py --scenario fedavg --n 20 --rounds 10 --horizon 1 --local-epochs {epochs}",
        f"FedAvg local_epochs={epochs}"
    )

print("\n--- Client Learning Rate ---")
for lr in [0.0001, 0.001, 0.01]:
    run_and_extract(
        f"python -u pipeline/run_fl.py --scenario fedavg --n 20 --rounds 10 --horizon 1 --lr {lr}",
        f"FedAvg client_lr={lr}"
    )

print("\n--- Personalised Fine-tune Depth ---")
# For this we need to temporarily modify the fine-tune epochs
# Just test current default for now
run_and_extract(
    "python -u pipeline/run_fl.py --scenario personalised --n 20 --rounds 10 --horizon 1",
    "Personalised (5 fine-tune epochs)"
)

print("\n" + "=" * 70)
print("SENSITIVITY ANALYSIS COMPLETE")
print("=" * 70)