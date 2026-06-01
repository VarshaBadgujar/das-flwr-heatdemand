# This script extracts and summarizes the computational cost (training time) for different scenarios from the logs, and prints a summary for the paper.
import re

# Parse the actual timing from logs
print("=== COMPUTATIONAL COST PER SCENARIO ===\n")
print(f"{'Scenario':<30} {'Time (min)':>10} {'Time (hrs)':>10}")
print("-" * 55)

# From experiment_log_v4.txt: 3 sequential runs
# FedAvg, FedAdam, Pers FedProx
v4_times = [
    ('FedAvg (250 bldgs, 20 rnd)', 59.6),
    ('FedAdam (250 bldgs, 20 rnd)', 58.0),
    ('Pers FedProx (250 bldgs, 20 rnd)', 65.5),
]

# From experiment_log_fedadam_v4.txt
v4_times.append(('Pers FedAdam (250 bldgs, 20 rnd)', 65.4))

# From experiment_log_remaining_v4.txt
v4_times.append(('FedProx standalone (250 bldgs, 20 rnd)', 48.8))

# From experiment_log_horizons.txt: 15 runs, each ~47 min
# Average per horizon run
v4_times.append(('FL per horizon avg (250 bldgs, 20 rnd)', (46.4+46.7+48.6)/3))

for name, mins in v4_times:
    print(f"  {name:<30} {mins:>8.1f} min {mins/60:>8.1f} hrs")

# Summary for paper
print(f"\n{'='*55}")
print("FOR THE PAPER:")
print(f"{'='*55}")
print(f"  FL training (20 rounds, 250 clients): ~60 min")
print(f"  Personalisation (fine-tuning 250 bldgs): ~7 min")
print(f"    (included in the ~65 min personalised total)")
print(f"  Local MLP (250 buildings): need to check")
print(f"  Centralised MLP: need to check")
print(f"\n  Hardware: DGX Station, 4x Tesla V100-DGXS-32GB")
print(f"  Docker: NGC TF 2.14, CUDA 12.3")

# Check local MLP and centralised timing
import os
for logfile in ['logs/local_mlp_250_log.txt', 'logs/centralised_mlp_log.txt']:
    if os.path.exists(logfile):
        with open(logfile) as f:
            content = f.read()
        times = re.findall(r'(\d+\.?\d*)s \((\d+\.?\d*) min\)', content)
        if times:
            print(f"\n  {logfile}: {times[-1][1]} min")