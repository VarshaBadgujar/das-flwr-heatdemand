#!/usr/bin/env python3
"""
DAS-FL: Extract Computation & Communication Costs
==================================================
Run AFTER experiments complete.
Produces a summary table for the paper.

Usage:
    cd ~/Desktop/varsha_projects/das-flwr-heatdemand/
    python pipeline/extract_costs.py
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

print("=" * 65)
print("DAS-FL: COMPUTATION & COMMUNICATION COST SUMMARY")
print("=" * 65)

# ============================================================
# 1. COMPUTATION COST — Extract timing from experiment log
# ============================================================
print("\n1. COMPUTATION COST (Training Time)")
print("-" * 65)

log_file = LOG_DIR / "v5_portfolio_batch_log.txt"
if log_file.exists():
    with open(log_file) as f:
        log_text = f.read()
    
    # Extract timestamps and scenario markers
    # Look for patterns like "Time: fre 27 mar 2026 16:41:29 CET"
    # and "ALL FL EXPERIMENTS COMPLETE — Xs (Y min)"
    # and "FL completed in Xs (Y min)"
    
    # Find scenario sections
    scenarios = []
    current_scenario = None
    
    lines = log_text.split('\n')
    for i, line in enumerate(lines):
        if '--- ' in line and ' ---' in line:
            current_scenario = line.strip().strip('-').strip()
        
        # Look for timing info
        time_match = re.search(r'(\d+)s \((\d+\.?\d*) min\)', line)
        if time_match and current_scenario:
            seconds = int(time_match.group(1))
            minutes = float(time_match.group(2))
            scenarios.append({
                'scenario': current_scenario,
                'seconds': seconds,
                'minutes': minutes,
                'line': line.strip()
            })
        
        # Look for "Time:" timestamps
        if 'STARTED' in line:
            print(f"  Experiment started: {line.strip()}")
        if 'FINISHED' in line:
            print(f"  Experiment finished: {line.strip()}")
    
    if scenarios:
        print(f"\n  {'Scenario':<25} {'Time (s)':>10} {'Time (min)':>12}")
        print(f"  {'-'*25} {'-'*10} {'-'*12}")
        total_s = 0
        for s in scenarios:
            print(f"  {s['scenario']:<25} {s['seconds']:>10} {s['minutes']:>12.1f}")
            total_s += s['seconds']
        print(f"  {'-'*25} {'-'*10} {'-'*12}")
        print(f"  {'TOTAL':<25} {total_s:>10} {total_s/60:>12.1f}")
    else:
        print("  No timing data found in log. Check if experiments completed.")
        print("  Looking for timing in result files instead...")
else:
    print(f"  Log file not found: {log_file}")

# ============================================================
# 2. COMMUNICATION COST — Model weight payload
# ============================================================
print(f"\n2. COMMUNICATION COST (FL Weight Exchange)")
print("-" * 65)

# MLP [64,32] with 51 input features
n_features = 51
# Layer 1: 51 inputs × 64 neurons + 64 biases = 3328
# Layer 2: 64 × 32 + 32 = 2080  
# Layer 3 (output): 32 × 1 + 1 = 33
# Total parameters
params_l1 = n_features * 64 + 64
params_l2 = 64 * 32 + 32
params_l3 = 32 * 1 + 1
total_params = params_l1 + params_l2 + params_l3

# Each parameter = 4 bytes (float32)
bytes_per_param = 4
payload_bytes = total_params * bytes_per_param
payload_kb = payload_bytes / 1024

print(f"  MLP Architecture: [51] → [64] → [32] → [1]")
print(f"  Layer 1 params: {params_l1:,} (input→hidden1)")
print(f"  Layer 2 params: {params_l2:,} (hidden1→hidden2)")
print(f"  Layer 3 params: {params_l3:,} (hidden2→output)")
print(f"  Total parameters: {total_params:,}")
print(f"  Payload per round: {payload_bytes:,} bytes ({payload_kb:.1f} KB)")

# Per-scenario communication
n_rounds = 20
n_buildings_main = 250
n_buildings_list = [20, 50, 100, 200, 250, 500, 988]

print(f"\n  --- Per-round communication (K clients) ---")
print(f"  {'K clients':<12} {'Upload (MB)':>12} {'Download (MB)':>14} {'Total (MB)':>12}")
print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*12}")
for K in [250, 988]:
    # Upload: K clients × payload (client → server)
    upload_mb = K * payload_kb / 1024
    # Download: K clients × payload (server → client) 
    download_mb = K * payload_kb / 1024
    total_mb = upload_mb + download_mb
    print(f"  {K:<12} {upload_mb:>12.2f} {download_mb:>14.2f} {total_mb:>12.2f}")

print(f"\n  --- Total communication ({n_rounds} rounds) ---")
print(f"  {'K clients':<12} {'Total (MB)':>12} {'Total (GB)':>12}")
print(f"  {'-'*12} {'-'*12} {'-'*12}")
for K in [250, 988]:
    total_mb = 2 * K * payload_kb / 1024 * n_rounds
    total_gb = total_mb / 1024
    print(f"  {K:<12} {total_mb:>12.1f} {total_gb:>12.3f}")

# Compare with centralised data transfer
print(f"\n  --- Comparison: FL vs Centralised ---")
# Centralised: must transfer raw data (51 features × 4 bytes × ~35,000 hours per building)
hours_per_building = 35000
raw_data_per_building_mb = 51 * 4 * hours_per_building / (1024 * 1024)
for K in [250, 988]:
    centralised_mb = K * raw_data_per_building_mb
    fl_total_mb = 2 * K * payload_kb / 1024 * n_rounds
    ratio = centralised_mb / fl_total_mb
    print(f"  K={K}:")
    print(f"    Centralised raw data transfer: {centralised_mb:,.0f} MB ({centralised_mb/1024:.1f} GB)")
    print(f"    FL weight transfer (30 rounds): {fl_total_mb:,.1f} MB")
    print(f"    Reduction factor: {ratio:,.0f}× less data transferred")

# ============================================================
# 3. RESULT SUMMARY — Extract from v2 CSV files
# ============================================================
print(f"\n3. EXPERIMENT RESULTS SUMMARY")
print("-" * 65)

v2_files = sorted(glob.glob(str(LOG_DIR / "*_v2.csv")))
if v2_files:
    for fpath in v2_files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath)
            n_rows = len(df)
            
            if 'mae' in df.columns and 'horizon' in df.columns:
                print(f"\n  {fname} ({n_rows} rows)")
                
                # Group by horizon
                for h in sorted(df['horizon'].unique()):
                    hdf = df[df['horizon'] == h]
                    n_bldg = hdf['building_id'].nunique() if 'building_id' in hdf.columns else len(hdf)
                    mae = hdf['mae'].mean()
                    r2 = hdf['r2'].mean() if 'r2' in hdf.columns else float('nan')
                    print(f"    t+{h:<4}  MAE={mae:.3f}  R²={r2:.3f}  ({n_bldg} buildings)")
            elif 'mae' in df.columns:
                print(f"\n  {fname} ({n_rows} rows)")
                mae = df['mae'].mean()
                print(f"    MAE={mae:.3f}")
            else:
                print(f"\n  {fname} ({n_rows} rows) — columns: {list(df.columns)[:5]}")
        except Exception as e:
            print(f"\n  {fname} — Error reading: {e}")
else:
    print("  No _v2.csv files found yet. Experiments may still be running.")

# ============================================================
# 4. SCALABILITY RESULTS
# ============================================================
print(f"\n4. SCALABILITY RESULTS")
print("-" * 65)

scalability_files = sorted(glob.glob(str(LOG_DIR / "fl_scalability_*_v2.csv")))
if scalability_files:
    print(f"\n  {'K':<6} ", end="")
    strategies = []
    for fpath in scalability_files:
        name = os.path.basename(fpath).replace('fl_scalability_', '').replace('_v2.csv', '')
        strategies.append(name)
        print(f" {name:>12}", end="")
    print()
    print(f"  {'-'*6} " + " ".join(['-'*12] * len(strategies)))
    
    # Collect all K values
    all_data = {}
    for fpath in scalability_files:
        name = os.path.basename(fpath).replace('fl_scalability_', '').replace('_v2.csv', '')
        df = pd.read_csv(fpath)
        if 'n_buildings' in df.columns:
            for K in sorted(df['n_buildings'].unique()):
                kdf = df[df['n_buildings'] == K]
                mae = kdf['mae'].mean()
                if K not in all_data:
                    all_data[K] = {}
                all_data[K][name] = mae
    
    for K in sorted(all_data.keys()):
        print(f"  {K:<6} ", end="")
        for s in strategies:
            val = all_data[K].get(s, float('nan'))
            print(f" {val:>12.3f}", end="")
        print()
else:
    print("  No scalability _v2.csv files found yet.")

# ============================================================
# 5. COST TABLE (LaTeX)
# ============================================================
print(f"\n5. COST SENTENCE")
print("-" * 65)
print(f"""
  For the paper, add this to Methodology or Results:

  "Training 250 FL clients for 30 communication rounds on an 
  NVIDIA DGX Station with 4× Tesla V100-DGXS-32GB GPUs completed 
  in [INSERT TIME FROM LOG]. Each FL round exchanges approximately 
  {payload_kb:.0f}~KB of model weights per client, totaling 
  {2 * 250 * payload_kb / 1024 * 30:.0f}~MB across all rounds --- 
  a {250 * raw_data_per_building_mb / (2 * 250 * payload_kb / 1024 * 30):,.0f}$\\times$ 
  reduction compared to centralising raw substation data."
""")

print("\n" + "=" * 65)
print("Done. Run this again after experiments complete for full timing data.")
print("=" * 65)