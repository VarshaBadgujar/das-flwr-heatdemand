#!/usr/bin/env python3
"""
Complete Results Table Generator
==============================================================
Generates the main results table across all 9 scenarios and 4 horizons.

Output:
  - logs/complete_results_table.csv      (machine-readable)
  - logs/complete_results_table.txt      (formatted text)
  - logs/complete_results_table.tex      (LaTeX for paper)
  - Console output

Usage:
    cd ~/Desktop/varsha_projects/das-flwr-heatdemand
    python pipeline/generate_results_table.py
    python pipeline/generate_results_table.py --include-median
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

EXCLUDE = [B001, B037, B238, B028]
HORIZONS = [1, 6, 24, 168]

# ============================================================
# FILE MAPPING
# ============================================================

# All-in-one files (contain all horizons in one CSV)
ALL_IN_ONE = {
    'Centralised XGB':  'centralised_xgboost_results_v2.csv',
    'Local MLP':        'local_mlp_matched_results_250.csv',
    'Local XGB':        'local_baseline_results_v2.csv',
    'Centralised MLP':  'centralised_mlp_results_v2.csv',
}

# Per-horizon files (t+1 in base file, others in _h{horizon} files)
PER_HORIZON = {
    'FedAvg': {
        1:   'fl_fedavg_final_v4.csv',
        6:   'fl_fedavg_final_v4_h6.csv',
        24:  'fl_fedavg_final_v4_h24.csv',
        168: 'fl_fedavg_final_v4_h168.csv',
    },
    'FedAdam': {
        1:   'fl_fedadam_final_v4.csv',
        6:   'fl_fedadam_final_v4_h6.csv',
        24:  'fl_fedadam_final_v4_h24.csv',
        168: 'fl_fedadam_final_v4_h168.csv',
    },
    'FedProx': {
        1:   'fl_fedprox_final_v4.csv',
        6:   'fl_fedprox_final_v4_h6.csv',
        24:  'fl_fedprox_final_v4_h24.csv',
        168: 'fl_fedprox_final_v4_h168.csv',
    },
    'Pers FL (FedAdam)': {
        1:   'fl_personalised_final_fedadam_v4.csv',
        6:   'fl_personalised_final_fedadam_v4_h6.csv',
        24:  'fl_personalised_final_fedadam_v4_h24.csv',
        168: 'fl_personalised_final_fedadam_v4_h168.csv',
    },
    'Pers FL (FedProx)': {
        1:   'fl_personalised_final_fedprox_v4.csv',
        6:   'fl_personalised_final_fedprox_v4_h6.csv',
        24:  'fl_personalised_final_fedprox_v4_h24.csv',
        168: 'fl_personalised_final_fedprox_v4_h168.csv',
    },
}

# Scenario display order (best to worst at t+1)
SCENARIO_ORDER = [
    'Centralised XGB',
    'Local MLP',
    'Pers FL (FedAdam)',
    'Pers FL (FedProx)',
    'Local XGB',
    'Centralised MLP',
    'FedProx',
    'FedAvg',
    'FedAdam',
]

# Category for each scenario
CATEGORIES = {
    'Centralised XGB':   'Centralised',
    'Local MLP':         'Local',
    'Pers FL (FedAdam)': 'FL (personalised)',
    'Pers FL (FedProx)': 'FL (personalised)',
    'Local XGB':         'Local',
    'Centralised MLP':   'Centralised',
    'FedProx':           'FL (global)',
    'FedAvg':            'FL (global)',
    'FedAdam':           'FL (global)',
}


def get_metrics(fpath, horizon, exclude, metric='mean'):
    """Load CSV and compute metrics for a given horizon."""
    df = pd.read_csv(fpath)
    df = df[(df['horizon'] == horizon) & (~df['building_id'].isin(exclude))]
    
    if len(df) == 0:
        return None, None, None, None
    
    if metric == 'mean':
        return df['mae'].mean(), df['r2'].mean(), df['rmse'].mean() if 'rmse' in df.columns else None, len(df)
    else:
        return df['mae'].median(), df['r2'].median(), df['rmse'].median() if 'rmse' in df.columns else None, len(df)


def collect_all_results(metric='mean'):
    """Collect results for all scenarios and horizons."""
    results = []
    
    for name in SCENARIO_ORDER:
        for h in HORIZONS:
            # Determine file path
            if name in ALL_IN_ONE:
                fpath = LOG_DIR / ALL_IN_ONE[name]
            elif name in PER_HORIZON:
                fname = PER_HORIZON[name].get(h)
                fpath = LOG_DIR / fname if fname else None
            else:
                fpath = None
            
            if fpath and fpath.exists():
                mae, r2, rmse, n = get_metrics(str(fpath), h, EXCLUDE, metric)
                results.append({
                    'scenario': name,
                    'category': CATEGORIES[name],
                    'horizon': h,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse,
                    'n_buildings': n,
                })
            else:
                results.append({
                    'scenario': name,
                    'category': CATEGORIES[name],
                    'horizon': h,
                    'mae': None,
                    'r2': None,
                    'rmse': None,
                    'n_buildings': None,
                })
    
    return pd.DataFrame(results)


def print_text_table(df, title=""):
    """Print formatted text table."""
    lines = []
    lines.append(f"\n{'='*100}")
    lines.append(title)
    lines.append(f"{'='*100}")
    
    header = f"{'#':<3} {'Scenario':<22} {'Category':<18}"
    for h in HORIZONS:
        header += f" {'t+'+str(h)+' MAE':>9} {'R²':>7}"
    lines.append(header)
    lines.append("-" * 100)
    
    for i, name in enumerate(SCENARIO_ORDER):
        sdf = df[df['scenario'] == name]
        row = f"{i+1:<3} {name:<22} {CATEGORIES[name]:<18}"
        for h in HORIZONS:
            hdf = sdf[sdf['horizon'] == h]
            if len(hdf) > 0 and hdf.iloc[0]['mae'] is not None:
                mae = hdf.iloc[0]['mae']
                r2 = hdf.iloc[0]['r2']
                row += f" {mae:>9.3f} {r2:>7.3f}"
            else:
                row += f" {'—':>9} {'—':>7}"
        lines.append(row)
    
    text = '\n'.join(lines)
    print(text, flush=True)
    return text


def generate_latex_table(df):
    """Generate LaTeX table for Springer LNCS."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean MAE (kWh/h) and $R^2$ across nine scenarios and four forecast horizons (246 buildings, excluding seasonal-mismatch cases).}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{llcccccccc}")
    lines.append(r"\hline")
    lines.append(r"\# & Scenario & \multicolumn{2}{c}{$h$=1} & \multicolumn{2}{c}{$h$=6} & \multicolumn{2}{c}{$h$=24} & \multicolumn{2}{c}{$h$=168} \\")
    lines.append(r" & & MAE & $R^2$ & MAE & $R^2$ & MAE & $R^2$ & MAE & $R^2$ \\")
    lines.append(r"\hline")
    
    prev_category = None
    for i, name in enumerate(SCENARIO_ORDER):
        cat = CATEGORIES[name]
        if cat != prev_category and prev_category is not None:
            lines.append(r"\hdashline")
        prev_category = cat
        
        sdf = df[df['scenario'] == name]
        
        # Short name for table
        short_names = {
            'Centralised XGB': 'Centr.\\ XGB',
            'Local MLP': 'Local MLP',
            'Pers FL (FedAdam)': 'Pers.\\ FL (FedAdam)',
            'Pers FL (FedProx)': 'Pers.\\ FL (FedProx)',
            'Local XGB': 'Local XGB',
            'Centralised MLP': 'Centr.\\ MLP',
            'FedProx': 'FedProx (global)',
            'FedAvg': 'FedAvg (global)',
            'FedAdam': 'FedAdam (global)',
        }
        
        row = f"{i+1} & {short_names.get(name, name)}"
        
        for h in HORIZONS:
            hdf = sdf[sdf['horizon'] == h]
            if len(hdf) > 0 and hdf.iloc[0]['mae'] is not None:
                mae = hdf.iloc[0]['mae']
                r2 = hdf.iloc[0]['r2']
                
                # Bold the best MAE per horizon
                all_mae_h = df[(df['horizon'] == h) & (df['mae'].notna())]['mae']
                is_best = (mae == all_mae_h.min())
                
                if is_best:
                    row += f" & \\textbf{{{mae:.3f}}} & \\textbf{{{r2:.3f}}}"
                else:
                    row += f" & {mae:.3f} & {r2:.3f}"
            else:
                row += r" & --- & ---"
        
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate complete results table")
    parser.add_argument("--include-median", action='store_true',
                        help="Also generate median table")
    args = parser.parse_args()
    
    print("=" * 60, flush=True)
    print("GENERATING COMPLETE RESULTS TABLE", flush=True)
    print("=" * 60, flush=True)
    
    # Mean table
    df_mean = collect_all_results(metric='mean')
    text_mean = print_text_table(df_mean, "MEAN METRICS (246 buildings, excluding 4 outliers)")
    
    # Save CSV
    csv_path = LOG_DIR / "complete_results_table.csv"
    df_mean.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}", flush=True)
    
    # Save text
    txt_path = LOG_DIR / "complete_results_table.txt"
    with open(txt_path, 'w') as f:
        f.write(text_mean)
    print(f"Text saved: {txt_path}", flush=True)
    
    # Generate and save LaTeX
    latex = generate_latex_table(df_mean)
    tex_path = LOG_DIR / "complete_results_table.tex"
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX saved: {tex_path}", flush=True)
    
    # Median table (if requested)
    if args.include_median:
        df_median = collect_all_results(metric='median')
        print_text_table(df_median, "MEDIAN METRICS (246 buildings, excluding 4 outliers)")
        
        csv_med_path = LOG_DIR / "complete_results_table_median.csv"
        df_median.to_csv(csv_med_path, index=False)
        print(f"\nMedian CSV saved: {csv_med_path}", flush=True)
    
    # Key findings summary
    print(f"\n{'='*60}", flush=True)
    print("KEY FINDINGS", flush=True)
    print(f"{'='*60}", flush=True)
    
    h1 = df_mean[df_mean['horizon'] == 1].dropna(subset=['mae'])
    best = h1.loc[h1['mae'].idxmin()]
    worst_mlp = h1[h1['scenario'].str.contains('MLP|Pers|Fed')].loc[
        h1[h1['scenario'].str.contains('MLP|Pers|Fed')]['mae'].idxmax()]
    
    print(f"  Best overall (t+1):     {best['scenario']} — MAE={best['mae']:.3f}", flush=True)
    
    pers_adam = h1[h1['scenario'] == 'Pers FL (FedAdam)'].iloc[0]
    pers_prox = h1[h1['scenario'] == 'Pers FL (FedProx)'].iloc[0]
    local_mlp = h1[h1['scenario'] == 'Local MLP'].iloc[0]
    centr_mlp = h1[h1['scenario'] == 'Centralised MLP'].iloc[0]
    
    gap_adam = (pers_adam['mae'] - local_mlp['mae']) / local_mlp['mae'] * 100
    gap_prox = (pers_prox['mae'] - local_mlp['mae']) / local_mlp['mae'] * 100
    centr_gap = (centr_mlp['mae'] - local_mlp['mae']) / local_mlp['mae'] * 100
    
    print(f"  Pers FL (FedAdam) vs Local MLP: {gap_adam:+.1f}% (MAE {pers_adam['mae']:.3f} vs {local_mlp['mae']:.3f})", flush=True)
    print(f"  Pers FL (FedProx) vs Local MLP: {gap_prox:+.1f}% (MAE {pers_prox['mae']:.3f} vs {local_mlp['mae']:.3f})", flush=True)
    print(f"  Centralised MLP vs Local MLP:   {centr_gap:+.1f}% (pooling hurts)", flush=True)
    
    # Horizon degradation
    print(f"\n  Horizon degradation (t+1 → t+168):", flush=True)
    for name in ['Local MLP', 'Pers FL (FedAdam)', 'Pers FL (FedProx)']:
        sdf = df_mean[df_mean['scenario'] == name].dropna(subset=['mae'])
        if len(sdf) >= 2:
            h1_mae = sdf[sdf['horizon'] == 1]['mae'].values[0]
            h168_mae = sdf[sdf['horizon'] == 168]['mae'].values[0]
            degrad = (h168_mae - h1_mae) / h1_mae * 100
            print(f"    {name:<22}: {h1_mae:.3f} → {h168_mae:.3f} ({degrad:+.1f}%)", flush=True)
    
    print(f"\n  At t+168 (weekly planning):", flush=True)
    h168 = df_mean[df_mean['horizon'] == 168].dropna(subset=['mae'])
    for _, row in h168.sort_values('mae').iterrows():
        print(f"    {row['scenario']:<22}: MAE={row['mae']:.3f}, R²={row['r2']:.3f}", flush=True)


if __name__ == "__main__":
    main()