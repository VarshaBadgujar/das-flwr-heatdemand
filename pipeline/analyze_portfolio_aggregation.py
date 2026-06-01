#!/usr/bin/env python3
"""
Portfolio Aggregation Analysis (Section 4.3.1)
==============================================
Sums per-building predictions across all 250 buildings to produce
a portfolio-level demand forecast — demonstrating what the housing
company would see from the operational channel.

Reads from: logs/predictions/*.npz (y_actual, y_pred per building)
Reads from: data/processed/*.parquet (for timestamps via test split)
Outputs:    logs/portfolio_forecast_analysis.txt (summary stats)
            das-fl-paper/paper/figures/portfolio_aggregation.pdf (figure)

Usage:
    python analyze_portfolio_aggregation.py
    python analyze_portfolio_aggregation.py --data-dir benchmark/aalborg/data/processed --building-ids-file benchmark/aalborg/logs/building_ids_50.txt --suffix aalborg
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
PREDICTIONS_DIR = Path("logs/predictions")
DATA_DIR = Path("data/processed")
BUILDING_IDS_FILE = Path("logs/fl_building_ids_250.txt")
FIGURES_DIR = Path("das-fl-paper/paper/figures")
LOGS_DIR = Path("logs")
ANONYMISE = True  # Use B001, B002 etc.

# Scenarios to compare
SCENARIOS = {
    "local_mlp": "Local MLP",
    "personalised_fedadam": "Personalised FL",
    "fedadam": "FedAdam Global",
    "centralised_mlp": "Centralised MLP",
}


def get_test_timestamps(building_id, data_dir):
    """Reconstruct test set timestamps from parquet using 60/15/10/15 split."""
    parquet_path = data_dir / f"{building_id}.parquet"
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    n = len(df)
    test_start = int(n * 0.85)
    return df.index[test_start:]


def load_building_predictions(building_id, scenario):
    """Load y_actual and y_pred for one building and scenario."""
    npz_path = PREDICTIONS_DIR / f"{scenario}_{building_id}_h1.npz"
    if not npz_path.exists():
        return None, None
    data = np.load(npz_path)
    return data['y_actual'], data['y_pred']


def main():
    parser = argparse.ArgumentParser(description="Portfolio aggregation analysis")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--building-ids-file", type=str, default=str(BUILDING_IDS_FILE))
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--week-start", type=str, default=None,
                        help="Start date for weekly zoom (YYYY-MM-DD), auto-selects winter week if not given")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    suffix = f"_{args.suffix}" if args.suffix else ""

    # Load building IDs
    with open(args.building_ids_file) as f:
        building_ids = [line.strip() for line in f if line.strip()]
    print(f"Buildings: {len(building_ids)}", flush=True)

    # ============================================================
    # 1. Load all predictions and align by timestamp
    # ============================================================
    # Use UNION of timestamps (buildings have different test periods)
    # Sum whatever buildings have data at each hour
    all_timestamps = set()
    valid_buildings = []
    building_series = {}  # bid -> (Series actual, Series pred)

    for bid in building_ids:
        ts = get_test_timestamps(bid, data_dir)
        if ts is None:
            continue
        y_actual, y_pred = load_building_predictions(bid, "local_mlp")
        if y_actual is None:
            continue
        min_len = min(len(ts), len(y_actual))
        ts = ts[:min_len]
        all_timestamps.update(ts)
        building_series[bid] = (
            pd.Series(y_actual[:min_len], index=ts),
            pd.Series(y_pred[:min_len], index=ts),
        )
        valid_buildings.append(bid)

    if len(valid_buildings) == 0:
        print("ERROR: No valid buildings found with predictions", flush=True)
        sys.exit(1)

    common_ts = sorted(all_timestamps)
    print(f"Valid buildings: {len(valid_buildings)}", flush=True)
    print(f"Total hours (union): {len(common_ts)}", flush=True)
    print(f"Period: {common_ts[0]} to {common_ts[-1]}", flush=True)

    # Count buildings contributing per hour
    ts_index_full = pd.DatetimeIndex(common_ts)
    bldg_count = pd.Series(0, index=ts_index_full, dtype=int)
    for bid in valid_buildings:
        s_actual, _ = building_series[bid]
        bldg_count[bldg_count.index.isin(s_actual.index)] += 1
    
    # Use hours where at least 80% of buildings have data
    threshold = int(len(valid_buildings) * 0.80)
    good_mask = bldg_count >= threshold
    common_ts = list(ts_index_full[good_mask])
    print(f"Hours with ≥80% buildings ({threshold}+): {len(common_ts)}", flush=True)
    if len(common_ts) == 0:
        # Fallback: use hours where at least 50% have data
        threshold = int(len(valid_buildings) * 0.50)
        good_mask = bldg_count >= threshold
        common_ts = list(ts_index_full[good_mask])
        print(f"Fallback — hours with ≥50% buildings ({threshold}+): {len(common_ts)}", flush=True)
    if len(common_ts) == 0:
        print("ERROR: No overlapping hours found", flush=True)
        sys.exit(1)
    print(f"Analysis period: {common_ts[0]} to {common_ts[-1]}", flush=True)

    # ============================================================
    # 2. Aggregate per scenario
    # ============================================================
    results = {}

    for scenario, label in SCENARIOS.items():
        sum_actual = np.zeros(len(common_ts))
        sum_pred = np.zeros(len(common_ts))
        n_loaded = 0

        for bid in valid_buildings:
            ts = get_test_timestamps(bid, data_dir)
            y_actual, y_pred = load_building_predictions(bid, scenario)
            if y_actual is None:
                continue

            min_len = min(len(ts), len(y_actual))
            ts = ts[:min_len]
            y_actual = y_actual[:min_len]
            y_pred = y_pred[:min_len]

            # Create Series for alignment
            s_actual = pd.Series(y_actual, index=ts)
            s_pred = pd.Series(y_pred, index=ts)

            # Align to common timestamps
            s_actual = s_actual.reindex(common_ts)
            s_pred = s_pred.reindex(common_ts)

            # Skip if too many NaN
            if s_actual.isna().sum() > len(common_ts) * 0.1:
                continue

            # Fill small gaps
            s_actual = s_actual.fillna(0)
            s_pred = s_pred.fillna(0)

            sum_actual += s_actual.values
            sum_pred += s_pred.values
            n_loaded += 1

        if n_loaded == 0:
            print(f"  {label}: No predictions found — skipping", flush=True)
            continue

        portfolio_mae = np.mean(np.abs(sum_actual - sum_pred))
        portfolio_mape = np.mean(np.abs(sum_actual - sum_pred) / np.clip(sum_actual, 1, None)) * 100
        portfolio_r2 = 1 - np.sum((sum_actual - sum_pred)**2) / np.sum((sum_actual - np.mean(sum_actual))**2)

        results[scenario] = {
            'label': label,
            'sum_actual': sum_actual,
            'sum_pred': sum_pred,
            'mae': portfolio_mae,
            'mape': portfolio_mape,
            'r2': portfolio_r2,
            'n_buildings': n_loaded,
        }

        print(f"  {label}: {n_loaded} buildings, "
              f"Portfolio MAE={portfolio_mae:.1f} kWh/h, "
              f"MAPE={portfolio_mape:.1f}%, R²={portfolio_r2:.3f}", flush=True)

    if not results:
        print("ERROR: No scenario predictions loaded", flush=True)
        sys.exit(1)

    # ============================================================
    # 3. Generate figures
    # ============================================================
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ts_index = pd.DatetimeIndex(common_ts)

    # Auto-select winter week if not specified
    if args.week_start:
        week_start = pd.Timestamp(args.week_start)
    else:
        # Find a December/January week
        winter_mask = (ts_index.month == 12) | (ts_index.month == 1)
        if winter_mask.any():
            winter_start = ts_index[winter_mask][0]
            # Find Monday
            days_to_monday = winter_start.weekday()
            week_start = winter_start - pd.Timedelta(days=days_to_monday)
        else:
            week_start = ts_index[0]

    week_end = week_start + pd.Timedelta(days=7)
    week_mask = (ts_index >= week_start) & (ts_index < week_end)

    # --- Figure A: Full test period (daily aggregation) ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Daily sums
    df_daily = pd.DataFrame({'timestamp': ts_index})
    ref_scenario = list(results.keys())[0]
    df_daily['actual'] = results[ref_scenario]['sum_actual']
    df_daily = df_daily.set_index('timestamp').resample('D').sum()

    axes[0].plot(df_daily.index, df_daily['actual'], 'k-', linewidth=1.5, label='Actual', alpha=0.8)

    colors = {'local_mlp': '#9467bd', 'personalised_fedadam': '#2ca02c',
              'fedadam': '#d62728', 'centralised_mlp': '#ff7f0e'}
    linestyles = {'local_mlp': '--', 'personalised_fedadam': '-',
                  'fedadam': ':', 'centralised_mlp': '-.'}

    for scenario, res in results.items():
        df_pred = pd.DataFrame({'timestamp': ts_index, 'pred': res['sum_pred']})
        df_pred = df_pred.set_index('timestamp').resample('D').sum()
        axes[0].plot(df_pred.index, df_pred['pred'],
                     color=colors.get(scenario, 'gray'),
                     linestyle=linestyles.get(scenario, '-'),
                     linewidth=1.2, label=f"{res['label']} (MAE={res['mae']:.0f})",
                     alpha=0.8)

    axes[0].set_ylabel('Portfolio demand (kWh/day)', fontsize=12)
    axes[0].set_title(f'(a) Daily portfolio demand — {len(valid_buildings)} buildings, '
                      f'test period', fontsize=13)
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Residuals (hourly, for best scenario)
    best_scenario = min(results, key=lambda s: results[s]['mae'])
    residuals = results[best_scenario]['sum_actual'] - results[best_scenario]['sum_pred']
    axes[1].fill_between(ts_index, residuals, alpha=0.4, color=colors.get(best_scenario, 'green'))
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Residual (kWh/h)', fontsize=11)
    axes[1].set_title(f'(b) Hourly residual — {results[best_scenario]["label"]}', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"portfolio_aggregation{suffix}.pdf"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fig_path}", flush=True)

    # --- Figure B: Winter week zoom (hourly) ---
    if week_mask.any():
        fig, ax = plt.subplots(figsize=(14, 5))

        week_ts = ts_index[week_mask]
        ax.plot(week_ts, results[ref_scenario]['sum_actual'][week_mask],
                'k-', linewidth=1.8, label='Actual')

        for scenario, res in results.items():
            ax.plot(week_ts, res['sum_pred'][week_mask],
                    color=colors.get(scenario, 'gray'),
                    linestyle=linestyles.get(scenario, '-'),
                    linewidth=1.2,
                    label=f"{res['label']}")

        ax.set_ylabel('Portfolio demand (kWh/h)', fontsize=12)
        ax.set_title(f'Portfolio demand — winter week '
                     f'({week_start.strftime("%d %b %Y")})', fontsize=13)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a'))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path2 = FIGURES_DIR / f"portfolio_week_zoom{suffix}.pdf"
        plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path2}", flush=True)

    # ============================================================
    # 4. Save summary report
    # ============================================================
    report_path = LOGS_DIR / f"portfolio_forecast_analysis{suffix}.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PORTFOLIO AGGREGATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Buildings: {len(valid_buildings)}\n")
        f.write(f"Common test hours: {len(common_ts)}\n")
        f.write(f"Period: {common_ts[0]} to {common_ts[-1]}\n")
        f.write(f"Total actual demand: {np.sum(results[ref_scenario]['sum_actual']):.0f} kWh\n")
        f.write(f"Mean hourly portfolio demand: {np.mean(results[ref_scenario]['sum_actual']):.1f} kWh/h\n")
        f.write(f"Peak hourly portfolio demand: {np.max(results[ref_scenario]['sum_actual']):.1f} kWh/h\n\n")

        f.write(f"{'Scenario':<25} {'MAE (kWh/h)':<15} {'MAPE %':<10} {'R²':<10} {'Buildings':<10}\n")
        f.write("-" * 70 + "\n")
        for scenario, res in sorted(results.items(), key=lambda x: x[1]['mae']):
            f.write(f"{res['label']:<25} {res['mae']:<15.1f} {res['mape']:<10.1f} "
                    f"{res['r2']:<10.3f} {res['n_buildings']:<10}\n")

        f.write("\n\nPAPER TEXT SUGGESTION:\n")
        f.write("-" * 40 + "\n")
        best = results[min(results, key=lambda s: results[s]['mae'])]
        f.write(f'"At the portfolio level, aggregating {best["n_buildings"]} per-building\n')
        f.write(f'forecasts from {best["label"]} yields a fleet demand forecast with\n')
        f.write(f'MAE = {best["mae"]:.1f} kWh/h and R² = {best["r2"]:.3f}, demonstrating\n')
        f.write(f'that the operational channel delivers actionable portfolio-level\n')
        f.write(f'demand intelligence without centralised data access."\n')

    print(f"Saved: {report_path}", flush=True)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()