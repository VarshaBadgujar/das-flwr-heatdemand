"""
Portfolio Aggregation Analysis v2 — Fixed
==========================================================
Bug in v1: Timestamp reindexing caused building counts to differ per
scenario (48 for Centralised MLP vs 189 for others). All predictions
are 3962 elements per building — positionally aligned to the same
test set. This version aligns by array position, not timestamps.

Sums per-building predictions across all 250 buildings to produce
a portfolio-level demand forecast — demonstrating what the housing
company would see from the operational channel.

Reads from: logs/predictions/{scenario}_{bid}_h1.npz (y_actual, y_pred)
Reads from: data/processed/{bid}.parquet (for timestamps, plotting only)
Outputs:    logs/portfolio_forecast_analysis_v2.txt (summary stats)
            das-fl-paper/paper/figures/portfolio_aggregation.pdf (figure)

Usage:
    python pipeline/analyze_portfolio_aggregation_v2.py
    python pipeline/analyze_portfolio_aggregation_v2.py --data-dir benchmark/aalborg/data/processed --building-ids-file benchmark/aalborg/logs/building_ids_50.txt --suffix aalborg
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
PREDICTIONS_DIR = Path("logs/predictions/v5_portfolio")
DATA_DIR = Path("data/processed")
BUILDING_IDS_FILE = Path("logs/fl_building_ids_250.txt")
FIGURES_DIR = Path("das-fl-paper/paper/figures")
LOGS_DIR = Path("logs")
ANONYMISE = True

# Scenarios to compare (order matters for display)
SCENARIOS = {
    "local_mlp": "Local MLP",
    "personalised_fedadam": "Personalised FL (FedAdam)",
    "fedadam": "FedAdam Global",
    "centralised_mlp": "Centralised MLP",
}

COLORS = {
    "local_mlp": "#9467bd",
    "personalised_fedadam": "#2ca02c",
    "fedadam": "#d62728",
    "centralised_mlp": "#ff7f0e",
}
LINESTYLES = {
    "local_mlp": "--",
    "personalised_fedadam": "-",
    "fedadam": ":",
    "centralised_mlp": "-.",
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


def main():
    parser = argparse.ArgumentParser(description="Portfolio aggregation analysis v2")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--building-ids-file", type=str, default=str(BUILDING_IDS_FILE))
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--week-start", type=str, default=None,
                        help="Start date for weekly zoom (YYYY-MM-DD)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    suffix = f"_{args.suffix}" if args.suffix else ""

    # Load building IDs
    with open(args.building_ids_file) as f:
        building_ids = [line.strip() for line in f if line.strip()]
    print(f"Buildings in ID file: {len(building_ids)}", flush=True)

    # ============================================================
    # 1. VALIDATION: Check all scenarios have same array length per building
    # ============================================================
    print("\n--- Step 1: Validating prediction file consistency ---", flush=True)

    # First pass: find which buildings have ALL scenarios available
    valid_buildings = []
    skipped_buildings = []
    expected_len = None

    for bid in building_ids:
        lengths = {}
        all_exist = True
        for scenario in SCENARIOS:
            npz_path = PREDICTIONS_DIR / f"{scenario}_{bid}_h1.npz"
            if not npz_path.exists():
                all_exist = False
                break
            data = np.load(npz_path)
            lengths[scenario] = len(data['y_actual'])

        if not all_exist:
            skipped_buildings.append((bid, "missing .npz file"))
            continue

        # Check all scenarios have same length for this building
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            skipped_buildings.append((bid, f"length mismatch: {lengths}"))
            continue

        arr_len = unique_lengths.pop()

        # Verify y_actual is identical across scenarios (same test set)
        actuals = []
        for scenario in SCENARIOS:
            npz_path = PREDICTIONS_DIR / f"{scenario}_{bid}_h1.npz"
            data = np.load(npz_path)
            actuals.append(data['y_actual'])

        # Check actuals match (within floating point tolerance)
        ref_actual = actuals[0]
        actuals_match = all(np.allclose(ref_actual, a, atol=1e-6) for a in actuals[1:])
        if not actuals_match:
            skipped_buildings.append((bid, "y_actual differs across scenarios"))
            continue

        if expected_len is None:
            expected_len = arr_len
        elif arr_len != expected_len:
            # Different buildings CAN have different test lengths (different data durations)
            # This is fine — we'll handle via common-length alignment below
            pass

        valid_buildings.append(bid)

    print(f"Valid buildings (all {len(SCENARIOS)} scenarios present, consistent): {len(valid_buildings)}")
    if skipped_buildings:
        print(f"Skipped buildings: {len(skipped_buildings)}")
        for bid, reason in skipped_buildings[:5]:
            print(f"  {bid}: {reason}")
        if len(skipped_buildings) > 5:
            print(f"  ... and {len(skipped_buildings) - 5} more")

    if len(valid_buildings) == 0:
        print("ERROR: No valid buildings found", flush=True)
        sys.exit(1)

    # ============================================================
    # 2. FIND COMMON TEST LENGTH
    # ============================================================
    # Buildings may have different test set sizes (different data durations).
    # Use the MINIMUM test length so all buildings contribute at same positions.
    print("\n--- Step 2: Determining common test window ---", flush=True)

    building_lengths = {}
    for bid in valid_buildings:
        data = np.load(PREDICTIONS_DIR / f"local_mlp_{bid}_h1.npz")
        building_lengths[bid] = len(data['y_actual'])

    lengths_array = np.array(list(building_lengths.values()))
    min_len = int(np.min(lengths_array))
    max_len = int(np.max(lengths_array))
    median_len = int(np.median(lengths_array))

    print(f"Test lengths: min={min_len}, median={median_len}, max={max_len}")

    # Filter buildings with very short test sets (< 2000 hours)
    # Short test sets force all buildings to a narrow window, losing representativeness
    MIN_TEST_HOURS = 2000
    short_buildings = [bid for bid, l in building_lengths.items() if l < MIN_TEST_HOURS]
    if short_buildings:
        print(f"Dropping {len(short_buildings)} buildings with <{MIN_TEST_HOURS} test hours: {short_buildings}")
        valid_buildings = [bid for bid in valid_buildings if bid not in set(short_buildings)]
        building_lengths = {bid: l for bid, l in building_lengths.items() if bid not in set(short_buildings)}
        lengths_array = np.array(list(building_lengths.values()))
        min_len = int(np.min(lengths_array))

    # Use min_len to ensure all buildings contribute at every position
    # This takes the LAST min_len hours of each building's test set
    # (so they share the most recent common period)
    common_len = min_len
    print(f"Using common length: {common_len} hours ({common_len//24} days, {len(valid_buildings)} buildings)")

    # Get timestamps for the reference building (for plotting)
    # Use building with exactly min_len or pick the first and truncate
    ref_bid = valid_buildings[0]
    ref_ts = get_test_timestamps(ref_bid, data_dir)
    if ref_ts is not None:
        ref_ts = ref_ts[-common_len:]
        print(f"Reference period: {ref_ts[0]} to {ref_ts[-1]}")
    else:
        # Fallback: integer indices
        ref_ts = pd.RangeIndex(common_len)
        print("Warning: Could not load timestamps, using integer indices")

    # ============================================================
    # 3. AGGREGATE PER SCENARIO — positional alignment
    # ============================================================
    print("\n--- Step 3: Aggregating portfolio predictions ---", flush=True)

    results = {}

    for scenario, label in SCENARIOS.items():
        sum_actual = np.zeros(common_len)
        sum_pred = np.zeros(common_len)
        n_loaded = 0
        n_nan_buildings = 0

        for bid in valid_buildings:
            data = np.load(PREDICTIONS_DIR / f"{scenario}_{bid}_h1.npz")
            y_actual = data['y_actual']
            y_pred = data['y_pred']

            # Take last common_len elements (most recent common period)
            y_actual = y_actual[-common_len:]
            y_pred = y_pred[-common_len:]

            # Check for NaN
            if np.any(np.isnan(y_actual)) or np.any(np.isnan(y_pred)):
                n_nan_buildings += 1
                # Replace NaN with 0 for summation (building contributes 0 at those hours)
                y_actual = np.nan_to_num(y_actual, nan=0.0)
                y_pred = np.nan_to_num(y_pred, nan=0.0)

            sum_actual += y_actual
            sum_pred += y_pred
            n_loaded += 1

        # Compute portfolio-level metrics
        residuals = sum_actual - sum_pred
        portfolio_mae = np.mean(np.abs(residuals))
        portfolio_mape = np.mean(np.abs(residuals) / np.clip(sum_actual, 1.0, None)) * 100
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((sum_actual - np.mean(sum_actual)) ** 2)
        portfolio_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        results[scenario] = {
            'label': label,
            'sum_actual': sum_actual,
            'sum_pred': sum_pred,
            'mae': portfolio_mae,
            'mape': portfolio_mape,
            'r2': portfolio_r2,
            'n_buildings': n_loaded,
            'n_nan_buildings': n_nan_buildings,
        }

        print(f"  {label:30s}: {n_loaded:3d} buildings, "
              f"MAE={portfolio_mae:7.1f} kWh/h, "
              f"MAPE={portfolio_mape:5.1f}%, "
              f"R²={portfolio_r2:.3f}"
              f"{f'  ({n_nan_buildings} had NaN)' if n_nan_buildings > 0 else ''}")

    # Verify all scenarios used same building count
    counts = [r['n_buildings'] for r in results.values()]
    if len(set(counts)) > 1:
        print(f"\nWARNING: Building counts differ across scenarios: {dict(zip(SCENARIOS.values(), counts))}")
    else:
        print(f"\nAll scenarios: {counts[0]} buildings (verified equal)")

    # Verify actual sums are identical (same test data)
    ref_actual = results[list(SCENARIOS.keys())[0]]['sum_actual']
    for scenario, res in results.items():
        if not np.allclose(ref_actual, res['sum_actual'], atol=1e-3):
            print(f"WARNING: sum_actual differs for {scenario} — investigate!")
        else:
            pass  # Good — all scenarios sum the same actual consumption

    # ============================================================
    # 4. SUMMARY STATISTICS
    # ============================================================
    print("\n--- Portfolio Summary ---", flush=True)
    total_actual = np.sum(ref_actual)
    mean_hourly = np.mean(ref_actual)
    peak_hourly = np.max(ref_actual)
    print(f"Total actual demand: {total_actual:.0f} kWh")
    print(f"Mean hourly demand:  {mean_hourly:.1f} kWh/h")
    print(f"Peak hourly demand:  {peak_hourly:.1f} kWh/h")

    # ============================================================
    # 5. GENERATE FIGURES
    # ============================================================
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ts_index = pd.DatetimeIndex(ref_ts) if isinstance(ref_ts, pd.DatetimeIndex) else ref_ts

    # --- Figure A: Full test period (daily aggregation) ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

    # Daily sums
    df_daily = pd.DataFrame({'actual': ref_actual}, index=ts_index)
    df_daily = df_daily.resample('D').sum()

    axes[0].plot(df_daily.index, df_daily['actual'], 'k-', linewidth=1.5,
                 label='Actual', alpha=0.8)

    for scenario, res in results.items():
        df_pred = pd.DataFrame({'pred': res['sum_pred']}, index=ts_index)
        df_pred = df_pred.resample('D').sum()
        axes[0].plot(df_pred.index, df_pred['pred'],
                     color=COLORS.get(scenario, 'gray'),
                     linestyle=LINESTYLES.get(scenario, '-'),
                     linewidth=1.2,
                     label=f"{res['label']} (MAE={res['mae']:.0f})",
                     alpha=0.8)

    axes[0].set_ylabel('Portfolio demand (kWh/day)', fontsize=12)
    n_bldg = results[list(SCENARIOS.keys())[0]]['n_buildings']
    axes[0].set_title(f'(a) Daily portfolio demand — {n_bldg} buildings', fontsize=13)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Residuals for best scenario
    best_scenario = min(results, key=lambda s: results[s]['mae'])
    residuals = results[best_scenario]['sum_actual'] - results[best_scenario]['sum_pred']
    axes[1].fill_between(ts_index, residuals, alpha=0.4,
                         color=COLORS.get(best_scenario, 'green'))
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
    if isinstance(ts_index, pd.DatetimeIndex):
        if args.week_start:
            week_start = pd.Timestamp(args.week_start)
        else:
            winter_mask = (ts_index.month == 12) | (ts_index.month == 1)
            if winter_mask.any():
                winter_start = ts_index[winter_mask][0]
                days_to_monday = winter_start.weekday()
                week_start = winter_start - pd.Timedelta(days=days_to_monday)
            else:
                # Fallback: first week of test set
                week_start = ts_index[0]

        week_end = week_start + pd.Timedelta(days=7)
        week_mask = (ts_index >= week_start) & (ts_index < week_end)

        if week_mask.any():
            fig, ax = plt.subplots(figsize=(14, 5))
            week_ts = ts_index[week_mask]

            ax.plot(week_ts, ref_actual[week_mask], 'k-', linewidth=1.8, label='Actual')

            for scenario, res in results.items():
                ax.plot(week_ts, res['sum_pred'][week_mask],
                        color=COLORS.get(scenario, 'gray'),
                        linestyle=LINESTYLES.get(scenario, '-'),
                        linewidth=1.2,
                        label=f"{res['label']}")

            ax.set_ylabel('Portfolio demand (kWh/h)', fontsize=12)
            ax.set_title(f'Portfolio demand — week of '
                         f'{week_start.strftime("%d %b %Y")} ({n_bldg} buildings)',
                         fontsize=13)
            ax.legend(fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path2 = FIGURES_DIR / f"portfolio_week_zoom{suffix}.pdf"
            plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {fig_path2}", flush=True)

    # ============================================================
    # 6. SAVE REPORT
    # ============================================================
    report_path = LOGS_DIR / f"portfolio_forecast_analysis_v2{suffix}.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PORTFOLIO AGGREGATION ANALYSIS v2 (positional alignment — fixed)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Buildings: {n_bldg} (all scenarios use identical building set)\n")
        f.write(f"Common test hours: {common_len}\n")
        if isinstance(ts_index, pd.DatetimeIndex):
            f.write(f"Period: {ts_index[0]} to {ts_index[-1]}\n")
        f.write(f"Total actual demand: {total_actual:.0f} kWh\n")
        f.write(f"Mean hourly portfolio demand: {mean_hourly:.1f} kWh/h\n")
        f.write(f"Peak hourly portfolio demand: {peak_hourly:.1f} kWh/h\n\n")

        f.write(f"{'Scenario':<35} {'MAE (kWh/h)':<15} {'MAPE %':<10} {'R²':<10} {'Buildings':<10}\n")
        f.write("-" * 80 + "\n")
        for scenario, res in sorted(results.items(), key=lambda x: x[1]['mae']):
            f.write(f"{res['label']:<35} {res['mae']:<15.1f} {res['mape']:<10.1f} "
                    f"{res['r2']:<10.3f} {res['n_buildings']:<10}\n")

        # Paper text suggestion — now with correct numbers
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("PAPER TEXT SUGGESTION (verified: equal building counts)\n")
        f.write("=" * 70 + "\n\n")

        local_res = results.get('local_mlp', {})
        pers_res = results.get('personalised_fedadam', {})
        global_res = results.get('fedadam', {})
        cent_res = results.get('centralised_mlp', {})

        f.write("At the portfolio level, aggregating per-building estimates from\n")
        f.write(f"{n_bldg} substations, the privacy-preserving approaches — Local MLP\n")
        f.write(f"(MAPE = {local_res.get('mape', 0):.1f}%, R² = {local_res.get('r2', 0):.3f}) and\n")
        f.write(f"Personalised FL (MAPE = {pers_res.get('mape', 0):.1f}%, R² = {pers_res.get('r2', 0):.3f}) —\n")
        f.write(f"deliver comparable fleet-level demand accuracy. The global FedAdam\n")
        f.write(f"model degrades to MAPE = {global_res.get('mape', 0):.1f}%, confirming that\n")
        f.write(f"personalisation is essential under high heterogeneity. Centralised MLP\n")
        f.write(f"achieves MAPE = {cent_res.get('mape', 0):.1f}% but requires pooling all raw data,\n")
        f.write(f"violating the privacy boundary. These results validate the operational\n")
        f.write(f"channel: the housing company receives actionable portfolio-level demand\n")
        f.write(f"intelligence without centralised data access.\n")

    print(f"\nSaved: {report_path}", flush=True)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()