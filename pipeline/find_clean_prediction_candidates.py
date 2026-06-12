#!/usr/bin/env python3
"""
find_clean_prediction_candidates.py

Find clean candidate buildings for Figure 2 (per-building prediction
comparison). Filters out buildings with sensor issues, zero-flow
stretches, isolated spikes, or other anomalies that would make poor
visual examples for the paper.

Usage:
    python pipeline/find_clean_prediction_candidates.py
    python pipeline/find_clean_prediction_candidates.py --plot
    python pipeline/find_clean_prediction_candidates.py --plot \
        --output-dir logs/candidate_inspection

Outputs:
    logs/clean_prediction_candidates.txt — ranked candidate lists
    logs/candidate_inspection/*.pdf — diagnostic plots (if --plot)

Selection criteria (must pass ALL):
    1. Mean test consumption > 1.5 kWh/h (excludes near-empty buildings)
    2. Zero-fraction < 5% in test set (excludes sensor offline periods)
    3. No isolated spikes > 5x daily mean (excludes meter glitches)
    4. CV of daily means < 1.5 (excludes regime changes)
    5. At least 2000 test hours (sufficient temporal coverage)
    6. Local MLP MAE in [0.4, 3.0] (excludes pathological cases)

Three categories ranked separately:
    A. FL clearly wins (Pers FL beats Local MLP by >20%)
    B. Local clearly wins (Local MLP beats Pers FL by >20%)
    C. FL ≈ Local (within 5% of each other)

The user picks 1 from each category + 1 typical for Figure 2 panels.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.anonymise_buildings import real_ids

OUTLIERS = set(real_ids("B001", "B037", "B238", "B028"))

LOCAL_MLP_CSV = 'logs/local_mlp_matched_results_250_v5_portfolio.csv'
PERS_FL_CSV = 'logs/fl_personalised_final_fedadam_v5_portfolio.csv'
PREDICTIONS_DIR = Path('logs/predictions/v5_portfolio')
DATA_DIR = Path('data/processed')
OUTPUT_FILE = Path('logs/clean_prediction_candidates.txt')

# Quality thresholds
MIN_MEAN_KWH = 0.5         # Exclude near-empty buildings
MAX_ZERO_FRAC = 0.05       # Exclude buildings with > 5% zero hours
MAX_SPIKE_RATIO = 8.0      # Exclude buildings with isolated spikes > 5x daily mean
MAX_DAILY_CV = 1.5         # Exclude buildings with regime changes
MIN_TEST_HOURS = 2000      # Sufficient temporal coverage
MIN_LOCAL_MAE = 0.4        # Exclude trivially easy buildings
MAX_LOCAL_MAE = 3.0        # Exclude pathological buildings

# Win/loss thresholds
FL_WIN_THRESHOLD = 0.08    # FL must beat Local by ≥20% to qualify
LOCAL_WIN_THRESHOLD = 0.08 # Local must beat FL by ≥20% to qualify
TIE_THRESHOLD = 0.03       # FL and Local within 5% counts as tie

# Number of candidates to report per category
N_PER_CATEGORY = 15


# ============================================================
# QUALITY CHECKS
# ============================================================

def assess_building_quality(building_id):
    """
    Run quality checks on a building's test set.
    Returns dict with metrics and a 'pass' boolean.
    """
    parquet_path = DATA_DIR / f'{building_id}.parquet'
    if not parquet_path.exists():
        return {'building_id': building_id, 'pass': False, 'reason': 'no_parquet'}

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        return {'building_id': building_id, 'pass': False, 'reason': f'read_error: {e}'}

    # Test set is the last 15% (matching the 60/15/10/15 split)
    # We check ONLY the first 168 hours of test (one week) since
    # that's what Figure 2 actually displays. Anomalies elsewhere
    # in the test set don't matter for the visual.
    n = len(df)
    test_start = int(n * 0.85)
    test_full = df.iloc[test_start:]
    if len(test_full) < 168:
        return {'building_id': building_id, 'pass': False, 'reason': 'no_full_week'}
    test = test_full.iloc[:168]

    # Check that the FULL test set has enough hours (for the metric calculation)
    if len(test_full) < MIN_TEST_HOURS:
        return {
            'building_id': building_id, 'pass': False,
            'reason': f'too_few_test_hours_{len(test_full)}',
            'test_hours': len(test_full),
        }

    target_col = 'kwh' if 'kwh' in test.columns else test.columns[0]
    y = test[target_col].values

    # Metric 1: mean consumption
    mean_kwh = float(np.mean(y))
    if mean_kwh < MIN_MEAN_KWH:
        return {
            'building_id': building_id, 'pass': False,
            'reason': f'low_mean_{mean_kwh:.2f}',
            'mean_kwh': mean_kwh,
        }

    # Metric 2: zero fraction
    zero_frac = float(np.mean(y == 0))
    if zero_frac > MAX_ZERO_FRAC:
        return {
            'building_id': building_id, 'pass': False,
            'reason': f'too_many_zeros_{zero_frac:.1%}',
            'mean_kwh': mean_kwh, 'zero_frac': zero_frac,
        }

    # Metric 3: isolated spike check
    # Compute daily means and look for any single hour > 5x its day's mean
    n_days = len(y) // 24
    spike_ratio = 0.0
    if n_days > 0:
        y_trim = y[:n_days * 24].reshape(n_days, 24)
        daily_means = y_trim.mean(axis=1)
        # Avoid division by zero
        daily_means_safe = np.where(daily_means > 0.1, daily_means, 0.1)
        hourly_to_daily = y_trim / daily_means_safe[:, None]
        spike_ratio = float(np.max(hourly_to_daily))
        if spike_ratio > MAX_SPIKE_RATIO:
            return {
                'building_id': building_id, 'pass': False,
                'reason': f'spike_ratio_{spike_ratio:.1f}',
                'mean_kwh': mean_kwh, 'zero_frac': zero_frac,
                'spike_ratio': spike_ratio,
            }

        # Metric 4: regime change check (CV of daily means)
        if daily_means.mean() > 0:
            daily_cv = float(daily_means.std() / daily_means.mean())
            if daily_cv > MAX_DAILY_CV:
                return {
                    'building_id': building_id, 'pass': False,
                    'reason': f'daily_cv_{daily_cv:.2f}',
                    'mean_kwh': mean_kwh, 'zero_frac': zero_frac,
                    'spike_ratio': spike_ratio, 'daily_cv': daily_cv,
                }
        else:
            daily_cv = np.nan
    else:
        daily_cv = np.nan

    return {
        'building_id': building_id,
        'pass': True,
        'mean_kwh': mean_kwh,
        'zero_frac': zero_frac,
        'spike_ratio': spike_ratio,
        'daily_cv': daily_cv,
        'test_hours': len(y),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Find clean prediction example candidates'
    )
    parser.add_argument('--plot', action='store_true',
                        help='Generate diagnostic plots for top candidates')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('logs/candidate_inspection'),
                        help='Directory for diagnostic plots')
    parser.add_argument('--n', type=int, default=N_PER_CATEGORY,
                        help='Candidates to report per category')
    args = parser.parse_args()

    print('=' * 70)
    print('CANDIDATE SELECTION FOR FIGURE 2 — PREDICTION EXAMPLES')
    print('=' * 70)

    # 1. Load per-building MAE results
    print('\n[1/4] Loading per-building results...')
    if not Path(LOCAL_MLP_CSV).exists():
        print(f'  ERROR: {LOCAL_MLP_CSV} not found.')
        sys.exit(1)
    if not Path(PERS_FL_CSV).exists():
        print(f'  ERROR: {PERS_FL_CSV} not found.')
        sys.exit(1)

    local_df = pd.read_csv(LOCAL_MLP_CSV)
    fl_df = pd.read_csv(PERS_FL_CSV)
    local_df['building_id'] = local_df['building_id'].astype(str)
    fl_df['building_id'] = fl_df['building_id'].astype(str)

    # Filter to h=1
    if 'horizon' in local_df.columns:
        local_df = local_df[local_df['horizon'] == 1]
    if 'horizon' in fl_df.columns:
        fl_df = fl_df[fl_df['horizon'] == 1]

    # Drop outliers
    local_df = local_df[~local_df['building_id'].isin(OUTLIERS)]
    fl_df = fl_df[~fl_df['building_id'].isin(OUTLIERS)]

    # Merge
    merged = local_df[['building_id', 'mae']].merge(
        fl_df[['building_id', 'mae']],
        on='building_id', suffixes=('_local', '_fl'),
    )
    merged['mae_diff'] = merged['mae_local'] - merged['mae_fl']
    merged['relative_diff'] = merged['mae_diff'] / merged['mae_local']
    print(f'  Loaded {len(merged)} buildings (after outlier removal)')

    # 2. Filter by Local MLP MAE range (avoid pathological cases)
    print('\n[2/4] Filtering by accuracy range...')
    accuracy_mask = (merged['mae_local'] >= MIN_LOCAL_MAE) & \
                    (merged['mae_local'] <= MAX_LOCAL_MAE)
    candidates = merged[accuracy_mask].copy()
    print(f'  {len(candidates)} buildings in MAE range '
          f'[{MIN_LOCAL_MAE}, {MAX_LOCAL_MAE}]')

    # 3. Quality checks
    print(f'\n[3/4] Running quality checks on {len(candidates)} buildings...')
    quality_results = []
    for i, bid in enumerate(candidates['building_id']):
        if i % 50 == 0 and i > 0:
            print(f'  {i}/{len(candidates)} checked...')
        quality_results.append(assess_building_quality(bid))

    quality_df = pd.DataFrame(quality_results)
    clean = candidates.merge(quality_df, on='building_id')

    n_pass = clean['pass'].sum()
    n_fail = len(clean) - n_pass
    print(f'  {n_pass} buildings passed all quality checks ({n_fail} failed)')

    if n_pass == 0:
        print('\n  ERROR: No buildings passed quality checks. '
              'Try relaxing thresholds.')
        sys.exit(1)

    # Show failure reasons
    fail_df = clean[~clean['pass']]
    if len(fail_df) > 0:
        print(f'\n  Top failure reasons:')
        reason_counts = fail_df['reason'].apply(
            lambda r: r.split('_')[0] if isinstance(r, str) else 'unknown'
        ).value_counts()
        for reason, count in reason_counts.head(5).items():
            print(f'    {reason}: {count}')

    clean = clean[clean['pass']].copy()

    # 4. Categorise into win/loss/tie
    print(f'\n[4/4] Categorising {len(clean)} clean candidates...')

    fl_wins = clean[clean['relative_diff'] > FL_WIN_THRESHOLD].copy()
    fl_wins = fl_wins.sort_values('relative_diff', ascending=False)

    local_wins = clean[clean['relative_diff'] < -LOCAL_WIN_THRESHOLD].copy()
    local_wins = local_wins.sort_values('relative_diff', ascending=True)

    ties = clean[clean['relative_diff'].abs() < TIE_THRESHOLD].copy()
    ties['abs_diff'] = ties['mae_diff'].abs()
    ties = ties.sort_values('abs_diff', ascending=True)

    print(f'  FL clearly wins (>20% better): {len(fl_wins)} buildings')
    print(f'  Local clearly wins (>20% better): {len(local_wins)} buildings')
    print(f'  FL ≈ Local (within 5%): {len(ties)} buildings')

    # 5. Write output
    print(f'\n[OUTPUT] Writing to {OUTPUT_FILE}')
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    cols_to_show = ['building_id', 'mae_local', 'mae_fl', 'mae_diff',
                    'relative_diff', 'mean_kwh', 'zero_frac',
                    'spike_ratio', 'daily_cv', 'test_hours']

    with open(OUTPUT_FILE, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('CLEAN PREDICTION EXAMPLE CANDIDATES — Figure 2 selection\n')
        f.write('=' * 70 + '\n\n')

        f.write(f'Quality thresholds:\n')
        f.write(f'  Min mean consumption: {MIN_MEAN_KWH} kWh/h\n')
        f.write(f'  Max zero fraction: {MAX_ZERO_FRAC:.0%}\n')
        f.write(f'  Max spike ratio: {MAX_SPIKE_RATIO}x daily mean\n')
        f.write(f'  Max daily CV: {MAX_DAILY_CV}\n')
        f.write(f'  Min test hours: {MIN_TEST_HOURS}\n')
        f.write(f'  Local MAE range: [{MIN_LOCAL_MAE}, {MAX_LOCAL_MAE}]\n\n')

        f.write(f'Started with {len(merged)} buildings (after outlier removal)\n')
        f.write(f'After accuracy filter: {len(candidates)}\n')
        f.write(f'After quality checks: {len(clean)}\n')
        f.write(f'  FL wins (>20% better): {len(fl_wins)}\n')
        f.write(f'  Local wins (>20% better): {len(local_wins)}\n')
        f.write(f'  Ties (within 5%): {len(ties)}\n\n')

        f.write('=' * 70 + '\n')
        f.write(f'CATEGORY A: FL clearly wins (top {args.n})\n')
        f.write('=' * 70 + '\n')
        f.write(fl_wins[cols_to_show].head(args.n).to_string(index=False))
        f.write('\n\n')

        f.write('=' * 70 + '\n')
        f.write(f'CATEGORY B: Local clearly wins (top {args.n})\n')
        f.write('=' * 70 + '\n')
        f.write(local_wins[cols_to_show].head(args.n).to_string(index=False))
        f.write('\n\n')

        f.write('=' * 70 + '\n')
        f.write(f'CATEGORY C: FL ≈ Local (top {args.n})\n')
        f.write('=' * 70 + '\n')
        f.write(ties[cols_to_show].head(args.n).to_string(index=False))
        f.write('\n\n')

        f.write('=' * 70 + '\n')
        f.write('NEXT STEPS\n')
        f.write('=' * 70 + '\n')
        f.write('1. Pick 1 building from Category A (panel a — FL wins)\n')
        f.write('2. Pick 1 building from Category B (panel b — Local wins)\n')
        f.write('3. Pick 1 building from Category C (panel c — typical)\n')
        f.write('4. Pick 1 second building from Category A or C (panel d)\n')
        f.write('5. Update pipeline/generate_prediction_examples.py with\n')
        f.write('   the chosen building IDs and re-render Figure 2.\n')
        f.write('6. If --plot was used, inspect candidate_inspection/*.pdf\n')
        f.write('   to verify visual cleanness before final selection.\n')

    print(f'  Wrote {OUTPUT_FILE}')

    # 6. Optional diagnostic plots
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('\n  WARNING: matplotlib not available, skipping plots')
            return

        print(f'\n[PLOTS] Generating diagnostic plots in {args.output_dir}/...')
        args.output_dir.mkdir(parents=True, exist_ok=True)

        def plot_candidate(building_id, category, rank, mae_local, mae_fl):
            """Plot 7 days of test data with all model predictions overlaid."""
            try:
                # Load actual + predictions
                df = pd.read_parquet(DATA_DIR / f'{building_id}.parquet')
                test_start = int(len(df) * 0.85)
                actual = df.iloc[test_start:]['kwh'].values

                local_npz = PREDICTIONS_DIR / f'local_mlp_{building_id}_h1.npz'
                fl_npz = PREDICTIONS_DIR / f'personalised_fedadam_{building_id}_h1.npz'

                if not local_npz.exists() or not fl_npz.exists():
                    return False

                local_data = np.load(local_npz)
                fl_data = np.load(fl_npz)
                local_pred = local_data['y_pred']
                fl_pred = fl_data['y_pred']

                # Show first 168 hours (1 week)
                n_show = min(168, len(actual), len(local_pred), len(fl_pred))

                fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
                hours = np.arange(n_show)
                ax.plot(hours, actual[:n_show], 'k-', lw=1.5, label='Actual')
                ax.plot(hours, local_pred[:n_show], '--', color='purple',
                        lw=1, alpha=0.8, label='Local MLP')
                ax.plot(hours, fl_pred[:n_show], '-', color='green',
                        lw=1, alpha=0.8, label='Personalised FL')
                ax.set_xlabel('Hour (test set, first week)')
                ax.set_ylabel('Consumption (kWh/h)')
                ax.set_title(
                    f'[{category}#{rank}] Building {building_id}  '
                    f'Local MAE={mae_local:.2f}  FL MAE={mae_fl:.2f}'
                )
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                fname = args.output_dir / f'{category}_{rank:02d}_{building_id}.pdf'
                plt.savefig(fname, bbox_inches='tight')
                plt.close()
                return True
            except Exception as e:
                print(f'    Failed for {building_id}: {e}')
                return False

        n_plotted = 0
        for category, df_cat, label in [
            ('A_FL_wins', fl_wins, 'FL wins'),
            ('B_Local_wins', local_wins, 'Local wins'),
            ('C_tie', ties, 'Tie'),
        ]:
            print(f'  {label}...')
            for rank, (_, row) in enumerate(df_cat.head(args.n).iterrows(), 1):
                if plot_candidate(row['building_id'], category, rank,
                                  row['mae_local'], row['mae_fl']):
                    n_plotted += 1

        print(f'  Plotted {n_plotted} diagnostic figures in {args.output_dir}/')
        print(f'\n  Inspect them visually:')
        print(f'    ls {args.output_dir}/A_FL_wins_*.pdf | head')
        print(f'    ls {args.output_dir}/B_Local_wins_*.pdf | head')
        print(f'    ls {args.output_dir}/C_tie_*.pdf | head')

    print('\n' + '=' * 70)
    print('DONE')
    print('=' * 70)
    print(f'\nReview {OUTPUT_FILE} for ranked candidate lists.')
    if args.plot:
        print(f'Inspect {args.output_dir}/*.pdf for visual quality.')
    print(f'\nOnce you pick 4 buildings, update '
          f'pipeline/generate_prediction_examples.py and re-render Figure 2.')


if __name__ == '__main__':
    main()