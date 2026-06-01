"""
Selective Personalisation Analysis

Purpose:
  1. Characterise the 32 buildings where personalisation improved vs 68 where it worsened
  2. Test hypothesis: high-consumption / high-variability buildings benefit more
  3. Find a threshold for selective personalisation
  4. Connect to peak identification for housing operator

Run from project root:
    cd ~/Desktop/varsha_projects/das-flwr-heatdemand
    python pipeline/selective_personalisation_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ── File paths ─
LOCAL_MLP_CSV    = "logs/local_mlp_matched_results_matched100.csv"
PERSONALISED_CSV = "logs/fl_personalised_final.csv"
DATA_DIR         = "data/processed"          # 988 parquet files, one per building
FL_IDS_FILE      = "logs/fl_building_ids.txt"  # 100 building IDs used in FL

# ── Step 1: Load and merge MAE results ────────
print("=" * 65)
print("STEP 1: Load MAE results")
print("=" * 65)

local_mlp    = pd.read_csv(LOCAL_MLP_CSV)
personalised = pd.read_csv(PERSONALISED_CSV)

local_h1 = (
    local_mlp[local_mlp['horizon'] == 1][['building_id', 'mae']]
    .rename(columns={'mae': 'mae_local'})
)
pers_h1 = (
    personalised[
        (personalised['scenario'] == 'personalised') &
        (personalised['horizon'] == 1)
    ][['building_id', 'mae']]
    .rename(columns={'mae': 'mae_personalised'})
)

merged = local_h1.merge(pers_h1, on='building_id')
merged['diff']       = merged['mae_local'] - merged['mae_personalised']
merged['improved']   = merged['diff'] > 0
merged['pct_change'] = (merged['diff'] / merged['mae_local']) * 100

print(f"  Matched buildings: {len(merged)}")
print(f"  Improved:  {merged['improved'].sum()} buildings")
print(f"  Worsened:  {(~merged['improved']).sum()} buildings")

# ── Step 2: Load consumption statistics for each building ─────
print("\n" + "=" * 65)
print("STEP 2: Load consumption statistics from processed data")
print("=" * 65)

building_stats = []

for bid in merged['building_id'].values:
    # Try common filename patterns
    candidates = [
        os.path.join(DATA_DIR, f"{bid}.parquet"),
        os.path.join(DATA_DIR, f"building_{bid}.parquet"),
        os.path.join(DATA_DIR, f"{bid}_features.parquet"),
    ]
    filepath = None
    for c in candidates:
        if os.path.exists(c):
            filepath = c
            break

    if filepath is None:
        # Try listing directory for partial match
        try:
            files = os.listdir(DATA_DIR)
            match = [f for f in files if str(bid) in f and f.endswith('.parquet')]
            if match:
                filepath = os.path.join(DATA_DIR, match[0])
        except:
            pass

    if filepath is None:
        print(f"  WARNING: No data file found for building {bid}")
        building_stats.append({
            'building_id': bid,
            'mean_kwh': np.nan,
            'std_kwh': np.nan,
            'cov': np.nan,
            'median_kwh': np.nan,
            'p90_kwh': np.nan,
            'p95_kwh': np.nan,
            'max_kwh': np.nan,
            'n_hours': np.nan
        })
        continue

    try:
        df = pd.read_parquet(filepath)

        # Find kwh column
        kwh_col = None
        for col in ['kwh', 'energy_kwh', 'heat_kwh', 'kWh']:
            if col in df.columns:
                kwh_col = col
                break
        if kwh_col is None:
            kwh_col = df.columns[0]

        kwh = df[kwh_col].dropna()
        kwh = kwh[kwh >= 0]  # remove negatives

        building_stats.append({
            'building_id': bid,
            'mean_kwh':   kwh.mean(),
            'std_kwh':    kwh.std(),
            'cov':        kwh.std() / kwh.mean() if kwh.mean() > 0 else np.nan,
            'median_kwh': kwh.median(),
            'p90_kwh':    kwh.quantile(0.90),
            'p95_kwh':    kwh.quantile(0.95),
            'max_kwh':    kwh.max(),
            'n_hours':    len(kwh)
        })
    except Exception as e:
        print(f"  ERROR loading {bid}: {e}")
        building_stats.append({
            'building_id': bid,
            'mean_kwh': np.nan, 'std_kwh': np.nan, 'cov': np.nan,
            'median_kwh': np.nan, 'p90_kwh': np.nan,
            'p95_kwh': np.nan, 'max_kwh': np.nan, 'n_hours': np.nan
        })

stats_df = pd.DataFrame(building_stats)
print(f"  Loaded stats for {stats_df['mean_kwh'].notna().sum()}/{len(stats_df)} buildings")

# ── Step 3: Merge stats with improvement labels 
print("\n" + "=" * 65)
print("STEP 3: Compare improved vs worsened buildings")
print("=" * 65)

analysis = merged.merge(stats_df, on='building_id')
improved  = analysis[analysis['improved'] == True]
worsened  = analysis[analysis['improved'] == False]

features = ['mean_kwh', 'std_kwh', 'cov', 'p90_kwh', 'p95_kwh', 'max_kwh']

print(f"\n{'Feature':<15} {'Improved (n={})'.format(len(improved)):<25} "
      f"{'Worsened (n={})'.format(len(worsened)):<25} {'Mann-Whitney p':<15} {'Conclusion'}")
print("-" * 95)

significant_features = []

for feat in features:
    imp_vals = improved[feat].dropna()
    wor_vals = worsened[feat].dropna()
    if len(imp_vals) < 3 or len(wor_vals) < 3:
        continue

    stat, p = stats.mannwhitneyu(imp_vals, wor_vals, alternative='two-sided')

    imp_med = imp_vals.median()
    wor_med = wor_vals.median()
    direction = "improved > worsened" if imp_med > wor_med else "improved < worsened"
    sig = "✅ SIGNIFICANT" if p < 0.05 else "— not significant"

    print(f"{feat:<15} median={imp_med:<10.3f}          median={wor_med:<10.3f}          "
          f"p={p:<10.4f}     {sig} ({direction})")

    if p < 0.05:
        significant_features.append((feat, imp_med, wor_med, p, direction))

# ── Step 4: Threshold analysis 
print("\n" + "=" * 65)
print("STEP 4: Threshold analysis — when does personalisation help?")
print("=" * 65)

if significant_features:
    # Use the most significant feature
    best_feat = sorted(significant_features, key=lambda x: x[3])[0]
    feat_name = best_feat[0]

    print(f"\nBest discriminating feature: {feat_name}")

    # Try different thresholds
    thresholds = np.percentile(
        analysis[feat_name].dropna(),
        [25, 33, 40, 50, 60, 67, 75]
    )

    print(f"\n{'Threshold':<15} {'Above: % improved':<25} "
          f"{'Below: % improved':<25} {'Accuracy'}")
    print("-" * 75)

    best_acc = 0
    best_thresh = None

    for thresh in thresholds:
        above = analysis[analysis[feat_name] >= thresh]
        below = analysis[analysis[feat_name] < thresh]

        if len(above) == 0 or len(below) == 0:
            continue

        pct_above_improved = above['improved'].mean() * 100
        pct_below_improved = below['improved'].mean() * 100

        # Accuracy: predict "improved" for above, "worsened" for below
        tp = above['improved'].sum()
        tn = (~below['improved']).sum()
        acc = (tp + tn) / len(analysis) * 100

        marker = " ← BEST" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

        print(f"{thresh:<15.3f} {pct_above_improved:<25.1f}% "
              f"{pct_below_improved:<25.1f}% {acc:.1f}%{marker}")

    print(f"\nBest threshold: {feat_name} >= {best_thresh:.3f}")
    print(f"Selective personalisation accuracy: {best_acc:.1f}%")

else:
    print("No features significantly distinguish improved vs worsened buildings.")
    print("Hypothesis may not hold — personalisation benefit is random across buildings.")
    feat_name  = 'mean_kwh'
    best_thresh = analysis['mean_kwh'].median()

# ── Step 5: Peak identification 
print("\n" + "=" * 65)
print("STEP 5: Are improved buildings the peak contributors?")
print("=" * 65)

# Top 20% consumers = peak buildings
peak_threshold = analysis['mean_kwh'].quantile(0.80)
analysis['is_peak'] = analysis['mean_kwh'] >= peak_threshold

peak_buildings    = analysis[analysis['is_peak']]
nonpeak_buildings = analysis[~analysis['is_peak']]

print(f"\nPeak threshold (top 20%): mean_kwh >= {peak_threshold:.3f}")
print(f"Peak buildings:     {len(peak_buildings)}")
print(f"Non-peak buildings: {len(nonpeak_buildings)}")

pct_peak_improved    = peak_buildings['improved'].mean() * 100
pct_nonpeak_improved = nonpeak_buildings['improved'].mean() * 100

print(f"\nOf peak buildings:     {pct_peak_improved:.1f}% improved under personalisation")
print(f"Of non-peak buildings: {pct_nonpeak_improved:.1f}% improved under personalisation")

if pct_peak_improved > pct_nonpeak_improved + 10:
    print("\n✅ HYPOTHESIS SUPPORTED: Peak buildings benefit more from personalisation")
    print("   → Selective personalisation = target the peak contributors")
    print("   → Housing operator can use personalised models for peak shaving")
elif pct_peak_improved > pct_nonpeak_improved:
    print("\n⚠️  WEAK SUPPORT: Peak buildings improve slightly more — trend but not strong")
else:
    print("\n❌ HYPOTHESIS NOT SUPPORTED: Peak buildings do NOT benefit more")
    print("   → Need alternative characterisation of which buildings benefit")

# ── Step 6: Summary statistics 
print("\n" + "=" * 65)
print("STEP 6: Summary — copy these numbers into paper discussion")
print("=" * 65)

print(f"""
Key numbers:
  Buildings improved:                {merged['improved'].sum()}/100 ({merged['improved'].mean()*100:.0f}%)
  Buildings worsened:                {(~merged['improved']).sum()}/100
  Mean MAE improvement (improved):   {improved['diff'].mean():.4f} kWh/h
  Mean MAE deterioration (worsened): {worsened['diff'].mean():.4f} kWh/h
  Peak buildings improved:           {pct_peak_improved:.1f}%
  Non-peak buildings improved:       {pct_nonpeak_improved:.1f}%
""")

# ── Save full analysis ─────────
analysis.to_csv("logs/selective_personalisation_analysis.csv", index=False)
print("Full analysis saved → logs/selective_personalisation_analysis.csv")