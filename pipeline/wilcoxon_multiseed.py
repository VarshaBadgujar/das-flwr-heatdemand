#!/usr/bin/env python3
"""
wilcoxon_multiseed.py — Per-seed Wilcoxon signed-rank tests for
camera-ready paper.

Replicates the Table 4 (tab:wilcoxon) tests from the SCAI 2026 paper
across all 5 seeds (42, 43, 44, 45, 46) and reports consistency.

Recipe matches pipeline/wilcoxon_test.py:
  - horizon=1
  - status=success
  - drop 4 outlier buildings (B001, B037, B238, B028)
  - paired Wilcoxon signed-rank test on per-building MAE
"""

import sys
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.anonymise_buildings import real_ids

LOG_DIR = Path("logs")
OUTLIERS = set(real_ids("B001", "B037", "B238", "B028"))

SCENARIOS = {
    "Local MLP":      "local_mlp_matched_results_250_v5",
    "Pers FedAdam":   "fl_personalised_final_fedadam_v5",
    "Pers FedProx":   "fl_personalised_final_fedprox_v5",
    "Centr MLP":      "centralised_mlp_results_v5",
}

# Map seed -> filename suffix for each scenario family
def filename(scenario_stub: str, seed: int) -> Path:
    if seed == 42:
        # The seed=42 file naming varies across scenarios — pick the right one
        if scenario_stub == "centralised_mlp_results_v5":
            return LOG_DIR / "centralised_mlp_results_v5_portfolio.csv"
        elif scenario_stub == "local_mlp_matched_results_250_v5":
            return LOG_DIR / "local_mlp_matched_results_250_v5_portfolio.csv"
        elif scenario_stub == "fl_personalised_final_fedadam_v5":
            return LOG_DIR / "fl_personalised_final_fedadam_v5_portfolio.csv"
        elif scenario_stub == "fl_personalised_final_fedprox_v5":
            return LOG_DIR / "fl_personalised_final_fedprox_v5_portfolio.csv"
    else:
        return LOG_DIR / f"{scenario_stub}_seed{seed}.csv"
    raise ValueError(f"unknown stub {scenario_stub} seed {seed}")


def load_mae_series(scenario_stub: str, seed: int) -> pd.Series:
    """Return per-building MAE indexed by building_id, after paper filter."""
    f = filename(scenario_stub, seed)
    if not f.exists():
        raise FileNotFoundError(f"{f} missing")
    df = pd.read_csv(f)
    if 'horizon' in df.columns:
        df = df[df['horizon'] == 1]
    if 'status' in df.columns:
        df = df[df['status'] == 'success']
    df = df[~df['building_id'].astype(str).isin(OUTLIERS)]
    return df.set_index(df['building_id'].astype(str))['mae']


COMPARISONS = [
    ("Pers FedAdam", "Local MLP",      "Pers FedAdam vs Local MLP"),
    ("Pers FedProx", "Local MLP",      "Pers FedProx vs Local MLP"),
    ("Pers FedAdam", "Centr MLP",      "Pers FedAdam vs Centr MLP"),
    ("Local MLP",    "Centr MLP",      "Local MLP vs Centr MLP"),
    ("Pers FedAdam", "Pers FedProx",   "Pers FedAdam vs Pers FedProx"),
]


def run_comparison(a_stub: str, b_stub: str, seed: int):
    """Wilcoxon test: is A's per-building MAE different from B's?
    Returns (delta_pct, p_value, n_improved, n_total).
    delta_pct: (mean_A - mean_B) / mean_B * 100 — negative means A is better."""
    a = load_mae_series(a_stub, seed)
    b = load_mae_series(b_stub, seed)
    # Inner join on building IDs both have
    both = a.to_frame('a').join(b.to_frame('b'), how='inner').dropna()
    if len(both) < 10:
        return None, None, 0, 0
    diff = both['a'] - both['b']
    # Wilcoxon signed-rank test
    try:
        stat, p = wilcoxon(diff, alternative='two-sided', zero_method='wilcox')
    except ValueError as e:
        # All zeros, or similar edge case
        p = 1.0
    delta_pct = (both['a'].mean() - both['b'].mean()) / both['b'].mean() * 100
    n_improved = int((diff < 0).sum())  # buildings where A is lower (better)
    return delta_pct, p, n_improved, len(both)


def main():
    print(f"\n{'='*78}")
    print(f"Per-seed Wilcoxon signed-rank tests (h=1, 246 buildings)")
    print(f"{'='*78}\n")

    SEEDS = [42, 43, 44, 45, 46]
    a_stubs = {
        "Local MLP":     "local_mlp_matched_results_250_v5",
        "Pers FedAdam":  "fl_personalised_final_fedadam_v5",
        "Pers FedProx":  "fl_personalised_final_fedprox_v5",
        "Centr MLP":     "centralised_mlp_results_v5",
    }

    for a_name, b_name, label in COMPARISONS:
        print(f"\n--- {label} ---")
        print(f"{'Seed':<8} {'ΔMAE %':>10} {'p-value':>12} {'Sig':>6} {'Improved':>12}")
        print("-" * 55)
        p_values = []
        delta_pcts = []
        for s in SEEDS:
            try:
                d, p, n_imp, n_tot = run_comparison(
                    a_stubs[a_name], a_stubs[b_name], s)
                if d is None:
                    print(f"{s:<8} {'SKIP':>10}")
                    continue
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = "n.s."
                print(f"{s:<8} {d:>+10.2f} {p:>12.4g} {sig:>6} {n_imp:>5}/{n_tot}")
                p_values.append(p)
                delta_pcts.append(d)
            except FileNotFoundError as e:
                print(f"{s:<8} MISSING: {e}")

        if p_values:
            n_sig_01 = sum(p < 0.001 for p in p_values)
            n_ns = sum(p > 0.05 for p in p_values)
            print(f"{'-'*55}")
            print(f"  ΔMAE range:   [{min(delta_pcts):+.2f}%, {max(delta_pcts):+.2f}%]")
            print(f"  median p:     {sorted(p_values)[len(p_values)//2]:.4g}")
            print(f"  p<0.001 in:   {n_sig_01}/{len(p_values)} seeds")
            print(f"  n.s. in:      {n_ns}/{len(p_values)} seeds")

    print(f"\n{'='*78}\n")


if __name__ == "__main__":
    main()