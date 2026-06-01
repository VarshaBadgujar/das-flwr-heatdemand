"""
Wilcoxon Signed-Rank Test: Pairwise Model Comparisons
Purpose: Statistically validate MAE differences at t+1 horizon

6 pairwise comparisons:
  1. Pers FL (FedAdam) vs Local MLP
  2. Pers FL (FedProx) vs Local MLP
  3. Pers FL (FedAdam) vs Centralised MLP
  4. Pers FL (FedProx) vs Centralised MLP
  5. Local MLP vs Centralised MLP
  6. Pers FL (FedAdam) vs Pers FL (FedProx)

Run:
    cd ~/Desktop/varsha_projects/das-flwr-heatdemand
    python pipeline/wilcoxon_test.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# ── File paths ────────────────────────────────────────────────────────────────
LOCAL_MLP_CSV   = LOG_DIR / "local_mlp_matched_results_250_v5_portfolio.csv"
PERS_ADAM_CSV   = LOG_DIR / "fl_personalised_final_fedadam_v5_portfolio.csv"
PERS_PROX_CSV   = LOG_DIR / "fl_personalised_final_fedprox_v5_portfolio.csv"
CENTRAL_MLP_CSV = LOG_DIR / "centralised_mlp_results_v5_portfolio.csv"

# ── Outlier exclusion ─────────────────────────────────────────────────────────
EXCLUDE = [B001, B037, B238, B028]


def load_h1(path, label):
    """Load CSV, exclude outliers, filter to h=1 success rows."""
    df = pd.read_csv(path)
    df = df[~df['building_id'].isin(EXCLUDE)]
    h1 = df[(df['horizon'] == 1) & (df['status'] == 'success')]
    print(f"  {label:<25} {len(df):>5} rows → {len(h1):>4} buildings (h=1)")
    return h1[['building_id', 'mae']].rename(columns={'mae': f'mae_{label}'})


def wilcoxon_pair(merged, col_a, col_b, name_a, name_b):
    """Run Wilcoxon signed-rank test on a paired comparison."""
    a = merged[col_a].values
    b = merged[col_b].values
    diff = a - b  # positive = B is better (lower MAE)

    stat, p_one = stats.wilcoxon(a, b, alternative='greater')
    _,    p_two = stats.wilcoxon(a, b, alternative='two-sided')

    n_nonzero = len(diff[diff != 0])
    r = 1 - (2 * stat) / (n_nonzero * (n_nonzero + 1) / 2) if n_nonzero > 0 else 0.0
    effect = "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small"

    if p_one < 0.001:
        sig_str = "p < 0.001"
    elif p_one < 0.01:
        sig_str = f"p = {p_one:.4f}"
    elif p_one < 0.05:
        sig_str = f"p = {p_one:.4f}"
    else:
        sig_str = f"p = {p_one:.4f}"

    n_improved = int((diff > 0).sum())
    n_worsened = int((diff < 0).sum())
    pct_improv = (diff.mean() / a.mean()) * 100 if a.mean() > 0 else 0.0

    result = {
        "comparison": f"{name_a} vs {name_b}",
        "n_buildings": len(merged),
        f"mean_mae_{name_a}": round(float(a.mean()), 4),
        f"mean_mae_{name_b}": round(float(b.mean()), 4),
        "mean_diff": round(float(diff.mean()), 4),
        "median_diff": round(float(np.median(diff)), 4),
        "pct_improvement": round(float(pct_improv), 2),
        "n_improved": n_improved,
        "n_worsened": n_worsened,
        "wilcoxon_stat": round(float(stat), 4),
        "p_value_onesided": float(p_one),
        "p_value_twosided": float(p_two),
        "effect_size_r": round(float(r), 4),
        "effect_interpretation": effect,
        "significant_005": bool(p_one < 0.05),
    }

    # Print summary
    sig_mark = "***" if p_one < 0.001 else "**" if p_one < 0.01 else "*" if p_one < 0.05 else "n.s."
    print(f"\n  {name_a} vs {name_b}")
    print(f"    {name_a}: {a.mean():.3f}   {name_b}: {b.mean():.3f}   "
          f"diff: {diff.mean():+.3f} ({pct_improv:+.1f}%)")
    print(f"    W={stat:.0f}, {sig_str} {sig_mark}, r={r:.3f} ({effect})")
    print(f"    {n_improved}/{len(merged)} improved, {n_worsened}/{len(merged)} worsened")

    # Paper-ready sentence
    if p_one < 0.05:
        sentence = (
            f"{name_b} achieves a mean MAE of {b.mean():.3f} kWh/h compared to "
            f"{a.mean():.3f} kWh/h for {name_a} at the t+1 horizon, representing a "
            f"{abs(pct_improv):.1f}% reduction across {len(merged)} buildings. "
            f"A Wilcoxon signed-rank test confirms this improvement is statistically "
            f"significant ({sig_str}, r={r:.2f}), with {n_improved} of {len(merged)} "
            f"buildings showing reduced prediction error."
        )
    else:
        sentence = (
            f"No statistically significant difference was found between {name_a} "
            f"(MAE={a.mean():.3f}) and {name_b} (MAE={b.mean():.3f}) at the t+1 "
            f"horizon ({sig_str}, r={r:.2f}, n={len(merged)})."
        )
    result["paper_sentence"] = sentence

    return result


def main():
    print("=" * 65)
    print("WILCOXON SIGNED-RANK TEST — All Pairwise Comparisons (t+1)")
    print(f"Excluding outliers: {EXCLUDE}")
    print("=" * 65)

    # ── Load all datasets ─────────────────────────────────────────────────────
    print("\nLoading datasets:")
    local     = load_h1(LOCAL_MLP_CSV,   "Local MLP")
    pers_adam = load_h1(PERS_ADAM_CSV,    "Pers FedAdam")
    pers_prox = load_h1(PERS_PROX_CSV,   "Pers FedProx")
    central   = load_h1(CENTRAL_MLP_CSV, "Centr MLP")

    # ── Define comparisons (A vs B: test whether B < A) ───────────────────────
    comparisons = [
        (local,   pers_adam, "Local MLP",    "Pers FedAdam"),
        (local,   pers_prox, "Local MLP",    "Pers FedProx"),
        (central, pers_adam, "Centr MLP",    "Pers FedAdam"),
        (central, pers_prox, "Centr MLP",    "Pers FedProx"),
        (central, local,     "Centr MLP",    "Local MLP"),
        (pers_adam, pers_prox, "Pers FedAdam", "Pers FedProx"),
    ]

    print(f"\n{'─' * 65}")
    print("Pairwise comparisons (H₁: first model has higher MAE than second):")
    print(f"{'─' * 65}")

    all_results = []
    for df_a, df_b, name_a, name_b in comparisons:
        col_a = [c for c in df_a.columns if c.startswith('mae_')][0]
        col_b = [c for c in df_b.columns if c.startswith('mae_')][0]

        merged = df_a.merge(df_b, on='building_id')

        if len(merged) < 10:
            print(f"\n  {name_a} vs {name_b}: SKIP — only {len(merged)} matched pairs")
            continue

        result = wilcoxon_pair(merged, col_a, col_b, name_a, name_b)
        all_results.append(result)

    # ── Results for paper
    print(f"\n{'=' * 65}")
    print("PAPER-READY SENTENCES")
    print(f"{'=' * 65}")
    for r in all_results:
        print(f"\n  [{r['comparison']}]")
        print(f"  {r['paper_sentence']}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = LOG_DIR / "wilcoxon_results_v5.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
