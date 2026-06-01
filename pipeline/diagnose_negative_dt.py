"""
DAS-FL Project — Diagnostic: Negative Delta-T Analysis
Purpose:
    Investigate the 804 buildings with negative delta-T (rt > ft).
    Understand: when does it happen, how often, which buildings,
    what patterns.

Run from project root:
    python pipeline/diagnose_negative_dt.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
import glob
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RAW_DATA_PATH = PROJECT_ROOT / config["paths"]["raw_data"]
LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def analyze_building_negative_dt(filepath: Path) -> dict:
    """Analyze negative dt patterns for one building."""
    building_id = filepath.stem
    df = pd.read_pickle(filepath)
    df.index.name = "timestamp"

    n_total = len(df)
    neg_mask = df["dt"] < 0
    n_neg = int(neg_mask.sum())

    if n_neg == 0:
        return {
            "building_id": building_id,
            "n_total": n_total,
            "n_negative_dt": 0,
            "pct_negative_dt": 0.0,
            "has_negative_dt": False,
        }

    neg_rows = df[neg_mask]

    # When does it happen? (hour of day)
    hour_counts = neg_rows.index.hour.value_counts().sort_index()
    peak_hour = hour_counts.idxmax() if len(hour_counts) > 0 else np.nan

    # Which month?
    month_counts = neg_rows.index.month.value_counts().sort_index()
    peak_month = month_counts.idxmax() if len(month_counts) > 0 else np.nan

    # Summer vs winter
    summer_months = [5, 6, 7, 8, 9]
    winter_months = [10, 11, 12, 1, 2, 3, 4]
    n_summer = int(neg_rows.index.month.isin(summer_months).sum())
    n_winter = int(neg_rows.index.month.isin(winter_months).sum())

    # How negative?
    dt_neg_values = neg_rows["dt"]
    min_dt = float(dt_neg_values.min())
    mean_dt = float(dt_neg_values.mean())
    median_dt = float(dt_neg_values.median())

    # What are ft and rt during negative dt?
    mean_ft_neg = float(neg_rows["ft"].mean())
    mean_rt_neg = float(neg_rows["rt"].mean())
    mean_kwh_neg = float(neg_rows["kwh"].mean()) if "kwh" in neg_rows.columns else np.nan

    # Compare with normal hours
    pos_rows = df[~neg_mask]
    mean_ft_pos = float(pos_rows["ft"].mean()) if len(pos_rows) > 0 else np.nan
    mean_rt_pos = float(pos_rows["rt"].mean()) if len(pos_rows) > 0 else np.nan
    mean_kwh_pos = float(pos_rows["kwh"].mean()) if len(pos_rows) > 0 else np.nan

    # Consecutive negative dt stretches
    neg_int = neg_mask.astype(int)
    changes = neg_int.diff().fillna(0)
    streak_starts = (changes == 1)
    if streak_starts.sum() > 0:
        # Calculate streak lengths
        streaks = []
        in_streak = False
        streak_len = 0
        for val in neg_mask:
            if val:
                streak_len += 1
                in_streak = True
            else:
                if in_streak:
                    streaks.append(streak_len)
                    streak_len = 0
                    in_streak = False
        if in_streak:
            streaks.append(streak_len)
        max_streak = max(streaks) if streaks else 0
        median_streak = float(np.median(streaks)) if streaks else 0
        n_streaks = len(streaks)
    else:
        max_streak = 0
        median_streak = 0
        n_streaks = 0

    # Is m3h (flow) zero or very low during negative dt?
    if "m3h" in neg_rows.columns:
        mean_m3h_neg = float(neg_rows["m3h"].mean())
        n_zero_flow_neg = int((neg_rows["m3h"] <= 0.01).sum())
    else:
        mean_m3h_neg = np.nan
        n_zero_flow_neg = 0

    return {
        "building_id": building_id,
        "n_total": n_total,
        "n_negative_dt": n_neg,
        "pct_negative_dt": round(n_neg / n_total * 100, 2),
        "has_negative_dt": True,
        "min_dt": round(min_dt, 2),
        "mean_dt_negative": round(mean_dt, 2),
        "median_dt_negative": round(median_dt, 2),
        "peak_hour": int(peak_hour) if not np.isnan(peak_hour) else np.nan,
        "peak_month": int(peak_month) if not np.isnan(peak_month) else np.nan,
        "n_summer": n_summer,
        "n_winter": n_winter,
        "summer_pct": round(n_summer / n_neg * 100, 1) if n_neg > 0 else 0,
        "mean_ft_during_neg": round(mean_ft_neg, 1),
        "mean_rt_during_neg": round(mean_rt_neg, 1),
        "mean_kwh_during_neg": round(mean_kwh_neg, 2),
        "mean_ft_normal": round(mean_ft_pos, 1),
        "mean_rt_normal": round(mean_rt_pos, 1),
        "mean_kwh_normal": round(mean_kwh_pos, 2),
        "mean_m3h_during_neg": round(mean_m3h_neg, 3) if not np.isnan(mean_m3h_neg) else np.nan,
        "n_zero_flow_during_neg": n_zero_flow_neg,
        "max_consecutive_hours": max_streak,
        "median_consecutive_hours": round(median_streak, 1),
        "n_streaks": n_streaks,
    }


def create_diagnostic_figures(df_neg: pd.DataFrame, raw_path: Path, fig_dir: Path):
    """Create diagnostic figures for negative delta-T analysis."""
    sns.set_style("whitegrid")

    # ── Figure 1: Overview of negative dt across buildings ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Negative Delta-T Diagnostic — 804 Buildings",
                 fontsize=14, fontweight="bold", y=0.98)

    df_has = df_neg[df_neg["has_negative_dt"]].copy()

    # (a) Distribution of % negative dt per building
    ax = axes[0, 0]
    ax.hist(df_has["pct_negative_dt"], bins=50, color="#E53935", edgecolor="white", alpha=0.85)
    ax.set_xlabel("% of hours with negative dt")
    ax.set_ylabel("Number of buildings")
    ax.set_title("(a) How common is negative dt?")
    ax.axvline(x=df_has["pct_negative_dt"].median(), color="black", ls="--", lw=1,
               label=f'Median: {df_has["pct_negative_dt"].median():.1f}%')
    ax.legend(fontsize=9)

    # (b) Summer vs Winter
    ax = axes[0, 1]
    ax.scatter(df_has["n_winter"], df_has["n_summer"], alpha=0.4, s=15, color="#2196F3")
    max_val = max(df_has["n_winter"].max(), df_has["n_summer"].max())
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="Equal line")
    ax.set_xlabel("Negative dt hours (winter)")
    ax.set_ylabel("Negative dt hours (summer)")
    ax.set_title("(b) Summer vs Winter occurrence")
    ax.legend(fontsize=9)

    # (c) How negative?
    ax = axes[0, 2]
    ax.hist(df_has["min_dt"], bins=50, color="#FF9800", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Most negative dt value (°C)")
    ax.set_ylabel("Number of buildings")
    ax.set_title("(c) Severity (minimum dt per building)")

    # (d) Peak hour distribution
    ax = axes[1, 0]
    hour_data = df_has["peak_hour"].dropna().astype(int)
    ax.hist(hour_data, bins=range(25), color="#4CAF50", edgecolor="white", alpha=0.85, align="left")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Number of buildings (peak hour)")
    ax.set_title("(d) When does negative dt most often occur?")
    ax.set_xticks(range(0, 24, 3))

    # (e) Temperature during negative dt vs normal
    ax = axes[1, 1]
    ax.scatter(df_has["mean_ft_during_neg"], df_has["mean_rt_during_neg"],
               alpha=0.4, s=15, color="#E53935", label="During negative dt")
    ax.scatter(df_has["mean_ft_normal"], df_has["mean_rt_normal"],
               alpha=0.2, s=15, color="#4CAF50", label="Normal hours")
    ax.plot([20, 100], [20, 100], "k--", lw=1, label="ft = rt")
    ax.set_xlabel("Supply temp ft (°C)")
    ax.set_ylabel("Return temp rt (°C)")
    ax.set_title("(e) ft vs rt: negative dt vs normal")
    ax.legend(fontsize=8)

    # (f) Consecutive hours
    ax = axes[1, 2]
    streak_data = df_has["max_consecutive_hours"]
    ax.hist(streak_data[streak_data <= 50], bins=50, color="#9C27B0",
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("Max consecutive hours of negative dt")
    ax.set_ylabel("Number of buildings")
    ax.set_title("(f) Duration of negative dt episodes")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "fig_negative_dt_diagnostic.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_dir / 'fig_negative_dt_diagnostic.png'}")
    plt.close()

    # ── Figure 2: Time series examples ──────────────────────
    # Pick 4 buildings: high %, medium %, low %, and the most severe
    df_sorted = df_has.sort_values("pct_negative_dt", ascending=False)
    examples = []
    if len(df_sorted) >= 4:
        examples.append(("Highest % negative dt", df_sorted.iloc[0]["building_id"]))
        examples.append(("Most negative dt value",
                         df_has.sort_values("min_dt").iloc[0]["building_id"]))
        mid_idx = len(df_sorted) // 2
        examples.append(("Median % negative dt", df_sorted.iloc[mid_idx]["building_id"]))
        examples.append(("Longest streak",
                         df_has.sort_values("max_consecutive_hours", ascending=False).iloc[0]["building_id"]))

    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(16, 3.5 * len(examples)))
        fig.suptitle("Negative Delta-T — Example Buildings",
                     fontsize=14, fontweight="bold", y=0.99)

        for i, (label, bid) in enumerate(examples):
            ax = axes[i]
            fpath = raw_path / f"{int(float(bid))}.pkl"
            if not fpath.exists():
                fpath = raw_path / f"{bid}.pkl"
            if not fpath.exists():
                ax.set_title(f"{label} — Building {bid} (file not found)")
                continue

            df = pd.read_pickle(fpath)
            df.index.name = "timestamp"

            # Pick 4 weeks with most negative dt
            df["neg_dt"] = df["dt"] < 0
            weekly_neg = df["neg_dt"].resample("W").sum()
            worst_week = weekly_neg.idxmax()
            start = worst_week - pd.Timedelta(days=14)
            end = worst_week + pd.Timedelta(days=14)
            window = df.loc[start:end]

            if len(window) == 0:
                window = df.iloc[:672]  # fallback: first 4 weeks

            # Plot dt with negative regions highlighted
            ax.plot(window.index, window["dt"], color="#2196F3", lw=0.7, label="dt (ft - rt)")
            ax.axhline(y=0, color="red", ls="-", lw=1.5, label="dt = 0")

            neg_window = window[window["dt"] < 0]
            if len(neg_window) > 0:
                ax.scatter(neg_window.index, neg_window["dt"],
                          color="#E53935", s=8, zorder=5, label=f"Negative dt ({len(neg_window)} hours)")

            stats = df_has[df_has["building_id"].astype(str) == str(bid)]
            pct = stats["pct_negative_dt"].values[0] if len(stats) > 0 else "?"

            ax.set_ylabel("dt (°C)")
            ax.set_title(f"{label} — Building {bid} | {pct}% negative dt hours", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Timestamp")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(fig_dir / "fig_negative_dt_examples.png", dpi=200, bbox_inches="tight")
        print(f"  Saved: {fig_dir / 'fig_negative_dt_examples.png'}")
        plt.close()


# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("DAS-FL PROJECT — NEGATIVE DELTA-T DIAGNOSTIC")
    print("=" * 60)

    files = sorted(RAW_DATA_PATH.glob("*.pkl"))
    print(f"\n  Scanning {len(files)} buildings for negative dt patterns...")

    results = []
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0 or i == 0 or (i + 1) == len(files):
            print(f"  Analyzing {i+1}/{len(files)}...")
        results.append(analyze_building_negative_dt(f))

    df_neg = pd.DataFrame(results)

    # Save detailed CSV
    csv_path = LOG_DIR / "negative_dt_analysis.csv"
    df_neg.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Summary
    df_has = df_neg[df_neg["has_negative_dt"]]
    n_affected = len(df_has)
    total_neg_hours = df_has["n_negative_dt"].sum()

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Buildings with negative dt: {n_affected} / {len(df_neg)}")
    print(f"  Total negative dt hours:    {total_neg_hours:,}")
    print(f"  Median % per building:      {df_has['pct_negative_dt'].median():.1f}%")
    print(f"  Max % per building:         {df_has['pct_negative_dt'].max():.1f}%")
    print(f"  Most negative dt value:     {df_has['min_dt'].min():.1f}°C")
    print(f"")
    print(f"  WHEN:")
    print(f"    Summer occurrence:         {df_has['summer_pct'].median():.0f}% (median)")
    print(f"    Most common peak hour:     {df_has['peak_hour'].mode().values}")
    print(f"")
    print(f"  TEMPERATURES DURING NEGATIVE DT:")
    print(f"    Mean supply temp (ft):     {df_has['mean_ft_during_neg'].mean():.1f}°C")
    print(f"    Mean return temp (rt):     {df_has['mean_rt_during_neg'].mean():.1f}°C")
    print(f"    (Normal: ft={df_has['mean_ft_normal'].mean():.1f}°C, rt={df_has['mean_rt_normal'].mean():.1f}°C)")
    print(f"")
    print(f"  FLOW DURING NEGATIVE DT:")
    print(f"    Mean m3h:                  {df_has['mean_m3h_during_neg'].mean():.3f}")
    print(f"    Zero flow hours:           {df_has['n_zero_flow_during_neg'].sum():,}")
    print(f"")
    print(f"  DURATION:")
    print(f"    Max consecutive hours:     {df_has['max_consecutive_hours'].max()}")
    print(f"    Median max streak:         {df_has['max_consecutive_hours'].median():.0f} hours")
    print(f"    Median streak count:       {df_has['n_streaks'].median():.0f} episodes")
    print(f"")
    print(f"  CONSUMPTION DURING NEGATIVE DT:")
    print(f"    Mean kwh during neg dt:    {df_has['mean_kwh_during_neg'].mean():.2f}")
    print(f"    Mean kwh normal:           {df_has['mean_kwh_normal'].mean():.2f}")

    # Create figures
    print(f"\nCreating diagnostic figures...")
    create_diagnostic_figures(df_neg, RAW_DATA_PATH, FIG_DIR)

    print(f"\n{'=' * 60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"  CSV:     {csv_path}")
    print(f"  Figures: {FIG_DIR}/")
    print(f"{'=' * 60}")