"""
DAS-FL Project — Step 0: Data Exploration
Purpose:
    Scan ALL 989 buildings, assess data quality, identify the
    common time overlap, detect sentinel values and outliers,
    and produce summary statistics + paper-quality figures.

Output:
    - logs/data_exploration_report.txt
    - logs/building_summary.csv
    - das-fl-paper/paper/figures/fig_data_overview.png
    - das-fl-paper/paper/figures/fig_sample_timeseries.png

Run from project root:
    python pipeline/explore_data.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for DGX
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import glob
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path


# ── Project root detection 
# pipeline/explore_data.py -> PROJECT_ROOT is one level up
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── Load config 
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
if not CONFIG_PATH.exists():
    print(f"ERROR: Config not found at {CONFIG_PATH}")
    print("Run from project root: python pipeline/explore_data.py")
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ── Paths 
RAW_DATA_PATH = PROJECT_ROOT / config["paths"]["raw_data"]
LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Quality thresholds ────
Q = config["data"]["quality"]
KWH_MAX_VALID = Q["kwh_max_valid"]
KWH_SENTINEL = Q["kwh_sentinel"]
DT_MIN_VALID = Q["dt_min_valid"]
FT_MIN = Q["ft_min_valid"]
FT_MAX = Q["ft_max_valid"]
RT_MIN = Q["rt_min_valid"]
RT_MAX = Q["rt_max_valid"]
MIN_HOURS = Q["min_hours_per_building"]


# ── Helper functions 

def load_building(filepath: str) -> pd.DataFrame:
    """Load a single building pkl file.

    Returns DataFrame with DatetimeIndex named 'timestamp'.
    """
    df = pd.read_pickle(filepath)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex, got {type(df.index)}")
    df.index.name = "timestamp"
    return df


def compute_building_stats(filepath: str) -> dict:
    """Compute summary statistics for one building.

    Checks: row count, date range, completeness, sentinel values,
    outliers, temperature anomalies, missing values.
    """
    building_id = Path(filepath).stem
    try:
        df = load_building(filepath)
    except Exception as e:
        return {"building_id": building_id, "error": str(e)}

    n_rows = len(df)
    date_min = df.index.min()
    date_max = df.index.max()
    duration_days = (date_max - date_min).days
    expected_hours = duration_days * 24
    completeness = (n_rows / expected_hours * 100) if expected_hours > 0 else 0

    # Missing values per column
    missing_pct = (df.isnull().sum() / n_rows * 100).to_dict()

    # Target variable analysis
    kwh = df["kwh"]
    n_sentinel = int((kwh >= KWH_SENTINEL).sum())
    n_kwh_high = int(((kwh > KWH_MAX_VALID) & (kwh < KWH_SENTINEL)).sum())
    kwh_clean = kwh[(kwh >= 0) & (kwh < KWH_MAX_VALID)]

    # Temperature anomalies
    n_negative_dt = int((df["dt"] < DT_MIN_VALID).sum())
    n_ft_oor = int(((df["ft"] < FT_MIN) | (df["ft"] > FT_MAX)).sum())
    n_rt_oor = int(((df["rt"] < RT_MIN) | (df["rt"] > RT_MAX)).sum())
    n_zero_kwh = int((kwh == 0).sum())

    # Check hourly regularity
    if n_rows > 1:
        time_diffs = df.index.to_series().diff().dropna()
        median_gap_hours = time_diffs.median().total_seconds() / 3600
        n_irregular = int((time_diffs != pd.Timedelta(hours=1)).sum())
    else:
        median_gap_hours = np.nan
        n_irregular = 0

    return {
        "building_id": building_id,
        "n_rows": n_rows,
        "date_min": date_min,
        "date_max": date_max,
        "duration_days": duration_days,
        "completeness_pct": round(completeness, 1),
        "median_gap_hours": round(median_gap_hours, 2) if not np.isnan(median_gap_hours) else np.nan,
        "n_irregular_gaps": n_irregular,
        "missing_m3h_pct": round(missing_pct.get("m3h", 0), 2),
        "missing_kwh_pct": round(missing_pct.get("kwh", 0), 2),
        "kwh_mean": round(kwh_clean.mean(), 2) if len(kwh_clean) > 0 else np.nan,
        "kwh_median": round(kwh_clean.median(), 2) if len(kwh_clean) > 0 else np.nan,
        "kwh_max_clean": round(kwh_clean.max(), 2) if len(kwh_clean) > 0 else np.nan,
        "kwh_std": round(kwh_clean.std(), 2) if len(kwh_clean) > 0 else np.nan,
        "kwh_p95": round(kwh_clean.quantile(0.95), 2) if len(kwh_clean) > 0 else np.nan,
        "n_sentinel_values": n_sentinel,
        "n_kwh_high": n_kwh_high,
        "n_zero_kwh": n_zero_kwh,
        "zero_kwh_pct": round(n_zero_kwh / n_rows * 100, 2) if n_rows > 0 else 0,
        "n_negative_dt": n_negative_dt,
        "n_ft_out_of_range": n_ft_oor,
        "n_rt_out_of_range": n_rt_oor,
        "ft_mean": round(df["ft"].mean(), 1),
        "rt_mean": round(df["rt"].mean(), 1),
        "dt_mean": round(df["dt"].mean(), 1),
        "m3h_mean": round(df["m3h"].mean(), 2) if "m3h" in df.columns else np.nan,
        "error": np.nan,
    }


def scan_all_buildings(data_path: Path) -> pd.DataFrame:
    """Scan all building pkl files and compile summary DataFrame."""
    files = sorted(glob.glob(str(data_path / "*.pkl")))
    n_files = len(files)
    print(f"  Found {n_files} building files in {data_path}")

    results = []
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0 or i == 0 or (i + 1) == n_files:
            print(f"  Processing building {i+1}/{n_files}...")
        stats = compute_building_stats(f)
        results.append(stats)

    df_summary = pd.DataFrame(results)
    df_valid = df_summary[df_summary["error"].isna()].drop(columns=["error"])
    n_errors = len(df_summary) - len(df_valid)
    if n_errors > 0:
        print(f"  WARNING: {n_errors} buildings had load errors")
    return df_valid


def find_overlap_info(df_summary: pd.DataFrame, min_hours: int) -> dict:
    """Compute overlap statistics for buildings with sufficient data."""
    df_long = df_summary[df_summary["n_rows"] >= min_hours].copy()

    if len(df_long) == 0:
        return {"start": None, "end": None, "days": 0,
                "n_buildings_pool": 0, "n_fully_covering": 0}

    # Strict common overlap: latest start → earliest end
    strict_start = df_long["date_min"].max()
    strict_end = df_long["date_max"].min()

    if strict_start < strict_end:
        n_covering = int(((df_long["date_min"] <= strict_start) &
                          (df_long["date_max"] >= strict_end)).sum())
        return {
            "start": strict_start,
            "end": strict_end,
            "days": (strict_end - strict_start).days,
            "n_buildings_pool": len(df_long),
            "n_fully_covering": n_covering,
        }

    # No strict overlap — use 80th/20th percentile approach
    p80_start = df_long["date_min"].quantile(0.80)
    p20_end = df_long["date_max"].quantile(0.20)
    n_covering = int(((df_long["date_min"] <= p80_start) &
                      (df_long["date_max"] >= p20_end)).sum())
    return {
        "start": p80_start,
        "end": p20_end,
        "days": (p20_end - p80_start).days if p20_end > p80_start else 0,
        "n_buildings_pool": len(df_long),
        "n_fully_covering": n_covering,
        "note": "No strict overlap; used 80th/20th percentile bounds",
    }


def generate_report(df_summary: pd.DataFrame) -> str:
    """Generate a human-readable data exploration report."""
    lines = []
    lines.append("=" * 70)
    lines.append("DAS-FL PROJECT — DATA EXPLORATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    n = len(df_summary)

    # ── 1. Overview ─
    lines.append(f"\n1. DATASET OVERVIEW")
    lines.append(f"   Total buildings scanned:    {n}")
    lines.append(f"   Earliest data point:        {df_summary['date_min'].min()}")
    lines.append(f"   Latest data point:          {df_summary['date_max'].max()}")
    total_rows = df_summary["n_rows"].sum()
    lines.append(f"   Total data rows:            {total_rows:,}")

    # ── 2. Time coverage ──
    lines.append(f"\n2. TIME COVERAGE PER BUILDING")
    lines.append(f"   Duration (days) — min:      {df_summary['duration_days'].min()}")
    lines.append(f"   Duration (days) — median:   {df_summary['duration_days'].median():.0f}")
    lines.append(f"   Duration (days) — max:      {df_summary['duration_days'].max()}")
    lines.append(f"   Completeness (%) — median:  {df_summary['completeness_pct'].median():.1f}")

    n_sufficient = int((df_summary["n_rows"] >= MIN_HOURS).sum())
    lines.append(f"   Buildings with >= 1 year:   {n_sufficient} / {n}")

    # ── 3. Overlap ──
    overlap = find_overlap_info(df_summary, MIN_HOURS)
    lines.append(f"\n3. TIME OVERLAP (buildings with >= 1 year data)")
    lines.append(f"   Overlap start:      {overlap['start']}")
    lines.append(f"   Overlap end:        {overlap['end']}")
    days = overlap["days"]
    lines.append(f"   Duration:           {days} days ({days/365:.1f} years)")
    lines.append(f"   Buildings in pool:  {overlap['n_buildings_pool']}")
    lines.append(f"   Fully covering:     {overlap['n_fully_covering']}")
    if "note" in overlap:
        lines.append(f"   Note:               {overlap['note']}")

    # ── 4. Data quality ───
    lines.append(f"\n4. DATA QUALITY ISSUES")
    n_sentinel = int((df_summary["n_sentinel_values"] > 0).sum())
    total_sentinel = int(df_summary["n_sentinel_values"].sum())
    n_kwh_high = int((df_summary["n_kwh_high"] > 0).sum())
    n_neg_dt = int((df_summary["n_negative_dt"] > 0).sum())
    total_neg_dt = int(df_summary["n_negative_dt"].sum())
    lines.append(f"   Sentinel kwh ({KWH_SENTINEL}):  {n_sentinel} buildings, {total_sentinel} rows total")
    lines.append(f"   kwh > {KWH_MAX_VALID}:               {n_kwh_high} buildings")
    lines.append(f"   Negative delta-T:         {n_neg_dt} buildings, {total_neg_dt} rows total")
    lines.append(f"   Mean zero-kwh %:          {df_summary['zero_kwh_pct'].mean():.1f}%")
    lines.append(f"   Median irregular gaps:    {df_summary['n_irregular_gaps'].median():.0f} per building")

    # ── 5. Consumption ────
    lines.append(f"\n5. ENERGY CONSUMPTION (kwh, sentinels removed)")
    lines.append(f"   Mean (across buildings):    {df_summary['kwh_mean'].mean():.1f} kWh/h")
    lines.append(f"   Median (across buildings):  {df_summary['kwh_median'].median():.1f} kWh/h")
    lines.append(f"   P95 (across buildings):     {df_summary['kwh_p95'].median():.1f} kWh/h")
    lines.append(f"   Max (any building):         {df_summary['kwh_max_clean'].max():.1f} kWh/h")

    # ── 6. Temperature ────
    lines.append(f"\n6. TEMPERATURES")
    lines.append(f"   Supply temp (mean of means):  {df_summary['ft_mean'].mean():.1f} °C")
    lines.append(f"   Return temp (mean of means):  {df_summary['rt_mean'].mean():.1f} °C")
    lines.append(f"   Delta-T (mean of means):      {df_summary['dt_mean'].mean():.1f} °C")

    # ── 7. Heterogeneity ──
    kwh_mean = df_summary["kwh_mean"].dropna()
    cov = kwh_mean.std() / kwh_mean.mean() if kwh_mean.mean() > 0 else np.nan
    lines.append(f"\n7. HETEROGENEITY (FL motivation)")
    lines.append(f"   kwh_mean — min building:    {kwh_mean.min():.1f}")
    lines.append(f"   kwh_mean — max building:    {kwh_mean.max():.1f}")
    lines.append(f"   kwh_mean — CoV:             {cov:.2f}")
    lines.append(f"   → High CoV = high heterogeneity = strong FL motivation")
    lines.append(f"   kwh_mean — Q1:              {kwh_mean.quantile(0.25):.1f}")
    lines.append(f"   kwh_mean — Q3:              {kwh_mean.quantile(0.75):.1f}")
    lines.append(f"   kwh_mean — IQR:             {kwh_mean.quantile(0.75) - kwh_mean.quantile(0.25):.1f}")

    lines.append(f"\n{'=' * 70}")
    lines.append("END OF REPORT")
    return "\n".join(lines)


# ── Figures ────

def create_overview_figure(df_summary: pd.DataFrame, fig_path: Path):
    """6-panel overview figure for paper/presentation."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Dataset Overview — {len(df_summary)} District Heating Buildings (Borås, Sweden)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # (a) Duration distribution
    ax = axes[0, 0]
    ax.hist(df_summary["duration_days"] / 365, bins=30,
            color="#2196F3", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Duration (years)")
    ax.set_ylabel("Number of buildings")
    ax.set_title("(a) Data availability per building")
    ax.axvline(x=1.0, color="red", ls="--", lw=1, label="1-year minimum")
    ax.legend(fontsize=9)

    # (b) Consumption heterogeneity
    ax = axes[0, 1]
    vals = df_summary["kwh_mean"].dropna()
    ax.hist(vals, bins=50, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Mean hourly consumption (kWh/h)")
    ax.set_ylabel("Number of buildings")
    ax.set_title("(b) Consumption heterogeneity")

    # (c) Temporal coverage
    ax = axes[0, 2]
    df_sorted = df_summary.sort_values("date_min").reset_index(drop=True)
    step = max(1, len(df_sorted) // 80)
    for i in range(0, len(df_sorted), step):
        row = df_sorted.iloc[i]
        ax.plot([row["date_min"], row["date_max"]], [i, i],
                color="#FF9800", alpha=0.6, lw=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Building (sorted by start)")
    ax.set_title("(c) Temporal coverage per building")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # (d) Temperature distributions
    ax = axes[1, 0]
    ax.hist(df_summary["ft_mean"], bins=30, color="#E91E63", alpha=0.7, label="Supply (ft)")
    ax.hist(df_summary["rt_mean"], bins=30, color="#03A9F4", alpha=0.7, label="Return (rt)")
    ax.set_xlabel("Mean temperature (°C)")
    ax.set_ylabel("Number of buildings")
    ax.set_title("(d) Supply / return temperature")
    ax.legend(fontsize=9)

    # (e) Completeness
    ax = axes[1, 1]
    ax.hist(df_summary["completeness_pct"], bins=30,
            color="#9C27B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Completeness (%)")
    ax.set_ylabel("Number of buildings")
    ax.set_title("(e) Data completeness")
    ax.axvline(x=90, color="red", ls="--", lw=1, label="90% threshold")
    ax.legend(fontsize=9)

    # (f) Quality issues
    ax = axes[1, 2]
    issues = {
        f"Sentinel\n(kwh={KWH_SENTINEL})": int((df_summary["n_sentinel_values"] > 0).sum()),
        f"kwh > {KWH_MAX_VALID}": int((df_summary["n_kwh_high"] > 0).sum()),
        "Negative\ndelta-T": int((df_summary["n_negative_dt"] > 0).sum()),
    }
    bars = ax.barh(list(issues.keys()), list(issues.values()), color="#F44336", alpha=0.8)
    for bar, val in zip(bars, issues.values()):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)
    ax.set_xlabel("Number of buildings affected")
    ax.set_title("(f) Data quality issues")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def create_timeseries_figure(data_path: Path, df_summary: pd.DataFrame,
                              fig_path: Path):
    """3 buildings: low / medium / high consumption (2 weeks winter)."""
    df_s = df_summary.dropna(subset=["kwh_mean"]).sort_values("kwh_mean").reset_index(drop=True)
    indices = [int(len(df_s) * p) for p in [0.10, 0.50, 0.90]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(
        "Sample Time Series — Low / Medium / High Consumption Buildings",
        fontsize=13, fontweight="bold",
    )
    labels = ["Low consumption", "Medium consumption", "High consumption"]
    colors = ["#2196F3", "#4CAF50", "#F44336"]

    for i, idx in enumerate(indices):
        row = df_s.iloc[idx]
        fpath = data_path / f"{row['building_id']}.pkl"
        df = pd.read_pickle(fpath)
        df.loc[df["kwh"] >= KWH_MAX_VALID, "kwh"] = np.nan

        # Prefer 2 weeks of winter (Jan/Feb)
        winter = df[df.index.month.isin([1, 2])]
        plot_data = winter.iloc[:336] if len(winter) >= 336 else df.iloc[:336]

        ax = axes[i]
        ax.plot(plot_data.index, plot_data["kwh"], color=colors[i], lw=0.6, alpha=0.9)
        ax.set_ylabel("kWh/h")
        ax.set_title(
            f'{labels[i]} — Building {row["building_id"]} '
            f'(mean: {row["kwh_mean"]:.1f} kWh/h)',
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestamp")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


# ── Main execution ──

if __name__ == "__main__":
    print("=" * 60)
    print("DAS-FL PROJECT — STEP 0: DATA EXPLORATION")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)

    # Check data path
    if not RAW_DATA_PATH.exists():
        print(f"\nERROR: Data path not found: {RAW_DATA_PATH}")
        print("Create symlink:")
        print("  ln -sf /path/to/data/features/meter_data/heat/be data/raw/be_heat")
        sys.exit(1)

    pkl_count = len(glob.glob(str(RAW_DATA_PATH / "*.pkl")))
    print(f"  Data path: {RAW_DATA_PATH} ({pkl_count} files)")

    # 1. Scan all buildings
    print("\n[1/4] Scanning all buildings...")
    df_summary = scan_all_buildings(RAW_DATA_PATH)

    # 2. Save per-building CSV
    csv_path = LOG_DIR / "building_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[2/4] Building summary saved: {csv_path}")

    # 3. Generate report
    report = generate_report(df_summary)
    report_path = LOG_DIR / "data_exploration_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[3/4] Report saved: {report_path}")
    print("\n" + report)

    # 4. Create figures
    print("\n[4/4] Creating figures...")
    create_overview_figure(df_summary, FIG_DIR / "fig_data_overview.png")
    create_timeseries_figure(RAW_DATA_PATH, df_summary,
                              FIG_DIR / "fig_sample_timeseries.png")

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print(f"  Report:  {report_path}")
    print(f"  CSV:     {csv_path}")
    print(f"  Figures: {FIG_DIR}/")
    print("=" * 60)
