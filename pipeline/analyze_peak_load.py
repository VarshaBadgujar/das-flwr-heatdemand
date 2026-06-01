"""
DAS-FL Project : Peak Load & CUR Analysis
Purpose:
    Generate a figure showing capacity utilization patterns
    for a few sample buildings. Uses P95 as proxy for P_max
    (since real flow-limiter setpoints are not yet available).

Output:
    - das-fl-paper/paper/figures/fig_cur_analysis.png

Run from project root:
    python pipeline/analyze_peak_load.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
import sys
from pathlib import Path
from datetime import datetime


# ── Setup 
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RAW_DATA_PATH = PROJECT_ROOT / config["paths"]["raw_data"]
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]
LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
FIG_DIR.mkdir(parents=True, exist_ok=True)

KWH_MAX_VALID = config["data"]["quality"]["kwh_max_valid"]
KWH_SENTINEL = config["data"]["quality"]["kwh_sentinel"]


def load_and_clean(building_id: str) -> pd.DataFrame:
    """Load a building and apply basic cleaning."""
    fpath = RAW_DATA_PATH / f"{building_id}.pkl"
    df = pd.read_pickle(fpath)
    df.index.name = "timestamp"

    # Remove sentinels and obvious errors
    df.loc[df["kwh"] >= KWH_MAX_VALID, "kwh"] = np.nan
    df.loc[df["kwh"] < 0, "kwh"] = np.nan

    return df


def compute_cur_timeseries(df: pd.DataFrame, p_max: float) -> pd.Series:
    """Compute hourly CUR = P_actual / P_max."""
    return df["kwh"] / p_max


def select_diverse_buildings(summary_path: Path, n: int = 4) -> list:
    """Pick buildings at different consumption percentiles."""
    df_s = pd.read_csv(summary_path)
    df_s = df_s.dropna(subset=["kwh_mean"])
    df_s = df_s[df_s["kwh_mean"] > 0.5]  # exclude near-zero buildings
    df_s = df_s[df_s["n_rows"] >= 8760]   # at least 1 year
    df_s = df_s.sort_values("kwh_mean").reset_index(drop=True)

    percentiles = [0.15, 0.40, 0.70, 0.95]
    indices = [int(len(df_s) * p) for p in percentiles]
    return [str(df_s.iloc[i]["building_id"]) for i in indices]


def create_cur_figure(building_ids: list, fig_path: Path):
    """
    4-panel figure: one row per building.
    Each panel shows 4 weeks of winter data with:
    - Actual consumption (blue line)
    - P_max proxy = P95 of full history (red dashed)
    - Color-coded regions: green (under-utilized), red (over-utilized)
    """
    fig, axes = plt.subplots(len(building_ids), 1, figsize=(16, 3.5 * len(building_ids)),
                              sharex=False)
    fig.suptitle(
        "Capacity Utilization Analysis — Estimated CUR Using P95 as P_max Proxy\n"
        "(Real flow-limiter setpoints would improve this analysis)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    labels = ["Low consumption", "Low-medium", "Medium-high", "High consumption"]
    
    for i, (bid, label) in enumerate(zip(building_ids, labels)):
        ax = axes[i]
        df = load_and_clean(bid)

        # Compute P_max proxy (95th percentile of historical consumption)
        kwh_valid = df["kwh"].dropna()
        p_max_p95 = kwh_valid.quantile(0.95)
        p_max_p99 = kwh_valid.quantile(0.99)

        # Select 4 weeks of winter (Jan–Feb) for visualization
        winter = df[df.index.month.isin([1, 2])].copy()
        if len(winter) < 672:  # 4 weeks
            plot_data = df.iloc[:672].copy()
        else:
            plot_data = winter.iloc[:672].copy()

        # Compute CUR
        plot_data["cur"] = plot_data["kwh"] / p_max_p95

        # Time axis
        t = plot_data.index
        kwh = plot_data["kwh"].values
        cur = plot_data["cur"].values

        # Plot consumption
        ax.plot(t, kwh, color="#2196F3", lw=0.7, alpha=0.9, label="Actual consumption")

        # P_max lines
        ax.axhline(y=p_max_p95, color="#E53935", ls="--", lw=1.5,
                    label=f"P95 = {p_max_p95:.1f} kWh/h (proxy P_max)")
        ax.axhline(y=p_max_p99, color="#FF9800", ls=":", lw=1.2,
                    label=f"P99 = {p_max_p99:.1f} kWh/h")

        # Color-code over/under-utilization
        ax.fill_between(t, kwh, p_max_p95,
                         where=(kwh > p_max_p95),
                         color="#E53935", alpha=0.25, label="Over-utilized (CUR > 1.0)")
        ax.fill_between(t, kwh, 0,
                         where=(kwh <= p_max_p95 * 0.3),
                         color="#4CAF50", alpha=0.15, label="Low utilization (CUR < 0.3)")

        # Labels
        n_over = (cur > 1.0).sum()
        n_low = (cur < 0.3).sum()
        pct_over = n_over / len(cur) * 100
        pct_low = n_low / len(cur) * 100
        mean_cur = np.nanmean(cur)

        ax.set_ylabel("kWh/h")
        ax.set_title(
            f"{label} — Building {bid} | "
            f"Mean CUR: {mean_cur:.2f} | "
            f"Over-utilized: {pct_over:.1f}% | "
            f"Low-util: {pct_low:.1f}%",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t.min(), t.max())

    axes[-1].set_xlabel("Timestamp")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def create_cur_distribution_figure(building_ids_all: list, fig_path: Path):
    """
    Summary figure showing CUR distribution across many buildings.
    Picks 50 buildings and shows box plots of their CUR values.
    """
    summary_path = LOG_DIR / "building_summary.csv"
    df_s = pd.read_csv(summary_path)
    df_s = df_s.dropna(subset=["kwh_mean"])
    df_s = df_s[df_s["kwh_mean"] > 0.5]
    df_s = df_s[df_s["n_rows"] >= 8760]
    df_s = df_s.sort_values("kwh_mean").reset_index(drop=True)

    # Sample 50 buildings evenly across consumption range
    step = max(1, len(df_s) // 50)
    sample_ids = [str(df_s.iloc[i]["building_id"]) for i in range(0, len(df_s), step)][:50]

    # Compute mean CUR and CUR stats for each
    results = []
    for bid in sample_ids:
        try:
            df = load_and_clean(bid)
            kwh = df["kwh"].dropna()
            if len(kwh) < 100:
                continue
            p_max = kwh.quantile(0.95)
            if p_max <= 0:
                continue
            cur = kwh / p_max
            results.append({
                "building_id": bid,
                "kwh_mean": kwh.mean(),
                "cur_mean": cur.mean(),
                "cur_median": cur.median(),
                "pct_over": (cur > 1.0).mean() * 100,
                "pct_under_30": (cur < 0.3).mean() * 100,
            })
        except Exception:
            continue

    df_r = pd.DataFrame(results).sort_values("kwh_mean")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Capacity Utilization Across Buildings (P95 as P_max Proxy)",
        fontsize=13, fontweight="bold",
    )

    # (a) Mean CUR by building (sorted by consumption)
    ax = axes[0]
    colors = ["#E53935" if r["cur_mean"] > 0.8 else "#FF9800" if r["cur_mean"] > 0.5
              else "#4CAF50" for _, r in df_r.iterrows()]
    ax.bar(range(len(df_r)), df_r["cur_mean"], color=colors, alpha=0.8, width=1.0)
    ax.axhline(y=1.0, color="red", ls="--", lw=1, label="CUR = 1.0 (at capacity)")
    ax.axhline(y=0.5, color="orange", ls=":", lw=1, label="CUR = 0.5 (half capacity)")
    ax.set_xlabel("Buildings (sorted by mean consumption →)")
    ax.set_ylabel("Mean CUR")
    ax.set_title("(a) Mean Capacity Utilization per Building")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.5)

    # (b) Over-utilized % vs Under-utilized %
    ax = axes[1]
    ax.scatter(df_r["pct_under_30"], df_r["pct_over"],
               c=df_r["kwh_mean"], cmap="RdYlGn_r", s=50, alpha=0.7,
               edgecolors="gray", linewidths=0.5)
    ax.set_xlabel("% hours under-utilized (CUR < 0.3)")
    ax.set_ylabel("% hours over-utilized (CUR > 1.0)")
    ax.set_title("(b) Under-utilization vs Over-utilization")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Mean consumption (kWh/h)")
    ax.grid(True, alpha=0.3)

    # Annotate quadrants
    ax.axhline(y=5, color="gray", ls=":", alpha=0.5)
    ax.axvline(x=30, color="gray", ls=":", alpha=0.5)
    ax.text(60, 12, "Under-sized?\n(high peak + low base)",
            fontsize=8, color="#E53935", ha="center", style="italic")
    ax.text(10, 1, "Well-sized\n(good utilization)",
            fontsize=8, color="#4CAF50", ha="center", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


# ── Main ─
if __name__ == "__main__":
    print("=" * 60)
    print("DAS-FL PROJECT — PEAK LOAD & CUR ANALYSIS")
    print("=" * 60)

    summary_path = LOG_DIR / "building_summary.csv"
    if not summary_path.exists():
        print("ERROR: Run explore_data.py first to generate building_summary.csv")
        sys.exit(1)

    # 1. Select diverse buildings
    print("\n[1/3] Selecting diverse buildings...")
    building_ids = select_diverse_buildings(summary_path, n=4)
    print(f"  Selected: {building_ids}")

    # 2. CUR time series figure (4 buildings, 4 weeks each)
    print("\n[2/3] Creating CUR time series figure...")
    create_cur_figure(building_ids,
                       FIG_DIR / "fig_cur_analysis.png")

    # 3. CUR distribution across 50 buildings
    print("\n[3/3] Creating CUR distribution figure...")
    create_cur_distribution_figure(building_ids,
                                    FIG_DIR / "fig_cur_distribution.png")

    print("\n" + "=" * 60)
    print("PEAK LOAD ANALYSIS COMPLETE")
    print(f"  Figures: {FIG_DIR}/")
    print("=" * 60)