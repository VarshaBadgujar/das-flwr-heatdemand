"""
Plot FL convergence curves from per-round client evaluation logs.
Creates Figure 7 for the paper: MAE vs FL rounds (global and local).

Usage:
    python pipeline/plot_fl_convergence.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

LOG_DIR = PROJECT_ROOT / "logs" / "fl_metrics"
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]


def load_all_client_logs():
    """Load all per-client evaluation CSVs into one DataFrame."""
    files = sorted(LOG_DIR.glob("client_eval_*.csv"))
    if not files:
        print("No FL metric logs found. Run FL experiments first.")
        return None

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} client logs, {len(df)} total records")
    return df


def plot_convergence(df: pd.DataFrame, save_path: Path):
    """Plot MAE convergence: global vs local, across rounds."""
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("FL Convergence — Global vs Local Model",
                 fontsize=14, fontweight="bold")

    # (a) Median MAE across buildings per round
    ax = axes[0]
    for model_type, color, label in [
        ("global", "#2196F3", "Global model (after aggregation)"),
        ("local", "#FF9800", "Local model (before aggregation)"),
    ]:
        subset = df[df["model_type"] == model_type]
        if len(subset) == 0:
            continue
        grouped = subset.groupby("round")["mae"].agg(["median", "quantile"])
        medians = subset.groupby("round")["mae"].median()
        q25 = subset.groupby("round")["mae"].quantile(0.25)
        q75 = subset.groupby("round")["mae"].quantile(0.75)

        rounds = medians.index
        ax.plot(rounds, medians.values, color=color, lw=2, label=label, marker="o", markersize=4)
        ax.fill_between(rounds, q25.values, q75.values, color=color, alpha=0.15)

    ax.set_xlabel("FL Round")
    ax.set_ylabel("MAE (kWh/h)")
    ax.set_title("(a) Median MAE Across Buildings")
    ax.legend(fontsize=9)

    # (b) Per-building MAE trajectories (sample 10 buildings)
    ax = axes[1]
    global_df = df[df["model_type"] == "global"]
    buildings = global_df["building_id"].unique()
    sample = buildings[:min(10, len(buildings))]

    for bid in sample:
        bdf = global_df[global_df["building_id"] == bid]
        ax.plot(bdf["round"], bdf["mae"], alpha=0.5, lw=1)

    # Add median line
    medians = global_df.groupby("round")["mae"].median()
    ax.plot(medians.index, medians.values, color="black", lw=3,
            label="Median (all buildings)", zorder=10)

    ax.set_xlabel("FL Round")
    ax.set_ylabel("MAE (kWh/h)")
    ax.set_title("(b) Per-Building Global MAE Trajectories")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_global_vs_local_comparison(df: pd.DataFrame, save_path: Path):
    """Plot global vs local MAE per building at the final round."""
    final_round = df["round"].max()
    final = df[df["round"] == final_round]

    global_final = final[final["model_type"] == "global"].set_index("building_id")["mae"]
    local_final = final[final["model_type"] == "local"].set_index("building_id")["mae"]

    # Merge
    comparison = pd.DataFrame({
        "global_mae": global_final,
        "local_mae": local_final,
    }).dropna()

    if len(comparison) == 0:
        print("No comparison data available")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(comparison["local_mae"], comparison["global_mae"],
               alpha=0.6, s=30, color="#2196F3")

    max_val = max(comparison["global_mae"].max(), comparison["local_mae"].max())
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="Equal line")

    below = (comparison["global_mae"] < comparison["local_mae"]).sum()
    total = len(comparison)

    ax.set_xlabel("Local Model MAE (kWh/h)")
    ax.set_ylabel("Global Model MAE (kWh/h)")
    ax.set_title(f"Global vs Local Model (Round {final_round})\n"
                 f"Global better for {below}/{total} buildings ({100*below/total:.0f}%)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    df = load_all_client_logs()
    if df is not None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plot_convergence(df, FIG_DIR / "fig_fl_convergence.png")
        plot_global_vs_local_comparison(df, FIG_DIR / "fig_fl_global_vs_local.png")
