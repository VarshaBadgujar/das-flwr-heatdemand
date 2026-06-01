"""
DAS-FL Project — Generate All Paper Figures
Creates publication-quality figures from existing results.

Figures:
    1. FL Convergence (MAE vs rounds, global vs local)
    2. Scenario Comparison (XGBoost vs MLP vs FedAvg)
    3. Per-building FL benefit (scatter: global vs local MAE)
    4. Horizon degradation (all scenarios)
    5. R² distribution across buildings
    6. Feature importance (top 20, with weather highlighted)

Run from project root:
    python pipeline/generate_paper_figures.py
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

LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent style for all paper figures
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
})

COLORS = {
    "xgboost": "#2196F3",
    "mlp": "#FF9800",
    "fedavg": "#4CAF50",
    "fedadam": "#FF9800",
    "personalised": "#E53935",
    "centralised": "#607D8B",
    "global": "#2196F3",
    "local": "#FF9800",
}


# ── Figure 1: FL Convergence ───────────────────────────────

def fig_fl_convergence():
    """Plot FL convergence from centralized evaluation history."""
    print("  Generating: FL Convergence...")

    # Try multiple convergence files
    conv_files = sorted(LOG_DIR.glob("fl_convergence_fedavg_h1_*.csv"))
    if not conv_files:
        print("    No convergence files found, skipping")
        return

    # Use the latest/largest one
    conv_file = conv_files[-1]
    df = pd.read_csv(conv_file)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["round"], df["loss"], color=COLORS["fedavg"],
            lw=2.5, marker="o", markersize=5, label="Centralized loss")

    # Annotate key points
    min_idx = df["loss"].idxmin()
    ax.annotate(f'Best: round {int(df.loc[min_idx, "round"])}',
                xy=(df.loc[min_idx, "round"], df.loc[min_idx, "loss"]),
                xytext=(10, 15), textcoords="offset points",
                fontsize=9, color=COLORS["fedavg"],
                arrowprops=dict(arrowstyle="->", color=COLORS["fedavg"]))

    ax.set_xlabel("FL Communication Round")
    ax.set_ylabel("Centralized Loss (MSE)")
    ax.set_title("FedAvg Convergence — Streaming FL")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_fl_convergence_centralized.png",
                dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_fl_convergence_centralized.png")
    plt.close()


# ── Figure 2: Per-Round Global vs Local ────────────────────

def fig_global_vs_local_per_round():
    """Plot global vs local MAE across rounds from per-client logs."""
    print("  Generating: Global vs Local per round...")

    metrics_dir = LOG_DIR / "fl_metrics"
    if not metrics_dir.exists():
        print("    No FL metrics directory, skipping")
        return

    files = sorted(metrics_dir.glob("client_eval_*.csv"))
    if not files:
        print("    No client eval files, skipping")
        return

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    if len(df) == 0:
        print("    Empty metrics, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Streaming FL — Global vs Local Model Performance",
                 fontsize=14, fontweight="bold")

    # (a) Median MAE per round
    ax = axes[0]
    for mtype, color, label in [
        ("global", COLORS["global"], "Global model (after aggregation)"),
        ("local", COLORS["local"], "Local model (before aggregation)"),
    ]:
        subset = df[df["model_type"] == mtype]
        if len(subset) == 0:
            continue
        medians = subset.groupby("round")["mae"].median()
        q25 = subset.groupby("round")["mae"].quantile(0.25)
        q75 = subset.groupby("round")["mae"].quantile(0.75)

        ax.plot(medians.index, medians.values, color=color, lw=2,
                marker="o", markersize=4, label=label)
        ax.fill_between(medians.index, q25.values, q75.values,
                        color=color, alpha=0.15)

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Median MAE (kWh/h)")
    ax.set_title("(a) Convergence: Global vs Local")
    ax.legend(fontsize=8)

    # (b) Per-building trajectories (global only, sample 10)
    ax = axes[1]
    global_df = df[df["model_type"] == "global"]
    buildings = global_df["building_id"].unique()
    sample = buildings[:min(10, len(buildings))]

    for bid in sample:
        bdf = global_df[global_df["building_id"] == bid]
        ax.plot(bdf["round"], bdf["mae"], alpha=0.4, lw=1)

    medians = global_df.groupby("round")["mae"].median()
    ax.plot(medians.index, medians.values, color="black", lw=3,
            label="Median", zorder=10)

    ax.set_xlabel("FL Round")
    ax.set_ylabel("MAE (kWh/h)")
    ax.set_title("(b) Per-Building Global MAE Trajectories")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig_fl_global_vs_local.png",
                dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_fl_global_vs_local.png")
    plt.close()


# ── Figure 3: FL Benefit Scatter ───────────────────────────

def fig_fl_benefit_scatter():
    """Scatter plot: global MAE vs local MAE at final round."""
    print("  Generating: FL Benefit Scatter...")

    metrics_dir = LOG_DIR / "fl_metrics"
    if not metrics_dir.exists():
        return

    files = sorted(metrics_dir.glob("client_eval_*.csv"))
    if not files:
        return

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    final_round = df["round"].max()
    final = df[df["round"] == final_round]

    g = final[final["model_type"] == "global"].set_index("building_id")
    l = final[final["model_type"] == "local"].set_index("building_id")

    comp = pd.DataFrame({
        "global_mae": g["mae"],
        "local_mae": l["mae"],
    }).dropna()

    if len(comp) == 0:
        print("    No comparison data, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by whether FL helps
    fl_helps = comp["global_mae"] < comp["local_mae"]
    ax.scatter(comp.loc[fl_helps, "local_mae"],
               comp.loc[fl_helps, "global_mae"],
               color=COLORS["fedavg"], alpha=0.7, s=40,
               label=f"FL helps ({fl_helps.sum()} buildings)")
    ax.scatter(comp.loc[~fl_helps, "local_mae"],
               comp.loc[~fl_helps, "global_mae"],
               color=COLORS["personalised"], alpha=0.7, s=40,
               label=f"Local better ({(~fl_helps).sum()} buildings)")

    max_val = max(comp["global_mae"].quantile(0.95),
                  comp["local_mae"].quantile(0.95))
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="Equal line")

    ax.set_xlabel("Local Model MAE (kWh/h)")
    ax.set_ylabel("Global Model MAE (kWh/h)")
    ax.set_title(f"FL Benefit per Building (Round {final_round})\n"
                 f"Points below diagonal = FL helps")
    ax.legend()
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_fl_benefit_scatter.png",
                dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_fl_benefit_scatter.png")
    plt.close()


# ── Figure 4: Scenario Comparison Bar Chart ────────────────

def fig_scenario_comparison():
    """Bar chart comparing all available scenarios across horizons."""
    print("  Generating: Scenario Comparison...")

    results = {}

    # Load XGBoost results (988 buildings)
    xgb_path = LOG_DIR / "local_baseline_results_matched100.csv"
    if xgb_path.exists():
        xgb = pd.read_csv(xgb_path)
        xgb_ok = xgb[xgb["status"] == "success"]
        for h in xgb_ok["horizon"].unique():
            hdf = xgb_ok[xgb_ok["horizon"] == h]
            results.setdefault(int(h), {})["Local XGBoost"] = {
                "mae": hdf["mae"].median(),
                "r2": hdf["r2"].median(),
            }

    # Load MLP results (100 buildings)
    mlp_path = LOG_DIR / "local_mlp_matched_results.csv"
    if mlp_path.exists():
        mlp = pd.read_csv(mlp_path)
        mlp_ok = mlp[mlp["status"] == "success"]
        for h in mlp_ok["horizon"].unique():
            hdf = mlp_ok[mlp_ok["horizon"] == h]
            results.setdefault(int(h), {})["Local MLP"] = {
                "mae": hdf["mae"].median(),
                "r2": hdf["r2"].median(),
            }

    # Load Centralised XGBoost results
    cxgb_path = LOG_DIR / "centralised_xgboost_results_matched100.csv"
    if cxgb_path.exists():
        cxgb = pd.read_csv(cxgb_path)
        cxgb_ok = cxgb[cxgb["status"] == "success"]
        for h in cxgb_ok["horizon"].unique():
            hdf = cxgb_ok[cxgb_ok["horizon"] == h]
            results.setdefault(int(h), {})["Centralised XGBoost"] = {
                "mae": hdf["mae"].median(),
                "r2": hdf["r2"].median(),
            }

    # Load FedAvg results
    fedavg_path = LOG_DIR / "fl_fedavg_final.csv"
    if fedavg_path.exists():
        fa = pd.read_csv(fedavg_path)
        fa_ok = fa[fa["status"] == "success"]
        for h in fa_ok["horizon"].unique():
            hdf = fa_ok[fa_ok["horizon"] == h]
            results.setdefault(int(h), {})["FedAvg"] = {
                "mae": hdf["mae"].median(),
                "r2": hdf["r2"].median(),
            }

    # Load FedAdam results
    fedprox_path = LOG_DIR / "fl_fedadam_final.csv"
    if fedprox_path.exists():
        fp = pd.read_csv(fedprox_path)
        fp_ok = fp[fp["status"] == "success"]
        for h in fp_ok["horizon"].unique():
            hdf = fp_ok[fp_ok["horizon"] == h]
            results.setdefault(int(h), {})["FedAdam"] = {
                "mae": hdf["mae"].median(),
                "r2": hdf["r2"].median(),
            }

    # Load Personalised results
    pers_path = LOG_DIR / "fl_personalised_final.csv"
    if pers_path.exists():
        ps = pd.read_csv(pers_path)
        ps_ok = ps[ps["status"] == "success"]
        for h in ps_ok["horizon"].unique():
            hdf = ps_ok[ps_ok["horizon"] == h]
            results.setdefault(int(h), {})["Personalised FL"] = {
                "mae": hdf["mae"].median(),
                "r2": hdf["r2"].median(),
            }

    if not results:
        print("    No results found, skipping")
        return

    # Build comparison chart
    horizons = sorted(results.keys())
    all_scenarios = []
    for h in horizons:
        all_scenarios.extend(results[h].keys())
    scenarios = list(dict.fromkeys(all_scenarios))  # unique, preserve order

    color_map = {
        "Local XGBoost": COLORS["xgboost"],
        "Local MLP": COLORS["mlp"],
        "FedAvg": COLORS["fedavg"],
        "FedAdam": COLORS["fedadam"],
        "Personalised FL": COLORS["personalised"],
        "Centralised XGBoost": COLORS["centralised"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Scenario Comparison Across Forecast Horizons",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(horizons))
    n_scenarios = len(scenarios)
    width = 0.8 / n_scenarios

    # (a) MAE
    ax = axes[0]
    for i, scenario in enumerate(scenarios):
        mae_vals = [results.get(h, {}).get(scenario, {}).get("mae", 0)
                    for h in horizons]
        bars = ax.bar(x + i * width - 0.4 + width/2, mae_vals, width,
                      label=scenario, color=color_map.get(scenario, "gray"),
                      alpha=0.85)
        # Value labels
        for bar, val in zip(bars, mae_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                        rotation=45)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Median MAE (kWh/h)")
    ax.set_title("(a) MAE Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{h}" for h in horizons])
    ax.legend(fontsize=8)

    # (b) R²
    ax = axes[1]
    for i, scenario in enumerate(scenarios):
        r2_vals = [results.get(h, {}).get(scenario, {}).get("r2", 0)
                   for h in horizons]
        bars = ax.bar(x + i * width - 0.4 + width/2, r2_vals, width,
                      label=scenario, color=color_map.get(scenario, "gray"),
                      alpha=0.85)
        for bar, val in zip(bars, r2_vals):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                        rotation=45)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Median R²")
    ax.set_title("(b) R² Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{h}" for h in horizons])
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", ls=":", lw=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig_scenario_comparison.png",
                dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_scenario_comparison.png")
    plt.close()


# ── Figure 5: R² Distribution ─────────────────────────────

def fig_r2_distribution():
    """Histogram of R² distribution for Local XGBoost baseline."""
    print("  Generating: R² Distribution...")

    xgb_path = LOG_DIR / "local_baseline_results_matched100.csv"
    if not xgb_path.exists():
        print("    No XGBoost results, skipping")
        return

    xgb = pd.read_csv(xgb_path)
    h1 = xgb[(xgb["horizon"] == 1) & (xgb["status"] == "success")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Local XGBoost — Building-Level Performance Distribution (t+1)",
                 fontsize=13, fontweight="bold")

    # (a) R² histogram
    ax = axes[0]
    r2_vals = h1["r2"].clip(-1, 1)  # clip extreme negatives for display

    bins_neg = np.linspace(-1, 0, 10)
    bins_pos = np.linspace(0, 1, 20)
    bins = np.concatenate([bins_neg, bins_pos[1:]])

    ax.hist(r2_vals, bins=bins, color=COLORS["xgboost"], edgecolor="white",
            alpha=0.8)
    ax.axvline(x=0, color="red", ls="--", lw=1.5, label="R²=0 (mean predictor)")
    ax.axvline(x=r2_vals.median(), color="black", ls="--", lw=1.5,
               label=f"Median: {r2_vals.median():.3f}")

    ax.set_xlabel("R²")
    ax.set_ylabel("Number of Buildings")
    ax.set_title(f"(a) R² Distribution (n={len(h1)})")
    ax.legend(fontsize=9)

    # (b) MAE vs R² scatter
    ax = axes[1]
    ax.scatter(h1["r2"], h1["mae"], alpha=0.4, s=15, color=COLORS["xgboost"])
    ax.set_xlabel("R²")
    ax.set_ylabel("MAE (kWh/h)")
    ax.set_title("(b) MAE vs R² per Building")
    ax.axvline(x=0, color="red", ls="--", lw=1, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig_r2_distribution.png",
                dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_r2_distribution.png")
    plt.close()


# ── Figure 6: Horizon Degradation ──────────────────────────

def fig_horizon_degradation():
    """Line plot showing MAE degradation across horizons for all scenarios."""
    print("  Generating: Horizon Degradation...")

    scenario_data = {}

    # XGBoost
    xgb_path = LOG_DIR / "local_baseline_results_matched100.csv"
    if xgb_path.exists():
        xgb = pd.read_csv(xgb_path)
        xgb_ok = xgb[xgb["status"] == "success"]
        scenario_data["Local XGBoost"] = xgb_ok.groupby("horizon")["mae"].median()

    # MLP
    mlp_path = LOG_DIR / "local_mlp_matched_results.csv"
    if mlp_path.exists():
        mlp = pd.read_csv(mlp_path)
        mlp_ok = mlp[mlp["status"] == "success"]
        scenario_data["Local MLP"] = mlp_ok.groupby("horizon")["mae"].median()

    # FedAvg
    fa_path = LOG_DIR / "fl_fedavg_final.csv"
    if fa_path.exists():
        fa = pd.read_csv(fa_path)
        fa_ok = fa[fa["status"] == "success"]
        scenario_data["FedAvg"] = fa_ok.groupby("horizon")["mae"].median()

    # Centralised XGBoost
    cxgb_path = LOG_DIR / "centralised_xgboost_results_matched100.csv"
    if cxgb_path.exists():
        cxgb = pd.read_csv(cxgb_path)
        cxgb_ok = cxgb[cxgb["status"] == "success"]
        scenario_data["Centralised XGBoost"] = cxgb_ok.groupby("horizon")["mae"].median()

    # FedAdam
    fp_path = LOG_DIR / "fl_fedadam_final.csv"
    if fp_path.exists():
        fp = pd.read_csv(fp_path)
        fp_ok = fp[fp["status"] == "success"]
        scenario_data["FedAdam"] = fp_ok.groupby("horizon")["mae"].median()

    # Personalised FL
    ps_path = LOG_DIR / "fl_personalised_final.csv"
    if ps_path.exists():
        ps = pd.read_csv(ps_path)
        ps_ok = ps[ps["status"] == "success"]
        scenario_data["Personalised FL"] = ps_ok.groupby("horizon")["mae"].median()

    if not scenario_data:
        print("    No results, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario, series in scenario_data.items():
        color = {
            "Local XGBoost": COLORS["xgboost"],
            "Local MLP": COLORS["mlp"],
            "Centralised XGBoost": COLORS["centralised"],
            "FedAvg": COLORS["fedavg"],
            "FedAdam": COLORS["fedadam"],
            "Personalised FL": COLORS["personalised"],
        }.get(scenario, "gray")

        ax.plot(series.index, series.values, color=color, lw=2.5,
                marker="o", markersize=8, label=scenario)

        # Annotate values
        for h, v in series.items():
            ax.annotate(f"{v:.3f}", xy=(h, v),
                        xytext=(5, 8), textcoords="offset points",
                        fontsize=8, color=color)

    ax.set_xlabel("Forecast Horizon (hours)")
    ax.set_ylabel("Median MAE (kWh/h)")
    ax.set_title("Prediction Accuracy vs Forecast Horizon",
                 fontsize=13, fontweight="bold")
    ax.set_xticks([1, 6, 24, 168])
    ax.set_xticklabels(["t+1\n(1h)", "t+6\n(6h)", "t+24\n(1d)", "t+168\n(7d)"])
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_horizon_degradation.png",
                dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_horizon_degradation.png")
    plt.close()


# ── Figure 7: Summary Results Table (as figure) ───────────

def fig_results_table():
    """Create a visual summary table of all scenario results."""
    print("  Generating: Results Summary Table...")

    rows = []

    # XGBoost
    xgb_path = LOG_DIR / "local_baseline_results_matched100.csv"
    if xgb_path.exists():
        xgb = pd.read_csv(xgb_path)
        xgb_ok = xgb[(xgb["status"] == "success") & (xgb["horizon"] == 1)]
        rows.append({
            "Scenario": "1. Local XGBoost",
            "Buildings": xgb_ok["building_id"].nunique(),
            "MAE": xgb_ok["mae"].median(),
            "RMSE": xgb_ok["rmse"].median(),
            "R²": xgb_ok["r2"].median(),
        })

    # MLP
    mlp_path = LOG_DIR / "local_mlp_matched_results.csv"
    if mlp_path.exists():
        mlp = pd.read_csv(mlp_path)
        mlp_ok = mlp[(mlp["status"] == "success") & (mlp["horizon"] == 1)]
        rows.append({
            "Scenario": "3. Local MLP",
            "Buildings": mlp_ok["building_id"].nunique(),
            "MAE": mlp_ok["mae"].median(),
            "RMSE": mlp_ok["rmse"].median(),
            "R²": mlp_ok["r2"].median(),
        })

    # Centralised XGBoost
    cxgb_path = LOG_DIR / "centralised_xgboost_results_matched100.csv"
    if cxgb_path.exists():
        cxgb = pd.read_csv(cxgb_path)
        cxgb_ok = cxgb[(cxgb["status"] == "success") & (cxgb["horizon"] == 1)]
        if len(cxgb_ok) > 0:
            rows.append({
                "Scenario": "2. Centralised XGBoost",
                "Buildings": cxgb_ok["building_id"].nunique(),
                "MAE": cxgb_ok["mae"].median(),
                "RMSE": cxgb_ok["rmse"].median(),
                "R²": cxgb_ok["r2"].median(),
            })

    # FedAvg
    fa_path = LOG_DIR / "fl_fedavg_final.csv"
    if fa_path.exists():
        fa = pd.read_csv(fa_path)
        fa_ok = fa[(fa["status"] == "success") & (fa["horizon"] == 1)]
        if len(fa_ok) > 0:
            rows.append({
                "Scenario": "4. FedAvg (streaming)",
                "Buildings": fa_ok["building_id"].nunique(),
                "MAE": fa_ok["mae"].median(),
                "RMSE": fa_ok["rmse"].median(),
                "R²": fa_ok["r2"].median(),
            })

    # FedAdam
    fp_path = LOG_DIR / "fl_fedadam_final.csv"
    if fp_path.exists():
        fp = pd.read_csv(fp_path)
        fp_ok = fp[(fp["status"] == "success") & (fp["horizon"] == 1)]
        if len(fp_ok) > 0:
            rows.append({
                "Scenario": "5. FedAdam (streaming)",
                "Buildings": fp_ok["building_id"].nunique(),
                "MAE": fp_ok["mae"].median(),
                "RMSE": fp_ok["rmse"].median(),
                "R²": fp_ok["r2"].median(),
            })

    # Personalised FL
    ps_path = LOG_DIR / "fl_personalised_final.csv"
    if ps_path.exists():
        ps = pd.read_csv(ps_path)
        ps_ok = ps[(ps["status"] == "success") & (ps["horizon"] == 1)]
        if len(ps_ok) > 0:
            rows.append({
                "Scenario": "6. Personalised FL",
                "Buildings": ps_ok["building_id"].nunique(),
                "MAE": ps_ok["mae"].median(),
                "RMSE": ps_ok["rmse"].median(),
                "R²": ps_ok["r2"].median(),
            })

    if not rows:
        print("    No results, skipping")
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=[[r["Scenario"], r["Buildings"],
                   f"{r['MAE']:.3f}", f"{r['RMSE']:.3f}", f"{r['R²']:.3f}"]
                  for _, r in df.iterrows()],
        colLabels=["Scenario", "Buildings", "MAE", "RMSE", "R²"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(5):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best MAE
    if len(rows) > 1:
        best_idx = df["MAE"].idxmin()
        for j in range(5):
            table[best_idx + 1, j].set_facecolor("#E8F5E9")

    ax.set_title("Results Summary — Horizon t+1",
                 fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_results_table.png",
                dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_results_table.png")
    plt.close()


# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)

    fig_fl_convergence()
    fig_global_vs_local_per_round()
    fig_fl_benefit_scatter()
    fig_scenario_comparison()
    fig_r2_distribution()
    fig_horizon_degradation()
    fig_results_table()

    print(f"\n{'=' * 60}")
    print("ALL FIGURES GENERATED")
    print(f"Directory: {FIG_DIR}")
    n_figs = len(list(FIG_DIR.glob("*.png")))
    print(f"Total figures: {n_figs}")
    print("=" * 60)