"""
DAS-FL Project — Step 2: Local XGBoost Baseline (Scenario 1)
Purpose:
    Train an independent XGBoost model per building using only
    its own data. This is Scenario 1 (local-only) — the baseline
    that FL must beat to justify its complexity.

    Evaluates at 4 forecast horizons: 1h, 6h, 24h, 168h.
    Produces per-building metrics + aggregated results.

Output:
    - logs/local_baseline_results.csv
    - logs/local_baseline_report.txt
    - das-fl-paper/paper/figures/fig_local_baseline_accuracy.png
    - das-fl-paper/paper/figures/fig_local_baseline_horizons.png
    - das-fl-paper/paper/figures/fig_feature_importance.png

Run from project root:
    python pipeline/train_local_baseline.py              # all processed
    python pipeline/train_local_baseline.py --n 20       # first 20
    python pipeline/train_local_baseline.py --buildings B001 REDACTED
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import yaml
import sys
import argparse
import time
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dasfl.data_utils import chronological_split, get_feature_target_split, get_building_files

# ── Load config 
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PROCESSED_PATH = PROJECT_ROOT / config["paths"]["processed_data"]
LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_HORIZONS = config["model"]["baseline"]["eval_horizons"]  # [1, 6, 24, 168]


# ── Metrics 

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE, R² for a single evaluation."""
    # Filter out any NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 10:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan,
                "n_samples": len(y_true)}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE: avoid division by zero — only compute where actual > threshold
    mape_mask = y_true > 0.5  # ignore near-zero hours for MAPE
    if mape_mask.sum() > 10:
        mape = mean_absolute_percentage_error(y_true[mape_mask], y_pred[mape_mask]) * 100
    else:
        mape = np.nan

    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 2) if not np.isnan(mape) else np.nan,
        "r2": round(r2, 4),
        "n_samples": len(y_true),
    }


# ── Model training

def train_and_evaluate_building(
    filepath: Path,
    horizons: list[int],
    seed: int = 42,
) -> list[dict]:
    """Train local XGBoost for one building at all horizons.

    Returns list of result dicts (one per horizon).
    """
    building_id = filepath.stem
    print(f"    reading parquet: {filepath.name}", flush=True)
    df = pd.read_parquet(filepath)
    print(f"    loaded {len(df)} rows x {len(df.columns)} cols", flush=True)

    # Split chronologically
    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio=config["data"]["time"]["train_ratio"],
        val_ratio=config["data"]["time"]["val_ratio"],
        test_ratio=config["data"]["time"]["test_ratio"],
    )

    results = []
    feature_importances = {}

    for horizon in horizons:
        target_col = f"target_kwh_t+{horizon}"
        print(f"    horizon t+{horizon}: start", flush=True)
        if target_col not in df.columns:
            print(f"    horizon t+{horizon}: missing target column — skip", flush=True)
            results.append({
                "building_id": building_id,
                "horizon": horizon,
                "status": "missing_target",
            })
            continue

        try:
            # Get features and targets
            X_train, y_train = get_feature_target_split(train_df, horizon=horizon)
            X_val, y_val = get_feature_target_split(val_df, horizon=horizon)
            X_test, y_test = get_feature_target_split(test_df, horizon=horizon)

            # Drop any remaining NaN
            train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
            test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())

            X_train, y_train = X_train[train_mask], y_train[train_mask]
            X_val, y_val = X_val[val_mask], y_val[val_mask]
            X_test, y_test = X_test[test_mask], y_test[test_mask]

            if len(X_train) < 100 or len(X_test) < 50:
                results.append({
                    "building_id": building_id,
                    "horizon": horizon,
                    "status": "insufficient_data",
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                })
                continue

            # Train XGBoost
            # NOTE: n_jobs=-1 hangs on this system with XGBoost 3.x (thread pool bug).
            # n_jobs=1 is reliable; single-threaded is fast enough per building.
            print(f"    horizon t+{horizon}: fitting XGBoost ({len(X_train)} train rows)", flush=True)
            t_fit = time.time()
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            print(f"    horizon t+{horizon}: fit done in {time.time()-t_fit:.1f}s", flush=True)

            # Predict on test set
            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0, None)  # kwh cannot be negative

            # Compute metrics
            metrics = compute_metrics(y_test.values, y_pred)

            # Store feature importance for horizon=1 (primary)
            if horizon == 1:
                importance = model.feature_importances_
                feature_importances = dict(zip(X_train.columns, importance))

            results.append({
                "building_id": building_id,
                "horizon": horizon,
                "status": "success",
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                **metrics,
            })
            print(f"    horizon t+{horizon}: RMSE={metrics['rmse']} MAE={metrics['mae']} R2={metrics['r2']}", flush=True)

        except Exception as e:
            print(f"    horizon t+{horizon}: ERROR — {e}", flush=True)
            results.append({
                "building_id": building_id,
                "horizon": horizon,
                "status": "error",
                "error": str(e),
            })

    return results, feature_importances


# ── Figures 

def create_accuracy_figure(df_results: pd.DataFrame, fig_path: Path):
    """Box plots of MAE across buildings for each horizon."""
    df_ok = df_results[df_results["status"] == "success"].copy()
    if len(df_ok) == 0:
        print("  WARNING: No successful results for accuracy figure")
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Scenario 1: Local XGBoost Baseline — Per-Building Performance",
        fontsize=14, fontweight="bold",
    )

    # (a) MAE distribution by horizon
    ax = axes[0]
    horizons_present = sorted(df_ok["horizon"].unique())
    data_for_box = [
        df_ok[df_ok["horizon"] == h]["mae"].dropna().values
        for h in horizons_present
    ]
    bp = ax.boxplot(
        data_for_box,
        labels=[f"t+{h}" for h in horizons_present],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E53935"]
    for patch, color in zip(bp["boxes"], colors[:len(horizons_present)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("MAE (kWh/h)")
    ax.set_title("(a) MAE Distribution Across Buildings")

    # (b) R² distribution by horizon
    ax = axes[1]
    data_for_box_r2 = [
        df_ok[df_ok["horizon"] == h]["r2"].dropna().values
        for h in horizons_present
    ]
    bp2 = ax.boxplot(
        data_for_box_r2,
        labels=[f"t+{h}" for h in horizons_present],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp2["boxes"], colors[:len(horizons_present)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("R²")
    ax.set_title("(b) R² Distribution Across Buildings")
    ax.axhline(y=0.0, color="gray", ls=":", lw=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def create_horizon_figure(df_results: pd.DataFrame, fig_path: Path):
    """Median MAE vs horizon — shows how accuracy degrades with distance."""
    df_ok = df_results[df_results["status"] == "success"].copy()
    if len(df_ok) == 0:
        return

    summary = df_ok.groupby("horizon").agg(
        mae_median=("mae", "median"),
        mae_q25=("mae", lambda x: x.quantile(0.25)),
        mae_q75=("mae", lambda x: x.quantile(0.75)),
        r2_median=("r2", "median"),
        n_buildings=("building_id", "nunique"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["horizon"], summary["mae_median"],
            "o-", color="#2196F3", lw=2, markersize=8, label="Median MAE")
    ax.fill_between(
        summary["horizon"],
        summary["mae_q25"],
        summary["mae_q75"],
        color="#2196F3", alpha=0.2, label="IQR (25th–75th)",
    )
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("MAE (kWh/h)", fontsize=12)
    ax.set_title("Local XGBoost: Accuracy vs Forecast Horizon", fontsize=14, fontweight="bold")
    ax.set_xticks(summary["horizon"])
    ax.set_xticklabels([f"t+{h}\n({h//24}d)" if h >= 24 else f"t+{h}" for h in summary["horizon"]])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add R² as secondary annotation
    for _, row in summary.iterrows():
        ax.annotate(
            f"R²={row['r2_median']:.2f}",
            xy=(row["horizon"], row["mae_median"]),
            xytext=(0, 15), textcoords="offset points",
            ha="center", fontsize=9, color="#666666",
        )

    plt.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def create_feature_importance_figure(
    all_importances: dict,
    fig_path: Path,
    top_n: int = 20,
):
    """Aggregated feature importance across buildings (SHAP-like from XGBoost)."""
    if not all_importances:
        print("  WARNING: No feature importances collected")
        return

    # Average importance across buildings
    imp_df = pd.DataFrame(all_importances).T
    mean_imp = imp_df.mean().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2196F3" if "kwh" in f else "#4CAF50" if "lag" in f
              else "#FF9800" if "rolling" in f
              else "#9C27B0" if ("sin" in f or "cos" in f or "weekend" in f)
              else "#E53935" for f in mean_imp.index]

    bars = ax.barh(range(len(mean_imp)), mean_imp.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(mean_imp)))
    ax.set_yticklabels(mean_imp.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Feature Importance (gain)", fontsize=12)
    ax.set_title(
        f"Top {top_n} Features — Local XGBoost (averaged across buildings)",
        fontsize=13, fontweight="bold",
    )

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Raw / current"),
        Patch(facecolor="#4CAF50", label="Lagged"),
        Patch(facecolor="#FF9800", label="Rolling stats"),
        Patch(facecolor="#9C27B0", label="Temporal"),
        Patch(facecolor="#E53935", label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


# ── Report 

def generate_report(df_results: pd.DataFrame, elapsed: float) -> str:
    """Generate a text report of local baseline results."""
    lines = []
    lines.append("=" * 70)
    lines.append("DAS-FL PROJECT — LOCAL XGBOOST BASELINE REPORT (SCENARIO 1)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Training time: {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    lines.append("=" * 70)

    df_ok = df_results[df_results["status"] == "success"]
    n_buildings = df_ok["building_id"].nunique()
    n_total = df_results["building_id"].nunique()
    n_failed = n_total - n_buildings

    lines.append(f"\n1. SUMMARY")
    lines.append(f"   Buildings evaluated:    {n_buildings} / {n_total}")
    lines.append(f"   Failed / insufficient:  {n_failed}")
    lines.append(f"   Horizons evaluated:     {sorted(df_ok['horizon'].unique())}")

    if len(df_ok) == 0:
        lines.append("\n   NO SUCCESSFUL RESULTS — check data pipeline")
        return "\n".join(lines)

    # Results by horizon
    lines.append(f"\n2. RESULTS BY FORECAST HORIZON")
    lines.append(f"   {'Horizon':<10} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R²':>8} {'Buildings':>10}")
    lines.append(f"   {'-'*58}")

    for h in sorted(df_ok["horizon"].unique()):
        df_h = df_ok[df_ok["horizon"] == h]
        lines.append(
            f"   t+{h:<7} "
            f"{df_h['mae'].median():>8.3f} "
            f"{df_h['rmse'].median():>8.3f} "
            f"{df_h['mape'].median():>7.1f}% "
            f"{df_h['r2'].median():>8.3f} "
            f"{df_h['building_id'].nunique():>10}"
        )

    # Best and worst buildings at t+1
    lines.append(f"\n3. BUILDING-LEVEL SPREAD (t+1 horizon)")
    df_h1 = df_ok[df_ok["horizon"] == 1].sort_values("mae")
    if len(df_h1) > 0:
        best = df_h1.iloc[0]
        worst = df_h1.iloc[-1]
        lines.append(f"   Best building:   {best['building_id']} — MAE={best['mae']:.3f}, R²={best['r2']:.3f}")
        lines.append(f"   Worst building:  {worst['building_id']} — MAE={worst['mae']:.3f}, R²={worst['r2']:.3f}")
        lines.append(f"   Median MAE:      {df_h1['mae'].median():.3f}")
        lines.append(f"   MAE IQR:         [{df_h1['mae'].quantile(0.25):.3f}, {df_h1['mae'].quantile(0.75):.3f}]")

    lines.append(f"\n{'=' * 70}")
    lines.append("END OF REPORT")
    return "\n".join(lines)


# ── Main 

def main():
    print("Starting...", flush=True)

    parser = argparse.ArgumentParser(
        description="Train local XGBoost baseline (Scenario 1)."
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Process first N buildings only.")
    parser.add_argument("--buildings", nargs="+", type=str, default=None,
                        help="Specific building IDs.")
    parser.add_argument("--building-ids-file", type=str, default=None,
                        help="Path to file with building IDs (one per line). "
                             "Ensures same buildings as FL experiments.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix for output files (e.g. 'v2' → local_baseline_results_v2.csv)")
    args = parser.parse_args()
    print("Args parsed.", flush=True)

    print("=" * 60)
    print("DAS-FL PROJECT — STEP 2: LOCAL XGBOOST BASELINE")
    print(f"Scenario 1: Each building trains independently")
    print("=" * 60, flush=True)

    # Find processed files
    print("Finding building files...", flush=True)
    if args.buildings:
        files = [PROCESSED_PATH / f"{bid}.parquet" for bid in args.buildings]
        files = [f for f in files if f.exists()]
    elif args.building_ids_file:
        # Load exact building IDs from file (matches FL experiments)
        ids_path = Path(args.building_ids_file)
        if not ids_path.is_absolute():
            ids_path = PROJECT_ROOT / ids_path
        with open(ids_path) as f:
            bid_list = [line.strip() for line in f if line.strip()]
        files = [PROCESSED_PATH / f"{bid}.parquet" for bid in bid_list]
        files = [f for f in files if f.exists()]
        print(f"  Loaded {len(files)} building IDs from {ids_path.name}", flush=True)
    else:
        files = get_building_files(PROCESSED_PATH, config, "*.parquet")
        if args.n:
            files = files[:args.n]
    print(f"Found {len(files)} files.", flush=True)

    n_files = len(files)
    if n_files == 0:
        print("\nERROR: No processed files found. Run build_features.py first.")
        sys.exit(1)

    print(f"\n  Buildings to evaluate: {n_files}")
    print(f"  Horizons: {EVAL_HORIZONS}")
    print(f"  Seed: {args.seed}")
    print(f"  Model: XGBoost (n_estimators=300, max_depth=6)", flush=True)

    # Train and evaluate
    print(f"\n[1/3] Training local models...", flush=True)
    start_time = time.time()
    all_results = []
    all_importances = {}

    for i, fpath in enumerate(files):
        print(f"  [{i+1}/{n_files}] Loading {fpath.stem}...", flush=True)
        t0 = time.time()

        results, importances = train_and_evaluate_building(fpath, EVAL_HORIZONS, args.seed)
        all_results.extend(results)
        if importances:
            all_importances[fpath.stem] = importances

        print(f"  [{i+1}/{n_files}] Done {fpath.stem} in {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.1f}s.", flush=True)
    df_results = pd.DataFrame(all_results)

    # Save results
    out_name = f"local_baseline_results_{args.suffix}.csv" if args.suffix else "local_baseline_results.csv"
    print(f"Saving results to {out_name}...", flush=True)
    df_results.to_csv(LOG_DIR / out_name, index=False)

    # Generate report
    print(f"\n[2/3] Generating report...", flush=True)
    report = generate_report(df_results, elapsed)
    report_path = LOG_DIR / "local_baseline_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print("\n" + report)

    # Create figures
    print(f"\n[3/3] Creating figures...", flush=True)
    create_accuracy_figure(
        df_results, FIG_DIR / "fig_local_baseline_accuracy.png"
    )
    print("  Accuracy figure done.", flush=True)
    create_horizon_figure(
        df_results, FIG_DIR / "fig_local_baseline_horizons.png"
    )
    print("  Horizon figure done.", flush=True)
    create_feature_importance_figure(
        all_importances, FIG_DIR / "fig_feature_importance.png"
    )
    print("  Feature importance figure done.", flush=True)

    print(f"\n{'=' * 60}")
    print("LOCAL BASELINE COMPLETE")
    print(f"  Results: {LOG_DIR / out_name}")
    print(f"  Report:  {report_path}")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Time:    {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
