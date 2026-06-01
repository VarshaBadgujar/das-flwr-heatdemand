"""
DAS-FL Project — Step 3: Centralised XGBoost (Scenario 2)
Purpose:
    Pool data from ALL buildings into one dataset, train a
    single XGBoost model, then evaluate on each building's
    test set individually.

    This is the "no privacy" upper bound — what you could
    achieve if you had access to everyone's data.

    Comparison:
    - vs Scenario 1 (local XGBoost): does pooling help?
    - vs Scenarios 4-6 (FL): can FL approach centralised
      accuracy while preserving privacy?

Output:
    - logs/centralised_xgboost_results.csv
    - logs/centralised_xgboost_report.txt
    - das-fl-paper/paper/figures/fig_centralised_vs_local.png

Run from project root:
    python pipeline/train_centralised_xgboost.py          # all buildings
    python pipeline/train_centralised_xgboost.py --n 100  # sample 100
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dasfl.data_utils import (
    chronological_split,
    get_feature_target_split,
    get_building_files,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PROCESSED_PATH = PROJECT_ROOT / config["paths"]["processed_data"]
LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_HORIZONS = config["model"]["baseline"]["eval_horizons"]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, MAPE, R²."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 10:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan,
                "r2": np.nan, "n_samples": len(y_true)}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mape_mask = y_true > 0.5
    if mape_mask.sum() > 10:
        mape = mean_absolute_percentage_error(
            y_true[mape_mask], y_pred[mape_mask]) * 100
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


def main():
    parser = argparse.ArgumentParser(
        description="Train centralised XGBoost (Scenario 2)."
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Sample N buildings (default: all)")
    parser.add_argument("--max-train-rows", type=int, default=2_000_000,
                        help="Max training rows to keep memory manageable")
    parser.add_argument("--building-ids-file", type=str, default=None,
                        help="Path to file with building IDs (one per line). "
                             "Ensures same buildings as FL experiments.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix for output files (e.g. 'v2' → centralised_xgboost_results_v2.csv)")
    args = parser.parse_args()

    # Wire the CLI seed into the model hyperparameters
    XGB_PARAMS["random_state"] = args.seed

    print("=" * 60)
    print("DAS-FL PROJECT — STEP 3: CENTRALISED XGBOOST")
    print("Scenario 2: All data pooled, one model, no privacy")
    print("=" * 60)

    if args.building_ids_file:
        # Load exact building IDs from file (matches FL experiments)
        ids_path = Path(args.building_ids_file)
        if not ids_path.is_absolute():
            ids_path = PROJECT_ROOT / ids_path
        with open(ids_path) as f:
            bid_list = [line.strip() for line in f if line.strip()]
        files = [PROCESSED_PATH / f"{bid}.parquet" for bid in bid_list]
        files = [f for f in files if f.exists()]
        print(f"  Loaded {len(files)} building IDs from {ids_path.name}")
    else:
        files = get_building_files(PROCESSED_PATH, config, "*.parquet")
        if args.n:
            # Evenly spaced sampling for diversity
            indices = np.linspace(0, len(files) - 1, args.n, dtype=int)
            files = [files[i] for i in indices]

    n_files = len(files)
    print(f"\n  Buildings: {n_files}")
    print(f"  Horizons: {EVAL_HORIZONS}")
    print(f"  Max train rows: {args.max_train_rows:,}")
    print(f"  Seed: {args.seed}")

    all_results = []
    start_time = time.time()

    for horizon in EVAL_HORIZONS:
        target_col = f"target_kwh_t+{horizon}"
        print(f"\n  --- Horizon t+{horizon} ---")

        # ── Phase 1: Pool training data from all buildings ──
        print(f"  [1/3] Pooling training data...")
        pool_start = time.time()

        train_chunks = []
        test_sets = {}  # per-building test sets for evaluation
        n_skipped = 0

        for i, fpath in enumerate(files):
            building_id = fpath.stem

            try:
                df = pd.read_parquet(fpath)

                if target_col not in df.columns:
                    n_skipped += 1
                    continue

                # Chronological split
                train_df, val_df, test_df = chronological_split(
                    df,
                    train_ratio=config["data"]["time"]["train_ratio"],
                    val_ratio=config["data"]["time"]["val_ratio"],
                    test_ratio=config["data"]["time"]["test_ratio"],
                )

                # Get features and target
                X_train, y_train = get_feature_target_split(train_df, horizon=horizon)
                X_test, y_test = get_feature_target_split(test_df, horizon=horizon)

                # Clean NaN
                train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
                test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())

                X_train_clean = X_train[train_mask]
                y_train_clean = y_train[train_mask]
                X_test_clean = X_test[test_mask]
                y_test_clean = y_test[test_mask]

                if len(X_train_clean) < 100 or len(X_test_clean) < 50:
                    n_skipped += 1
                    continue

                # Add building_id column for tracking (removed before training)
                X_train_clean = X_train_clean.copy()
                X_train_clean["_building_id"] = building_id

                train_chunks.append((X_train_clean, y_train_clean))
                test_sets[building_id] = (X_test_clean, y_test_clean)

            except Exception as e:
                n_skipped += 1
                continue

            if (i + 1) % 100 == 0:
                print(f"    Loaded {i+1}/{n_files}...")

        if not train_chunks:
            print(f"  ERROR: No training data pooled for horizon t+{horizon}")
            continue

        # Concatenate all training data
        X_train_all = pd.concat([c[0] for c in train_chunks], ignore_index=True)
        y_train_all = pd.concat([c[1] for c in train_chunks], ignore_index=True)

        # Remove building_id column
        X_train_all = X_train_all.drop(columns=["_building_id"])

        total_rows = len(X_train_all)
        print(f"    Pooled: {total_rows:,} rows from {len(test_sets)} buildings"
              f" (skipped: {n_skipped})")

        # Subsample if too large (memory constraint)
        if total_rows > args.max_train_rows:
            sample_idx = np.random.RandomState(args.seed).choice(
                total_rows, args.max_train_rows, replace=False
            )
            X_train_all = X_train_all.iloc[sample_idx]
            y_train_all = y_train_all.iloc[sample_idx]
            print(f"    Subsampled to {args.max_train_rows:,} rows")

        pool_elapsed = time.time() - pool_start
        print(f"    Pooling time: {pool_elapsed:.0f}s")

        # ── Phase 2: Train one centralised model ────────────
        print(f"  [2/3] Training centralised XGBoost...")
        train_start = time.time()

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train_all, y_train_all)

        train_elapsed = time.time() - train_start
        print(f"    Training time: {train_elapsed:.0f}s")

        # Free pooled training data
        del X_train_all, y_train_all, train_chunks

        # ── Phase 3: Evaluate on each building's test set ───
        print(f"  [3/3] Evaluating on {len(test_sets)} buildings...")

        for building_id, (X_test, y_test) in test_sets.items():
            try:
                y_pred = model.predict(X_test)
                y_pred = np.clip(y_pred, 0, None)
                metrics = compute_metrics(y_test.values, y_pred)

                all_results.append({
                    "scenario": "centralised_xgboost",
                    "building_id": building_id,
                    "horizon": horizon,
                    "status": "success",
                    **metrics,
                })
            except Exception as e:
                all_results.append({
                    "scenario": "centralised_xgboost",
                    "building_id": building_id,
                    "horizon": horizon,
                    "status": "error",
                    "error": str(e),
                })

        del model, test_sets

    elapsed = time.time() - start_time
    df_results = pd.DataFrame(all_results)
    out_name = f"centralised_xgboost_results_{args.suffix}.csv" if args.suffix else "centralised_xgboost_results.csv"
    df_results.to_csv(LOG_DIR / out_name, index=False)

    # ── Report ──────────────────────────────────────────────
    df_ok = df_results[df_results["status"] == "success"]

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("DAS-FL PROJECT — CENTRALISED XGBOOST REPORT (SCENARIO 2)")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    report_lines.append("=" * 70)

    report_lines.append(f"\n1. SUMMARY")
    report_lines.append(f"   Buildings evaluated: {df_ok['building_id'].nunique()}")
    report_lines.append(f"   Horizons: {sorted(df_ok['horizon'].unique())}")

    report_lines.append(f"\n2. RESULTS BY FORECAST HORIZON")
    report_lines.append(f"   {'Horizon':<10} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R²':>8} {'Buildings':>10}")
    report_lines.append(f"   {'-'*56}")

    for h in sorted(df_ok["horizon"].unique()):
        hdf = df_ok[df_ok["horizon"] == h]
        report_lines.append(
            f"   t+{h:<7} "
            f"{hdf['mae'].median():>8.3f} "
            f"{hdf['rmse'].median():>8.3f} "
            f"{hdf['mape'].median():>7.1f}% "
            f"{hdf['r2'].median():>8.3f} "
            f"{hdf['building_id'].nunique():>10}"
        )

    report_lines.append(f"\n{'='*70}")
    report_lines.append("END OF REPORT")

    report = "\n".join(report_lines)
    print("\n" + report)

    with open(LOG_DIR / "centralised_xgboost_report.txt", "w") as f:
        f.write(report)

    # ── Comparison figure ───────────────────────────────────
    print("\n  Creating comparison figure...")

    local_path = LOG_DIR / "local_baseline_results.csv"
    if local_path.exists() and len(df_ok) > 0:
        local_df = pd.read_csv(local_path)
        local_ok = local_df[local_df["status"] == "success"]

        horizons = sorted(df_ok["horizon"].unique())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Scenario 1 (Local) vs Scenario 2 (Centralised) — XGBoost",
                     fontsize=14, fontweight="bold")

        width = 0.35
        x = np.arange(len(horizons))

        # MAE
        ax = axes[0]
        local_mae = [local_ok[local_ok["horizon"] == h]["mae"].median()
                     for h in horizons]
        central_mae = [df_ok[df_ok["horizon"] == h]["mae"].median()
                       for h in horizons]

        b1 = ax.bar(x - width/2, local_mae, width, label="Local XGBoost",
                     color="#2196F3", alpha=0.8)
        b2 = ax.bar(x + width/2, central_mae, width, label="Centralised XGBoost",
                     color="#607D8B", alpha=0.8)

        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)

        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel("Median MAE (kWh/h)")
        ax.set_title("(a) MAE")
        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{h}" for h in horizons])
        ax.legend()

        # R²
        ax = axes[1]
        local_r2 = [local_ok[local_ok["horizon"] == h]["r2"].median()
                    for h in horizons]
        central_r2 = [df_ok[df_ok["horizon"] == h]["r2"].median()
                      for h in horizons]

        b1 = ax.bar(x - width/2, local_r2, width, label="Local XGBoost",
                     color="#2196F3", alpha=0.8)
        b2 = ax.bar(x + width/2, central_r2, width, label="Centralised XGBoost",
                     color="#607D8B", alpha=0.8)

        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)

        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel("Median R²")
        ax.set_title("(b) R²")
        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{h}" for h in horizons])
        ax.legend()
        ax.axhline(y=0, color="gray", ls=":", lw=1)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(FIG_DIR / "fig_centralised_vs_local.png",
                    dpi=200, bbox_inches="tight")
        print(f"  Saved: fig_centralised_vs_local.png")
        plt.close()

    print(f"\n{'='*60}")
    print("CENTRALISED XGBOOST COMPLETE")
    print(f"  Results: {LOG_DIR / out_name}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()