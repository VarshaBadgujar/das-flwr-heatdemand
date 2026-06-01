"""
DAS-FL Project — Step 4: Local MLP Baseline (Scenario 3)
Purpose:
    Train an independent MLP per building using only its own
    data. This is Scenario 3 — the MLP baseline that FL
    scenarios (4, 5, 6) must improve upon.

    Critical: comparing Scenario 3 vs 4/5/6
    isolates the FL effect (same architecture, only FL differs).

Output:
    - logs/local_mlp_results.csv
    - logs/local_mlp_report.txt
    - das-fl-paper/paper/figures/fig_local_mlp_accuracy.png
    - das-fl-paper/paper/figures/fig_local_mlp_horizons.png

Run from project root:
    python pipeline/train_local_mlp.py              # all buildings
    python pipeline/train_local_mlp.py --n 20       # first 20
    python pipeline/train_local_mlp.py --buildings B001 REDACTED
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
import os
import argparse
import time
import warnings
from pathlib import Path
from datetime import datetime

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
import gc
tf.get_logger().setLevel("ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dasfl.data_utils import (
    chronological_split,
    get_feature_target_split,
    get_building_files,
)
from dasfl.models import build_and_compile, get_model_summary

# ── Load config ─────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PROCESSED_PATH = PROJECT_ROOT / config["paths"]["processed_data"]
LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_HORIZONS = config["model"]["baseline"]["eval_horizons"]

# ── MLP Hyperparameters ─────────────────────────────────────
MLP_CONFIG = {
    "hidden_layers": [64, 32],
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 64,
    "patience": 5,         # early stopping patience
    "verbose": 0,           # 0=silent during training
}


# ── Metrics ─────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE, R²."""
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error,
        mean_absolute_percentage_error, r2_score,
    )

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


# ── Data preparation for MLP ────────────────────────────────

def prepare_mlp_data(X: pd.DataFrame, y: pd.Series):
    """Prepare data for MLP: convert to numpy, handle NaN.

    Returns:
        X_np, y_np as float32 numpy arrays.
    """
    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[mask].values.astype(np.float32)
    y_clean = y[mask].values.astype(np.float32)
    return X_clean, y_clean


def normalize_features(X_train, X_val, X_test):
    """Normalize features using train set statistics.

    MLP requires normalized inputs (unlike XGBoost).
    Uses mean/std from training set only to prevent leakage.

    Returns:
        X_train_norm, X_val_norm, X_test_norm, scaler_params
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    # Avoid division by zero for constant features
    std[std < 1e-8] = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, {"mean": mean, "std": std}


# ── Training ────────────────────────────────────────────────

def train_and_evaluate_building(
    filepath: Path,
    horizons: list[int],
) -> list[dict]:
    """Train local MLP for one building at all horizons.

    Returns list of result dicts (one per horizon).
    """
    building_id = filepath.stem
    df = pd.read_parquet(filepath)

    # Split chronologically
    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio=config["data"]["time"]["train_ratio"],
        val_ratio=config["data"]["time"]["val_ratio"],
        test_ratio=config["data"]["time"]["test_ratio"],
    )

    results = []

    for horizon in horizons:
        target_col = f"target_kwh_t+{horizon}"
        if target_col not in df.columns:
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

            # Convert to numpy and clean NaN
            X_train_np, y_train_np = prepare_mlp_data(X_train, y_train)
            X_val_np, y_val_np = prepare_mlp_data(X_val, y_val)
            X_test_np, y_test_np = prepare_mlp_data(X_test, y_test)

            if len(X_train_np) < 100 or len(X_test_np) < 50:
                results.append({
                    "building_id": building_id,
                    "horizon": horizon,
                    "status": "insufficient_data",
                    "n_train": len(X_train_np),
                    "n_test": len(X_test_np),
                })
                continue

            # Normalize features (critical for MLP convergence)
            X_train_norm, X_val_norm, X_test_norm, _ = normalize_features(
                X_train_np, X_val_np, X_test_np
            )

            n_features = X_train_norm.shape[1]

            # Build and compile model
            model = build_and_compile(
                n_features=n_features,
                hidden_layers=MLP_CONFIG["hidden_layers"],
                dropout_rate=MLP_CONFIG["dropout_rate"],
                learning_rate=MLP_CONFIG["learning_rate"],
            )

            # Early stopping on validation loss
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=MLP_CONFIG["patience"],
                restore_best_weights=True,
                verbose=0,
            )

            # Train
            history = model.fit(
                X_train_norm, y_train_np,
                validation_data=(X_val_norm, y_val_np),
                epochs=MLP_CONFIG["epochs"],
                batch_size=MLP_CONFIG["batch_size"],
                callbacks=[early_stop],
                verbose=MLP_CONFIG["verbose"],
            )

            # Predict on test set
            y_pred = model.predict(X_test_norm, verbose=0).flatten()
            y_pred = np.clip(y_pred, 0, None)  # kwh cannot be negative

            # Compute metrics
            metrics = compute_metrics(y_test_np, y_pred)

            # Training info
            n_epochs_trained = len(history.history["loss"])

            results.append({
                "building_id": building_id,
                "horizon": horizon,
                "status": "success",
                "n_train": len(X_train_np),
                "n_val": len(X_val_np),
                "n_test": len(X_test_np),
                "n_epochs": n_epochs_trained,
                "final_train_loss": round(history.history["loss"][-1], 6),
                "final_val_loss": round(history.history["val_loss"][-1], 6),
                **metrics,
            })

            # Clear model to free memory
            tf.keras.backend.clear_session()
            del model

        except Exception as e:
            results.append({
                "building_id": building_id,
                "horizon": horizon,
                "status": "error",
                "error": str(e),
            })

    return results


# ── Figures ─────────────────────────────────────────────────

def create_accuracy_figure(df_results: pd.DataFrame, fig_path: Path):
    """Box plots of MAE and R² across buildings per horizon."""
    df_ok = df_results[df_results["status"] == "success"].copy()
    if len(df_ok) == 0:
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Scenario 3: Local MLP Baseline — Per-Building Performance",
        fontsize=14, fontweight="bold",
    )

    horizons_present = sorted(df_ok["horizon"].unique())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E53935"]

    # (a) MAE
    ax = axes[0]
    data_mae = [df_ok[df_ok["horizon"] == h]["mae"].dropna().values
                for h in horizons_present]
    bp = ax.boxplot(data_mae, tick_labels=[f"t+{h}" for h in horizons_present],
                    patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors[:len(horizons_present)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("MAE (kWh/h)")
    ax.set_title("(a) MAE Distribution Across Buildings")

    # (b) R²
    ax = axes[1]
    data_r2 = [df_ok[df_ok["horizon"] == h]["r2"].dropna().values
               for h in horizons_present]
    bp2 = ax.boxplot(data_r2, tick_labels=[f"t+{h}" for h in horizons_present],
                     patch_artist=True, medianprops=dict(color="black", linewidth=2))
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


def create_comparison_figure(
    mlp_results: pd.DataFrame,
    xgb_results_path: Path,
    fig_path: Path,
):
    """Compare Local XGBoost vs Local MLP side by side."""
    mlp_ok = mlp_results[mlp_results["status"] == "success"].copy()

    # Load XGBoost results
    if not xgb_results_path.exists():
        print("  WARNING: XGBoost results not found, skipping comparison figure")
        return

    xgb_results = pd.read_csv(xgb_results_path)
    xgb_ok = xgb_results[xgb_results["status"] == "success"].copy()

    # Compute median metrics per horizon
    horizons = sorted(mlp_ok["horizon"].unique())

    mlp_summary = mlp_ok.groupby("horizon").agg(
        mae_median=("mae", "median"),
        r2_median=("r2", "median"),
    ).reset_index()
    mlp_summary["model"] = "Local MLP"

    xgb_summary = xgb_ok.groupby("horizon").agg(
        mae_median=("mae", "median"),
        r2_median=("r2", "median"),
    ).reset_index()
    xgb_summary["model"] = "Local XGBoost"

    combined = pd.concat([xgb_summary, mlp_summary])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Scenario 1 (XGBoost) vs Scenario 3 (MLP) — Local Training",
        fontsize=14, fontweight="bold",
    )

    # (a) MAE comparison
    ax = axes[0]
    width = 0.35
    x = np.arange(len(horizons))
    xgb_mae = xgb_summary.sort_values("horizon")["mae_median"].values
    mlp_mae = mlp_summary.sort_values("horizon")["mae_median"].values

    bars1 = ax.bar(x - width/2, xgb_mae, width, label="Local XGBoost",
                   color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width/2, mlp_mae, width, label="Local MLP",
                   color="#FF9800", alpha=0.8)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Median MAE (kWh/h)")
    ax.set_title("(a) MAE Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{h}" for h in horizons])
    ax.legend()

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # (b) R² comparison
    ax = axes[1]
    xgb_r2 = xgb_summary.sort_values("horizon")["r2_median"].values
    mlp_r2 = mlp_summary.sort_values("horizon")["r2_median"].values

    bars1 = ax.bar(x - width/2, xgb_r2, width, label="Local XGBoost",
                   color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width/2, mlp_r2, width, label="Local MLP",
                   color="#FF9800", alpha=0.8)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Median R²")
    ax.set_title("(b) R² Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{h}" for h in horizons])
    ax.legend()
    ax.axhline(y=0.0, color="gray", ls=":", lw=1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


# ── Report ──────────────────────────────────────────────────

def generate_report(df_results: pd.DataFrame, elapsed: float) -> str:
    """Generate text report of local MLP results."""
    lines = []
    lines.append("=" * 70)
    lines.append("DAS-FL PROJECT — LOCAL MLP BASELINE REPORT (SCENARIO 3)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Training time: {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    lines.append("=" * 70)

    lines.append(f"\nMLP Configuration:")
    for k, v in MLP_CONFIG.items():
        lines.append(f"   {k}: {v}")

    df_ok = df_results[df_results["status"] == "success"]
    n_buildings = df_ok["building_id"].nunique()
    n_total = df_results["building_id"].nunique()
    n_failed = n_total - n_buildings

    lines.append(f"\n1. SUMMARY")
    lines.append(f"   Buildings evaluated:    {n_buildings} / {n_total}")
    lines.append(f"   Failed / insufficient:  {n_failed}")
    lines.append(f"   Horizons evaluated:     {sorted(df_ok['horizon'].unique())}")

    if len(df_ok) > 0:
        lines.append(f"   Median epochs trained:  {df_ok['n_epochs'].median():.0f}")

    if len(df_ok) == 0:
        lines.append("\n   NO SUCCESSFUL RESULTS")
        return "\n".join(lines)

    lines.append(f"\n2. RESULTS BY FORECAST HORIZON")
    lines.append(f"   {'Horizon':<10} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R²':>8} {'Epochs':>8} {'Buildings':>10}")
    lines.append(f"   {'-'*66}")

    for h in sorted(df_ok["horizon"].unique()):
        df_h = df_ok[df_ok["horizon"] == h]
        lines.append(
            f"   t+{h:<7} "
            f"{df_h['mae'].median():>8.3f} "
            f"{df_h['rmse'].median():>8.3f} "
            f"{df_h['mape'].median():>7.1f}% "
            f"{df_h['r2'].median():>8.3f} "
            f"{df_h['n_epochs'].median():>7.0f} "
            f"{df_h['building_id'].nunique():>10}"
        )

    # Building-level spread at t+1
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


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train local MLP baseline (Scenario 3)."
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Process first N buildings only.")
    parser.add_argument("--buildings", nargs="+", type=str, default=None,
                        help="Specific building IDs.")
    args = parser.parse_args()

    print("=" * 60)
    print("DAS-FL PROJECT — STEP 4: LOCAL MLP BASELINE")
    print(f"Scenario 3: Each building trains MLP independently")
    print("=" * 60)

    # Find processed files
    if args.buildings:
        files = [PROCESSED_PATH / f"{bid}.parquet" for bid in args.buildings]
        files = [f for f in files if f.exists()]
    else:
        files = get_building_files(PROCESSED_PATH, config, "*.parquet")
        if args.n:
            files = files[:args.n]

    n_files = len(files)
    if n_files == 0:
        print("\nERROR: No processed files found. Run build_features.py first.")
        sys.exit(1)

    print(f"\n  Buildings to evaluate: {n_files}")
    print(f"  Horizons: {EVAL_HORIZONS}")
    print(f"  Model: MLP {MLP_CONFIG['hidden_layers']}")
    print(f"  Epochs: {MLP_CONFIG['epochs']} (early stop patience={MLP_CONFIG['patience']})")
    print(f"  Batch size: {MLP_CONFIG['batch_size']}")

    # Print model summary for first building
    sample_df = pd.read_parquet(files[0])
    target_cols = [c for c in sample_df.columns if c.startswith("target_")]
    n_features = len(sample_df.columns) - len(target_cols)
    sample_model = build_and_compile(n_features=n_features)
    summary = get_model_summary(sample_model)
    print(f"\n  Model params: {summary['total_params']:,} ({summary['size_kb']:.1f} KB)")
    print(f"  Input features: {n_features}")
    tf.keras.backend.clear_session()
    del sample_model

    # Train and evaluate
    print(f"\n[1/3] Training local MLP models...")
    start_time = time.time()
    all_results = []

    for i, fpath in enumerate(files):
        if (i + 1) % 20 == 0 or i == 0 or (i + 1) == n_files:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_files - i - 1) / rate if rate > 0 else 0
            print(f"  Training {i+1}/{n_files}... "
                  f"({rate:.1f} buildings/sec, ETA: {eta:.0f}s)")

        results = train_and_evaluate_building(fpath, EVAL_HORIZONS)
        all_results.extend(results)

        # Force memory cleanup between buildings
        tf.keras.backend.clear_session()
        gc.collect()

    elapsed = time.time() - start_time
    df_results = pd.DataFrame(all_results)

    # Save results
    df_results.to_csv(LOG_DIR / "local_mlp_results.csv", index=False)

    # Generate report
    print(f"\n[2/3] Generating report...")
    report = generate_report(df_results, elapsed)
    report_path = LOG_DIR / "local_mlp_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print("\n" + report)

    # Create figures
    print(f"\n[3/3] Creating figures...")
    create_accuracy_figure(
        df_results, FIG_DIR / "fig_local_mlp_accuracy.png"
    )
    create_comparison_figure(
        df_results,
        LOG_DIR / "local_baseline_results.csv",
        FIG_DIR / "fig_xgboost_vs_mlp_comparison.png",
    )

    print(f"\n{'=' * 60}")
    print("LOCAL MLP BASELINE COMPLETE")
    print(f"  Results: {LOG_DIR / 'local_mlp_results.csv'}")
    print(f"  Report:  {report_path}")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Time:    {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
