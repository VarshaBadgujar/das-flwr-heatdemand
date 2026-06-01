"""
DAS-FL Project — Centralised MLP Baseline (Scenario 2)
Purpose:
    Train ONE MLP on the pooled (concatenated) training data
    from all participating buildings, then evaluate per-building.

    This is the centralised upper-bound baseline: same MLP
    architecture as Scenario 3 (local) and Scenarios 4-6 (FL),
    but with full access to every building's data.

    Per-building normalisation is used (each building's train
    mean/std), matching how FL clients normalise locally.
    This isolates the training approach (pooled vs federated)
    without confounding normalisation differences.

Output:
    - logs/centralised_mlp_results_{suffix}.csv

Run from project root:
    python pipeline/train_centralised_mlp.py \
        --building-ids-file logs/fl_building_ids_250.txt \
        --suffix v2

    python pipeline/train_centralised_mlp.py --n 20  # first 20
"""

import numpy as np
import pandas as pd
import sys
import os
import gc
import time
import argparse
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dasfl.task import (
    get_building_ids,
    build_model,
    load_client_data,
    evaluate_client,
)

import yaml

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
LOG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_HORIZONS = config["model"]["baseline"]["eval_horizons"]


def run_centralised_mlp(
    building_ids: list[str],
    horizons: list[int],
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 5,
    pred_suffix: str = "",
) -> pd.DataFrame:
    """Train one pooled MLP per horizon, evaluate per building.

    Args:
        building_ids: Buildings to include.
        horizons: Forecast horizons (e.g. [1, 6, 24, 168]).
        epochs: Max training epochs (early stopping used).
        batch_size: Training batch size.
        patience: Early stopping patience.

    Returns:
        DataFrame with per-building, per-horizon results.
    """
    all_results = []

    for horizon in horizons:
        print(f"\n--- Horizon t+{horizon} ---")

        # Step 1: Load per-building data (each normalised with own mean/std)
        client_data = []
        for i, bid in enumerate(building_ids):
            try:
                data = load_client_data(i, building_ids, horizon=horizon)
                client_data.append(data)
            except Exception as e:
                print(f"  Skip {bid}: {e}")
                all_results.append({
                    "building_id": bid,
                    "horizon": horizon,
                    "status": "error",
                    "error": str(e),
                })

        if not client_data:
            print("  No buildings loaded, skipping horizon")
            continue

        n_features = client_data[0]["n_features"]

        # Step 2: Concatenate training and validation data
        X_train_all = np.concatenate([d["X_train"] for d in client_data])
        y_train_all = np.concatenate([d["y_train"] for d in client_data])
        X_val_all = np.concatenate([d["X_val"] for d in client_data])
        y_val_all = np.concatenate([d["y_val"] for d in client_data])

        print(f"  Pooled data: {len(X_train_all):,} train, "
              f"{len(X_val_all):,} val rows "
              f"from {len(client_data)} buildings")

        # Step 3: Train single centralised model
        model = build_model(n_features)

        history = model.fit(
            X_train_all, y_train_all,
            validation_data=(X_val_all, y_val_all),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=patience, restore_best_weights=True,
            )],
            verbose=0,
        )

        n_epochs = len(history.history["loss"])
        final_val_loss = history.history["val_loss"][-1]
        print(f"  Trained {n_epochs} epochs, val_loss={final_val_loss:.4f}")

        # Step 4: Evaluate on each building's own test set
        for data in client_data:
            bid = data["building_id"]
            metrics = evaluate_client(model, data["X_test"], data["y_test"])

            if horizon == 1:
                pred_dir = LOG_DIR / "predictions" / pred_suffix if pred_suffix else LOG_DIR / "predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)
                y_pred = model.predict(data["X_test"], verbose=0).flatten()
                np.savez(pred_dir / f"centralised_mlp_{bid}_h{horizon}.npz",
                         y_actual=data["y_test"],
                         y_pred=y_pred)

            all_results.append({
                "building_id": bid,
                "horizon": horizon,
                "status": "success",
                "mae": round(metrics["mae"], 4),
                "rmse": round(metrics["rmse"], 4),
                "r2": round(metrics["r2"], 4),
                "n_test": metrics["n_test"],
                "n_train_pooled": len(X_train_all),
                "n_buildings": len(client_data),
                "epochs": n_epochs,
            })

        # Cleanup
        tf.keras.backend.clear_session()
        del model, X_train_all, y_train_all, X_val_all, y_val_all, client_data
        gc.collect()

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Train centralised (pooled) MLP baseline."
    )
    parser.add_argument(
        "--building-ids-file", type=str, default=None,
        help="Path to text file with one building ID per line "
             "(e.g. logs/fl_building_ids_250.txt).",
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Use first N buildings (ignored if --building-ids-file set).")
    parser.add_argument("--suffix", type=str, default="v2",
                        help="Output file suffix (default: v2).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=None,
                        help="Single horizon (default: all 4).")
    parser.add_argument("--pred-suffix", type=str, default="",
                        help="Subdirectory under logs/predictions/ for versioned .npz output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Resolve building IDs
    if args.building_ids_file:
        id_path = Path(args.building_ids_file)
        if not id_path.is_absolute():
            id_path = PROJECT_ROOT / id_path
        with open(id_path) as f:
            building_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(building_ids)} building IDs from {id_path.name}")
    else:
        building_ids = get_building_ids()
        if args.n:
            building_ids = building_ids[:args.n]

    horizons = [args.horizon] if args.horizon else EVAL_HORIZONS

    print("=" * 60)
    print("DAS-FL PROJECT — CENTRALISED MLP BASELINE (SCENARIO 2)")
    print(f"  Buildings: {len(building_ids)}")
    print(f"  Horizons:  {horizons}")
    print(f"  Epochs:    {args.epochs} (patience={args.patience})")
    print("=" * 60)

    start = time.time()

    df = run_centralised_mlp(
        building_ids, horizons,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        pred_suffix=args.pred_suffix,
    )

    out_path = LOG_DIR / f"centralised_mlp_results_{args.suffix}.csv"
    df.to_csv(out_path, index=False)

    elapsed = time.time() - start

    # Summary
    df_ok = df[df["status"] == "success"]
    print(f"\n{'='*60}")
    print(f"CENTRALISED MLP COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    for h in sorted(df_ok["horizon"].unique()):
        hdf = df_ok[df_ok["horizon"] == h]
        print(f"  t+{h:<4}  MAE={hdf['mae'].median():.3f}  "
              f"R²={hdf['r2'].median():.3f}  N={len(hdf)}")

    print(f"\n  Results: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
