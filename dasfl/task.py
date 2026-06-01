"""
DAS-FL Project — task.py
Core functions shared by FL client and server:
    - Data loading and preparation per client
    - MLP model creation
    - Training and evaluation functions

This follows Flower's convention where task.py contains
shared logic imported by client_app.py and server_app.py.

Extends (2025) thesis:
    - 51 features (thesis: 2)
    - 4 forecast horizons (thesis: 1)
    - 988 clients (thesis: 3)
    - Feature normalization per client
"""

import numpy as np
import pandas as pd
import yaml
import os
import warnings
from pathlib import Path
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")


# ── Global weight storage (updated by evaluate_fn each round) ──
_latest_global_weights = None

# ── Configuration 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

PROCESSED_PATH = PROJECT_ROOT / _config["paths"]["processed_data"]
EXCLUDE_BUILDINGS = _config.get("data", {}).get("quality", {}).get("exclude_buildings", [])
EXCLUDE_IDS = set(str(b) for b in EXCLUDE_BUILDINGS)


def set_data_dir(path):
    global PROCESSED_PATH
    PROCESSED_PATH = Path(path)


# ── Building inventory ─────

def get_building_ids() -> list[str]:
    """Get sorted list of available building IDs (excluding blacklisted)."""
    files = sorted(PROCESSED_PATH.glob("*.parquet"))
    ids = [f.stem for f in files if f.stem not in EXCLUDE_IDS]
    return ids


def get_building_files_list() -> list[Path]:
    """Get sorted list of building parquet file paths."""
    files = sorted(PROCESSED_PATH.glob("*.parquet"))
    return [f for f in files if f.stem not in EXCLUDE_IDS]


# ── Data loading per client 

def load_client_data(
    partition_id: int,
    building_ids: list[str],
    horizon: int = 1,
    train_ratio: float = 0.60,
    adapt_ratio: float = 0.15,
    val_ratio: float = 0.10,
) -> dict:
    """Load and prepare data for one FL client.

    Each client = one building. Data is loaded from the
    pre-processed parquet file and split chronologically.

    Split: 60% train / 15% adapt / 10% val / 15% test
        - train: FL rounds only (different segment per round in streaming)
        - adapt: fine-tuning ONLY — model never sees this during FL
        - val:   early stopping during both FL and fine-tuning
        - test:  final evaluation — never touched during training

    Args:
        partition_id: Client index (0 to K-1).
        building_ids: List of building IDs participating in FL.
        horizon: Forecast horizon (1, 6, 24, or 168).
        train_ratio: Fraction for FL training (default 0.60).
        adapt_ratio: Fraction for fine-tuning (default 0.15).
        val_ratio: Fraction for validation (default 0.10).

    Returns:
        Dict with X_train, y_train, X_adapt, y_adapt, X_val, y_val,
        X_test, y_test (all numpy float32), plus metadata.
    """
    # Auto-discover buildings if list is empty (Ray workers)
    if not building_ids:
        building_ids = get_building_ids()

    if partition_id >= len(building_ids):
        raise ValueError(
            f"partition_id {partition_id} >= num buildings {len(building_ids)}"
        )

    building_id = building_ids[partition_id]
    filepath = PROCESSED_PATH / f"{building_id}.parquet"

    if not filepath.exists():
        raise FileNotFoundError(f"Building file not found: {filepath}")

    df = pd.read_parquet(filepath)

    # Target column
    target_col = f"target_kwh_t+{horizon}"
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found")

    # Separate features and target
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if c not in target_cols]

    X = df[feature_cols]
    y = df[target_col]

    # Drop rows with NaN in features or target
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    n = len(X)
    if n < 200:
        raise ValueError(f"Building {building_id}: insufficient data ({n} rows)")

    # Chronological split: 60% train / 15% adapt / 10% val / 15% test
    n_train = int(n * train_ratio)
    n_adapt = int(n * adapt_ratio)
    n_val = int(n * val_ratio)

    X_train = X.iloc[:n_train].values.astype(np.float32)
    y_train = y.iloc[:n_train].values.astype(np.float32)

    X_adapt = X.iloc[n_train:n_train + n_adapt].values.astype(np.float32)
    y_adapt = y.iloc[n_train:n_train + n_adapt].values.astype(np.float32)

    X_val = X.iloc[n_train + n_adapt:n_train + n_adapt + n_val].values.astype(np.float32)
    y_val = y.iloc[n_train + n_adapt:n_train + n_adapt + n_val].values.astype(np.float32)

    X_test = X.iloc[n_train + n_adapt + n_val:].values.astype(np.float32)
    y_test = y.iloc[n_train + n_adapt + n_val:].values.astype(np.float32)

    # Normalize using training statistics only (privacy-preserving)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_adapt = (X_adapt - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_adapt": X_adapt,
        "y_adapt": y_adapt,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "building_id": building_id,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_adapt": len(X_adapt),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "scaler": {"mean": mean, "std": std},
    }


# ── Streaming FL data loading ───────────────────────────────
# Each FL round gets a different time segment, simulating
# real deployment where new data arrives periodically.
# This extends the thesis approach (Badgujar 2025) to scale.

def prepare_streaming_rounds(
    partition_id: int,
    building_ids: list[str],
    horizon: int = 1,
    num_rounds: int = 30,
    train_pool_ratio: float = 0.60,
    test_ratio: float = 0.15,
) -> dict:
    """Pre-compute all round data segments for one building.

    Streaming rounds use ONLY the first 60% of data (the train pool).
    Rows between the train pool and test set (adapt 15% + val 10%)
    are untouched here — reserved for load_client_data() fine-tuning.

    Timeline (consistent with load_client_data 60/15/10/15 split):
        [== streaming pool 60% ==][adapt 15%][val 10%][== test 15% ==]
          rows [0 : n*0.60]        ← untouched →       [n*0.85 : end]
          ↓ split into 30 rounds   by streaming         kept as test

    For validation, each segment uses its last 15% as val.

    Args:
        partition_id: Client index.
        building_ids: List of building IDs.
        horizon: Forecast horizon.
        num_rounds: Number of FL rounds (= number of segments).
        train_pool_ratio: Fraction of data used for streaming rounds (0.60).
        test_ratio: Fraction reserved for testing (0.15).

    Returns:
        Dict with:
            - rounds: list of dicts, each with X_train, y_train, X_val, y_val
            - test: dict with X_test, y_test (shared across rounds)
            - metadata: building_id, n_features, etc.
    """
    if not building_ids:
        building_ids = get_building_ids()

    if partition_id >= len(building_ids):
        raise ValueError(
            f"partition_id {partition_id} >= num buildings {len(building_ids)}"
        )

    building_id = building_ids[partition_id]
    filepath = PROCESSED_PATH / f"{building_id}.parquet"

    if not filepath.exists():
        raise FileNotFoundError(f"Building file not found: {filepath}")

    df = pd.read_parquet(filepath)

    # Target column
    target_col = f"target_kwh_t+{horizon}"
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found")

    # Separate features and target
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if c not in target_cols]

    X = df[feature_cols]
    y = df[target_col]

    # Drop rows with NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    n = len(X)
    if n < 500:
        raise ValueError(f"Building {building_id}: insufficient data ({n} rows)")

    # Streaming pool: first 60% only (adapt + val sit between pool and test)
    print(f"STREAMING: train_pool_ratio={train_pool_ratio}, "
          f"test_ratio={test_ratio}, n={n}", flush=True)
    train_pool_end = int(n * train_pool_ratio)
    # Test set: final 15%
    test_start = n - int(n * test_ratio)

    X_test = X.iloc[test_start:].values.astype(np.float32)
    y_test = y.iloc[test_start:].values.astype(np.float32)

    # Split streaming pool into num_rounds segments
    segment_size = train_pool_end // num_rounds
    if segment_size < 100:
        # If segments too small, use fewer rounds
        num_rounds = max(1, train_pool_end // 200)
        segment_size = train_pool_end // num_rounds

    # Compute normalization from streaming pool only (privacy-preserving)
    X_train_all = X.iloc[:train_pool_end].values.astype(np.float32)
    mean = np.mean(X_train_all, axis=0)
    std = np.std(X_train_all, axis=0)
    std[std < 1e-8] = 1.0

    # Normalize test set
    X_test_norm = (X_test - mean) / std

    # Create per-round segments from [0 : train_pool_end] only
    rounds_data = []
    for r in range(num_rounds):
        seg_start = r * segment_size
        seg_end = min(seg_start + segment_size, train_pool_end)

        X_seg = X.iloc[seg_start:seg_end].values.astype(np.float32)
        y_seg = y.iloc[seg_start:seg_end].values.astype(np.float32)

        # Normalize segment using global statistics
        X_seg_norm = (X_seg - mean) / std

        # Split segment into train (85%) and val (15%)
        n_seg = len(X_seg_norm)
        n_val = max(int(n_seg * 0.15), 20)
        n_train = n_seg - n_val

        rounds_data.append({
            "X_train": X_seg_norm[:n_train],
            "y_train": y_seg[:n_train],
            "X_val": X_seg_norm[n_train:],
            "y_val": y_seg[n_train:],
            "n_train": n_train,
            "n_val": n_val,
            "seg_start": seg_start,
            "seg_end": seg_end,
        })

    return {
        "rounds": rounds_data,
        "test": {
            "X_test": X_test_norm,
            "y_test": y_test,
        },
        "building_id": building_id,
        "n_features": X_test_norm.shape[1],
        "n_test": len(X_test_norm),
        "num_rounds": num_rounds,
        "segment_size": segment_size,
        "scaler": {"mean": mean, "std": std},
    }


def load_client_data_streaming(
    partition_id: int,
    building_ids: list[str],
    horizon: int = 1,
    round_idx: int = 0,
    num_rounds: int = 30,
    test_ratio: float = 0.15,
) -> dict:
    """Load data for one FL client for a specific round.

    This is the streaming equivalent of load_client_data().
    Each round_idx gets a different time segment.

    Args:
        partition_id: Client index.
        building_ids: List of building IDs.
        horizon: Forecast horizon.
        round_idx: Current FL round (0-indexed).
        num_rounds: Total FL rounds.
        test_ratio: Fraction for testing.

    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test
        for this specific round.
    """
    # Pre-compute all rounds (cached per building in practice)
    all_rounds = prepare_streaming_rounds(
        partition_id, building_ids, horizon, num_rounds,
        train_pool_ratio=0.60,
        test_ratio=test_ratio,
    )

    # Clamp round_idx to available rounds
    actual_rounds = len(all_rounds["rounds"])
    r = min(round_idx, actual_rounds - 1)

    round_data = all_rounds["rounds"][r]

    return {
        "X_train": round_data["X_train"],
        "y_train": round_data["y_train"],
        "X_val": round_data["X_val"],
        "y_val": round_data["y_val"],
        "X_test": all_rounds["test"]["X_test"],
        "y_test": all_rounds["test"]["y_test"],
        "building_id": all_rounds["building_id"],
        "n_features": all_rounds["n_features"],
        "n_train": round_data["n_train"],
        "n_val": round_data["n_val"],
        "n_test": all_rounds["n_test"],
        "round_idx": r,
        "segment_size": all_rounds["segment_size"],
        "scaler": all_rounds["scaler"],
    }


# ── Model building and compilation 

def build_model(
    n_features: int,
    hidden_layers: list[int] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build and compile the MLP model.

    Architecture matches Scenario 3 (local MLP) exactly,
    ensuring fair comparison between local and FL results.

    Args:
        n_features: Number of input features (51).
        hidden_layers: Hidden layer sizes. Default: [64, 32].
        dropout_rate: Dropout rate.
        learning_rate: Adam learning rate.

    Returns:
        Compiled Keras model.
    """
    if hidden_layers is None:
        hidden_layers = [64, 32]

    model = tf.keras.Sequential(name="dh_mlp")
    model.add(tf.keras.layers.InputLayer(input_shape=(n_features,)))

    for i, units in enumerate(hidden_layers):
        model.add(tf.keras.layers.Dense(
            units,
            activation="relu",
            kernel_initializer="he_normal",
            name=f"dense_{i}",
        ))
        if dropout_rate > 0 and i < len(hidden_layers) - 1:
            model.add(tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i}"))

    model.add(tf.keras.layers.Dense(1, activation="linear", name="output"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


# ── Training and evaluation 

def train_client(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 5,
    batch_size: int = 64,
    early_stopping: bool = True,
    patience: int = 2,
) -> dict:
    """Train model on local client data.

    Args:
        model: Keras model with weights set from global model.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        epochs: Number of local epochs per FL round.
        batch_size: Batch size for training.
        early_stopping: Whether to use early stopping.
        patience: Epochs to wait before stopping (keep small for FL).

    Returns:
        Dict with training history and metrics.
    """
    callbacks = []
    if early_stopping and len(X_val) > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    actual_epochs = len(history.history["loss"])

    return {
        "train_loss": history.history["loss"][-1],
        "val_loss": history.history["val_loss"][-1],
        "train_mae": history.history["mae"][-1],
        "val_mae": history.history["val_mae"][-1],
        "epochs_trained": actual_epochs,
    }


def evaluate_client(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate model on local client test data.

    Args:
        model: Keras model.
        X_test, y_test: Test data.

    Returns:
        Dict with loss, mae, rmse, r2.
    """
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred = np.clip(y_pred, 0, None)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "loss": float(loss),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_test": len(y_test),
    }


def get_latest_global_weights():
    """Return the most recent global model weights saved during FL."""
    return _latest_global_weights


# ── Centralized evaluation (server-side) 

def get_evaluate_fn(
    building_ids: list[str],
    horizon: int,
    n_features: int,
):
    """Return a centralized evaluation function for the server.

    Evaluates the global model on a combined test set from
    a sample of buildings. Used for tracking FL convergence.
    """
    # Pre-load test data from a sample of buildings (max 50)
    sample_ids = building_ids[:min(50, len(building_ids))]
    test_data_list = []

    for i, bid in enumerate(sample_ids):
        try:
            data = load_client_data(i, sample_ids, horizon=horizon)
            test_data_list.append((data["X_test"], data["y_test"]))
        except Exception:
            continue

    if not test_data_list:
        return None

    X_test_all = np.concatenate([d[0] for d in test_data_list])
    y_test_all = np.concatenate([d[1] for d in test_data_list])

    def evaluate_fn(server_round, parameters, config):
        """Evaluate global model on centralized test set."""
        global _latest_global_weights

        model = build_model(n_features)
        model.set_weights(parameters)

        # Save the latest global weights for post-FL evaluation
        _latest_global_weights = parameters

        loss, mae = model.evaluate(X_test_all, y_test_all, verbose=0)

        y_pred = model.predict(X_test_all, verbose=0).flatten()
        y_pred = np.clip(y_pred, 0, None)
        rmse = np.sqrt(np.mean((y_test_all - y_pred) ** 2))

        tf.keras.backend.clear_session()
        del model

        return float(loss), {
            "mae": float(mae),
            "rmse": float(rmse),
            "round": server_round,
        }

    return evaluate_fn
