"""
DAS-FL Project — client_app.py

Flower ClientApp: each building is an independent FL client.

The client:
1. Receives global model weights from server
2. Loads its own building's parquet data
3. Trains locally for E epochs
4. Returns updated weights + metrics

Supports:
    - Scenario 4: FedAvg (standard local training)
    - Scenario 5: FedProx (adds proximal term)
    - Scenario 6: Personalised FL (extra fine-tuning after FL)

Usage:
    This file is called by Flower's simulation engine via
    `flwr run .` or via our custom runner script.

"""

import numpy as np
import os
import warnings
import gc
import pickle
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import csv
import flwr as fl
from flwr.client import NumPyClient, ClientApp

from dasfl.task import (
    load_client_data,
    load_client_data_streaming,
    build_model,
    train_client,
    evaluate_client,
    get_building_ids,
)


# ── Global state ────────────────────────────────────────────
# These are set by the runner script before FL starts
_building_ids: list[str] = []
_horizon: int = 1
_n_features: int = 51
_fedprox_mu: float = 0.0  # 0.0 = FedAvg, >0 = FedProx
_streaming: bool = False
_num_rounds: int = 30


def configure_client(
    building_ids: list[str],
    horizon: int,
    n_features: int,
    fedprox_mu: float = 0.0,
    streaming: bool = False,
    num_rounds: int = 30,
):
    """Set global client configuration before FL run."""
    global _building_ids, _horizon, _n_features, _fedprox_mu
    global _streaming, _num_rounds
    _building_ids = building_ids
    _horizon = horizon
    _n_features = n_features
    _fedprox_mu = fedprox_mu
    _streaming = streaming
    _num_rounds = num_rounds




# ── Per-round metric logging (matches thesis pattern) ───────

_LOG_DIR = None

def _get_log_dir():
    """Get or create the FL metrics log directory."""
    global _LOG_DIR
    if _LOG_DIR is None:
        from pathlib import Path
        _LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "fl_metrics"
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def log_round_metrics(building_id, round_num, model_type,
                      loss, mae, rmse, r2):
    """Log metrics for one building, one round, one model type.
    
    Creates CSV: logs/fl_metrics/client_eval_{building_id}.csv
    Columns: building_id, round, model_type, loss, mae, rmse, r2
    """
    log_dir = _get_log_dir()
    filepath = log_dir / f"client_eval_{building_id}.csv"
    
    file_exists = filepath.exists() and filepath.stat().st_size > 0
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["building_id", "round", "model_type",
                           "loss", "mae", "rmse", "r2"])
        writer.writerow([building_id, round_num, model_type,
                        f"{loss:.6f}", f"{mae:.4f}",
                        f"{rmse:.4f}", f"{r2:.4f}"])


# ── FedProx proximal loss ───────────────────────────────────

class FedProxLoss(tf.keras.losses.Loss):
    """MSE loss with proximal term for FedProx.

    L_local = MSE(y, y_hat) + (mu/2) * ||w - w_global||^2

    The proximal term penalises deviation from the global model,
    preventing client drift under heterogeneous data.
    """

    def __init__(self, mu: float = 0.01, global_weights=None, model=None, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.global_weights = global_weights
        self.model = model

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        if self.mu > 0 and self.global_weights is not None and self.model is not None:
            prox_term = 0.0
            for w_local, w_global in zip(
                self.model.trainable_weights, self.global_weights
            ):
                prox_term += tf.reduce_sum(tf.square(w_local - w_global))
            return mse + (self.mu / 2.0) * prox_term

        return mse


# ── Flower NumPyClient ──────────────────────────────────────

class DHClient(NumPyClient):
    """District Heating FL client for one building.

    Each instance handles one building's data for one horizon.
    """

    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.data = None
        self.model = None
        self.local_weights = None

    def _load_round_data(self, round_idx: int = 0):
        """Load data for a specific round (streaming) or all data (static)."""
        bids = _building_ids if _building_ids else get_building_ids()
        horizon = _horizon if _horizon else 1

        if _streaming:
            self.data = load_client_data_streaming(
                self.partition_id, bids,
                horizon=horizon,
                round_idx=round_idx,
                num_rounds=_num_rounds,
            )
        else:
            self.data = load_client_data(
                self.partition_id, bids,
                horizon=horizon,
            )

    def _ensure_data(self, round_idx: int = 0):
        """Lazy-load data. In streaming mode, reload each round."""
        if _streaming:
            # Always reload in streaming mode (different data per round)
            self._load_round_data(round_idx)
        elif self.data is None:
            self._load_round_data(round_idx)

    def get_parameters(self, config):
        """Return current model weights."""
        self._ensure_data()
        if self.model is None:
            self.model = build_model(self.data["n_features"])
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train on local data.

        Args:
            parameters: Global model weights from server.
            config: Training config (epochs, batch_size, etc.).

        Returns:
            Updated weights, number of training samples, metrics dict.
        """
        # Get round index from server config
        server_round = config.get("server_round", 1)
        round_idx = max(0, server_round - 1)

        # Load data for this round (streaming: different data each round)
        self._ensure_data(round_idx)

        # Build or reuse model
        if self.model is None:
            self.model = build_model(self.data["n_features"])

        # Set global weights
        self.model.set_weights(parameters)

        # Get config
        epochs = config.get("local_epochs", 5)
        batch_size = config.get("batch_size", 64)
        lr = config.get("lr", 0.001)

        # FedProx: recompile with proximal loss
        if _fedprox_mu > 0:
            global_weights = [w.numpy() for w in self.model.trainable_weights]
            prox_loss = FedProxLoss(
                mu=_fedprox_mu,
                global_weights=[tf.constant(w) for w in global_weights],
                model=self.model,
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=prox_loss,
                metrics=["mae"],
            )
        else:
            # Standard FedAvg — recompile with correct learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss="mse",
                metrics=["mae"],
            )

        # Apply decayed learning rate from server config
        tf.keras.backend.set_value(
            self.model.optimizer.learning_rate, lr)

        # Train locally with early stopping
        # More max epochs (10) but early stopping (patience=2)
        # prevents overfitting while allowing sufficient learning
        metrics = train_client(
            self.model,
            self.data["X_train"],
            self.data["y_train"],
            self.data["X_val"],
            self.data["y_val"],
            epochs=epochs,
            batch_size=batch_size,
            early_stopping=True,
            patience=2,
        )

        # Save local weights BEFORE returning to server (thesis pattern)
        # This allows evaluate() to compare global vs local
        self.local_weights = [w.copy() for w in self.model.get_weights()]

        # Also save to disk (Ray creates new instances between fit/evaluate)
        import pickle
        weights_dir = Path(__file__).resolve().parent.parent / "logs" / "fl_weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        with open(weights_dir / f"local_weights_{self.partition_id}.pkl", "wb") as wf:
            pickle.dump(self.local_weights, wf)

        return (
            self.model.get_weights(),
            self.data["n_train"],
            {
                "building_id": self.data["building_id"],
                "train_mae": float(metrics["train_mae"]),
                "val_mae": float(metrics["val_mae"]),
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate BOTH global and local models on test data.

        Matches thesis pattern (Badgujar 2025):
        1. Evaluate global model (weights from server)
        2. Evaluate local model (weights from fit, before aggregation)
        3. Log both to CSV for convergence analysis

        Returns global model metrics to server.
        """
        server_round = config.get("server_round", 1)
        round_idx = max(0, server_round - 1)
        self._ensure_data(round_idx)

        if self.model is None:
            self.model = build_model(self.data["n_features"])

        bid = self.data["building_id"]

        # --- 1. Evaluate GLOBAL model (from server) ---
        self.model.set_weights(parameters)
        metrics_global = evaluate_client(
            self.model,
            self.data["X_test"],
            self.data["y_test"],
        )

        # Log global metrics
        log_round_metrics(
            bid, server_round, "global",
            metrics_global["loss"], metrics_global["mae"],
            metrics_global["rmse"], metrics_global["r2"],
        )

        # --- 2. Evaluate LOCAL model (from fit, before aggregation) ---
        # Try to load from disk if not in memory (Ray stateless actors)
        if self.local_weights is None:
            weights_path = Path(__file__).resolve().parent.parent / "logs" / "fl_weights" / f"local_weights_{self.partition_id}.pkl"
            if weights_path.exists():
                with open(weights_path, "rb") as wf:
                    self.local_weights = pickle.load(wf)

        if self.local_weights is not None:
            self.model.set_weights(self.local_weights)
            metrics_local = evaluate_client(
                self.model,
                self.data["X_test"],
                self.data["y_test"],
            )

            # Log local metrics
            log_round_metrics(
                bid, server_round, "local",
                metrics_local["loss"], metrics_local["mae"],
                metrics_local["rmse"], metrics_local["r2"],
            )

            local_mae = float(metrics_local["mae"])
        else:
            local_mae = float("nan")

        # Return global model metrics to server
        return (
            float(metrics_global["loss"]),
            self.data["n_test"],
            {
                "building_id": bid,
                "mae": float(metrics_global["mae"]),
                "rmse": float(metrics_global["rmse"]),
                "r2": float(metrics_global["r2"]),
                "local_mae": local_mae,
            },
        )


# ── Client factory ──────────────────────────────────────────

def client_fn(context) -> DHClient:
    """Create a client for a given partition.

    Called by Flower's simulation engine. The partition_id
    maps to a building in the building_ids list.
    """
    partition_id = context.node_config.get("partition-id", 0)
    return DHClient(partition_id=partition_id)


# For Flower's ClientApp pattern (used by `flwr run`)
app = ClientApp(client_fn=client_fn)
