"""
DAS-FL Project — server_app.py
Flower ServerApp: coordinates federated training.

The server:
1. Initializes global model
2. Distributes weights to clients each round
3. Aggregates client updates (FedAvg or FedProx)
4. Evaluates global model on centralized test set
5. Logs metrics per round

Supports:
    - FedAvg: weighted averaging (McMahan et al., 2017)
    - FedProx: same aggregation, proximal term on client side

Usage:
    Called via our custom runner script (pipeline/run_fl.py)
    or via `flwr run .`
"""

import numpy as np
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import flwr as fl
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg, FedAdam, FedYogi, FedAdagrad

from dasfl.task import (
    build_model,
    get_evaluate_fn,
    get_building_ids,
)


# ── Global server state ─────────────────────────────────────
_building_ids: list[str] = []
_horizon: int = 1
_n_features: int = 51
_num_rounds: int = 50
_fraction_fit: float = 1.0
_fraction_evaluate: float = 0.5
_min_fit_clients: int = 2
_local_epochs: int = 5
_batch_size: int = 64
_learning_rate: float = 0.001
_strategy_name: str = "fedavg"


def configure_server(
    building_ids: list[str],
    horizon: int,
    n_features: int,
    num_rounds: int = 50,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.5,
    local_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
):
    """Set global server configuration before FL run."""
    global _building_ids, _horizon, _n_features, _num_rounds
    global _fraction_fit, _fraction_evaluate, _min_fit_clients
    global _local_epochs, _batch_size, _learning_rate

    _building_ids = building_ids
    _horizon = horizon
    _n_features = n_features
    _num_rounds = num_rounds
    _fraction_fit = fraction_fit
    _fraction_evaluate = fraction_evaluate
    _min_fit_clients = min(2, len(building_ids))
    _local_epochs = local_epochs
    _batch_size = batch_size
    _learning_rate = learning_rate


def set_strategy_name(name: str):
    """Set which FL strategy to use."""
    global _strategy_name
    _strategy_name = name


def _on_fit_config_fn(server_round: int) -> dict:
    """Return training config sent to clients each round.

    Implements round-based learning rate decay to prevent
    instability in later FL rounds (Badgujar 2025 thesis pattern).
    """
    initial_lr = 0.001
    decay_rate = 0.00001
    lr = initial_lr / (1.0 + decay_rate * server_round)
    return {
        "lr": lr,
        "local_epochs": _local_epochs,
        "batch_size": _batch_size,
        "server_round": server_round,
    }


def _on_evaluate_config_fn(server_round: int) -> dict:
    """Return evaluation config sent to clients each round."""
    return {
        "server_round": server_round,
    }


def create_strategy(
    n_features: int,
    building_ids: list[str],
    horizon: int,
):
    """Create FL strategy with centralized evaluation.

    Supports FedAvg (McMahan 2017) and FedAdam (Reddi 2021).
    FedAdam uses adaptive server-side optimization which handles
    heterogeneous client updates better than FedAvg.
    """
    # Initial model parameters
    initial_model = build_model(n_features)
    initial_parameters = initial_model.get_weights()
    tf.keras.backend.clear_session()
    del initial_model

    # Centralized evaluation function
    evaluate_fn = get_evaluate_fn(building_ids, horizon, n_features)

    params = fl.common.ndarrays_to_parameters(initial_parameters)

    if _strategy_name == "fedadam":
        strategy = FedAdam(
            fraction_fit=_fraction_fit,
            fraction_evaluate=_fraction_evaluate,
            min_fit_clients=_min_fit_clients,
            min_evaluate_clients=_min_fit_clients,
            min_available_clients=len(building_ids),
            on_fit_config_fn=_on_fit_config_fn,
            on_evaluate_config_fn=_on_evaluate_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=params,
            eta=1e-2,        # server learning rate (Reddi 2021: tune 0.01-1.0)
            eta_l=1e-3,      # client learning rate
            beta_1=0.9,      # momentum
            beta_2=0.99,     # second moment
            tau=1e-3,        # adaptivity (Reddi 2021 Fig 3: 1e-3 works broadly)
        )
    elif _strategy_name == "fedyogi":
        strategy = FedYogi(
            fraction_fit=_fraction_fit,
            fraction_evaluate=_fraction_evaluate,
            min_fit_clients=_min_fit_clients,
            min_evaluate_clients=_min_fit_clients,
            min_available_clients=len(building_ids),
            on_fit_config_fn=_on_fit_config_fn,
            on_evaluate_config_fn=_on_evaluate_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=params,
            eta=1e-2,        # server LR (same as tuned FedAdam)
            eta_l=1e-3,      # client LR
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-3,
        )
    elif _strategy_name == "fedadagrad":
        strategy = FedAdagrad(
            fraction_fit=_fraction_fit,
            fraction_evaluate=_fraction_evaluate,
            min_fit_clients=_min_fit_clients,
            min_evaluate_clients=_min_fit_clients,
            min_available_clients=len(building_ids),
            on_fit_config_fn=_on_fit_config_fn,
            on_evaluate_config_fn=_on_evaluate_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=params,
            eta=1e-2,        # server LR
            eta_l=1e-3,      # client LR
            tau=1e-3,
        )
    else:
        # Default: FedAvg
        strategy = FedAvg(
            fraction_fit=_fraction_fit,
            fraction_evaluate=_fraction_evaluate,
            min_fit_clients=_min_fit_clients,
            min_evaluate_clients=_min_fit_clients,
            min_available_clients=len(building_ids),
            on_fit_config_fn=_on_fit_config_fn,
            on_evaluate_config_fn=_on_evaluate_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=params,
        )

    return strategy


# ── Server factory for `flwr run` ──────────────────────────

def server_fn(context) -> fl.server.ServerAppComponents:
    """Create server components."""
    strategy = create_strategy(_n_features, _building_ids, _horizon)
    config = ServerConfig(num_rounds=_num_rounds)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
