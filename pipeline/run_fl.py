"""
============================================================
DAS-FL Project — Run Federated Learning Experiments
============================================================
Orchestrates FL training for Scenarios 4, 5, 6 and the
scalability experiment.

Usage:
    # Quick test: 20 buildings, 5 rounds, horizon t+1
    python pipeline/run_fl.py --scenario fedavg --n 20 --rounds 5 --horizon 1

    # Full FedAvg: 100 buildings, 50 rounds, all horizons
    python pipeline/run_fl.py --scenario fedavg --n 100

    # FedProx: 100 buildings
    python pipeline/run_fl.py --scenario fedprox --n 100

    # Personalised FL: 100 buildings
    python pipeline/run_fl.py --scenario personalised --n 100

    # Scalability experiment (runs FedAvg at multiple scales)
    python pipeline/run_fl.py --scenario scalability

    # Run all scenarios sequentially
    python pipeline/run_fl.py --scenario all --n 100
============================================================
"""

import random
import numpy as np
import pandas as pd
import os
import sys
import gc
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# ── Reproducibility seeds ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.random.set_seed(SEED)
tf.get_logger().setLevel("ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
from flwr.simulation import run_simulation

from dasfl.task import (
    get_building_ids,
    build_model,
    load_client_data,
    evaluate_client,
    get_evaluate_fn,
    get_latest_global_weights,
)
from dasfl.client_app import (
    configure_client,
    client_fn,
    DHClient,
)
from dasfl.server_app import (
    configure_server,
    create_strategy,
    set_strategy_name,
)

import yaml

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_HORIZONS = config["model"]["baseline"]["eval_horizons"]


# ── GPU check ──────────────────────────────────────────────

def _log_gpu():
    """Log GPU availability at start of each experiment."""
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    for g in gpus:
        print(f"  {g}")


# ── Building ID persistence ────────────────────────────────

_BUILDING_IDS_FILE = None  # set via --building-ids-file


def _get_or_create_building_ids(n_buildings: int) -> list[str]:
    """Load or create deterministic building IDs.

    If --building-ids-file was provided, loads IDs from that file
    (ignoring n_buildings). Otherwise ensures the SAME building IDs
    are used across ALL scenarios by saving/loading from
    logs/fl_building_ids_{n}.txt.
    """
    if _BUILDING_IDS_FILE is not None:
        with open(_BUILDING_IDS_FILE) as f:
            ids = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(ids)} building IDs from {_BUILDING_IDS_FILE}")
        return ids

    id_file = LOG_DIR / f"fl_building_ids_{n_buildings}.txt"
    if id_file.exists():
        with open(id_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        if len(ids) == n_buildings:
            print(f"  Loaded {len(ids)} building IDs from {id_file.name}")
            return ids

    all_ids = get_building_ids()
    if n_buildings >= len(all_ids):
        building_ids = all_ids
    else:
        indices = np.linspace(0, len(all_ids) - 1, n_buildings, dtype=int)
        building_ids = [all_ids[i] for i in indices]

    with open(id_file, "w") as f:
        for bid in building_ids:
            f.write(f"{bid}\n")
    print(f"  Saved {len(building_ids)} building IDs to {id_file.name}")

    return building_ids


# ── FL Experiment Runner ────────────────────────────────────

def _create_client(partition_id: int) -> DHClient:
    """Create client that reads config from file (Ray-compatible)."""
    import json
    config_path = PROJECT_ROOT / "logs" / "_fl_runtime_config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        configure_client(
            cfg["building_ids"], cfg["horizon"],
            cfg["n_features"], cfg.get("fedprox_mu", 0.0)
        )
    return DHClient(partition_id=partition_id)


def run_fl_experiment(
    scenario: str,
    n_buildings: int,
    num_rounds: int = 50,
    horizons: list[int] = None,
    fedprox_mu: float = 0.01,
    local_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    streaming: bool = True,
    pred_suffix: str = "",
) -> pd.DataFrame:
    """Run one FL experiment (one scenario, potentially multiple horizons).

    Args:
        scenario: 'fedavg', 'fedprox', or 'personalised'
        n_buildings: Number of FL clients
        num_rounds: Number of FL communication rounds
        horizons: List of forecast horizons to evaluate
        fedprox_mu: Proximal term strength (0 for FedAvg)
        local_epochs: Local epochs per round
        batch_size: Training batch size
        learning_rate: Client learning rate

    Returns:
        DataFrame with per-building, per-horizon results.
    """
    if horizons is None:
        horizons = EVAL_HORIZONS

    building_ids = _get_or_create_building_ids(n_buildings)
    n_clients = len(building_ids)

    # Determine n_features from first building
    sample_data = load_client_data(0, building_ids, horizon=horizons[0])
    n_features = sample_data["n_features"]
    del sample_data

    mu = fedprox_mu if scenario == "fedprox" else 0.0

    all_results = []

    for horizon in horizons:
        print(f"\n  --- Horizon t+{horizon} ---")

        # Configure client and server
        configure_client(
            building_ids, horizon, n_features,
            fedprox_mu=mu, streaming=streaming, num_rounds=num_rounds,
        )
        configure_server(
            building_ids, horizon, n_features,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Set strategy based on scenario
        strategy_map = {
            "fedadam": "fedadam",
            "fedyogi": "fedyogi",
            "fedadagrad": "fedadagrad",
        }
        set_strategy_name(strategy_map.get(scenario, "fedavg"))

        # Create strategy
        strategy = create_strategy(n_features, building_ids, horizon)

        # Collect round-level metrics via a custom callback
        round_metrics = []

        class MetricStrategy(fl.server.strategy.FedAvg):
            """Wrapper to capture per-round metrics."""

            def __init__(self, base_strategy):
                # Copy attributes from base strategy
                self.__dict__.update(base_strategy.__dict__)

            def aggregate_evaluate(self, server_round, results, failures):
                """Capture evaluation metrics after each round."""
                aggregated = super().aggregate_evaluate(server_round, results, failures)

                if results:
                    # Collect per-client metrics
                    for _, eval_res in results:
                        metrics = eval_res.metrics if hasattr(eval_res, 'metrics') else {}
                        round_metrics.append({
                            "round": server_round,
                            "building_id": metrics.get("building_id", "unknown"),
                            "mae": metrics.get("mae", np.nan),
                            "rmse": metrics.get("rmse", np.nan),
                            "r2": metrics.get("r2", np.nan),
                        })

                return aggregated

        # Run FL simulation
        print(f"  Starting FL: {n_clients} clients, {num_rounds} rounds...")
        start = time.time()

        # Write config for Ray workers (they can't see globals)
        import json
        _fl_config = {
            "building_ids": building_ids,
            "horizon": horizon,
            "n_features": n_features,
            "fedprox_mu": mu,
        }
        _fl_config_path = PROJECT_ROOT / "logs" / "_fl_runtime_config.json"
        with open(_fl_config_path, "w") as f:
            json.dump(_fl_config, f)

        # Use Flower's simulation
        history = fl.simulation.start_simulation(
            client_fn=lambda cid: _create_client(int(cid)),
            num_clients=n_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        elapsed = time.time() - start
        print(f"  FL completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Get final global weights saved during FL evaluate_fn
        final_weights = get_latest_global_weights()
        if final_weights is None:
            print("  WARNING: No global weights captured. Results may be inaccurate.")

        print(f"  Evaluating final model on {n_clients} buildings...")

        for i, bid in enumerate(building_ids):
            try:
                data = load_client_data(i, building_ids, horizon=horizon)

                eval_model = build_model(n_features)
                if final_weights is not None:
                    eval_model.set_weights(final_weights)

                metrics = evaluate_client(
                    eval_model,
                    data["X_test"],
                    data["y_test"],
                )

                if horizon == 1:
                    pred_dir = LOG_DIR / "predictions" / pred_suffix if pred_suffix else LOG_DIR / "predictions"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    y_pred = eval_model.predict(data["X_test"], verbose=0).flatten()
                    np.savez(pred_dir / f"{scenario}_{bid}_h{horizon}.npz",
                             y_actual=data["y_test"],
                             y_pred=y_pred)

                all_results.append({
                    "scenario": scenario,
                    "n_buildings": n_clients,
                    "horizon": horizon,
                    "building_id": bid,
                    "status": "success",
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "n_test": metrics["n_test"],
                    "num_rounds": num_rounds,
                })

                del eval_model
                gc.collect()

            except Exception as e:
                all_results.append({
                    "scenario": scenario,
                    "n_buildings": n_clients,
                    "horizon": horizon,
                    "building_id": bid,
                    "status": "error",
                    "error": str(e),
                })

        # Save convergence history
        if history.losses_centralized:
            conv_df = pd.DataFrame({
                "round": [r for r, _ in history.losses_centralized],
                "loss": [l for _, l in history.losses_centralized],
            })
            conv_path = LOG_DIR / f"fl_convergence_{scenario}_h{horizon}_n{n_clients}.csv"
            conv_df.to_csv(conv_path, index=False)
            print(f"  Convergence saved: {conv_path.name}")

        tf.keras.backend.clear_session()
        gc.collect()

    return pd.DataFrame(all_results)


# ── Personalised FL ─────────────────────────────────────────

def run_personalised_fl(
    n_buildings: int,
    num_rounds: int = 50,
    fine_tune_epochs: int = 5,
    horizons: list[int] = None,
    base_strategy: str = "fedadam",
    fedprox_mu: float = 0.0,
    fine_tune_lr: float = 0.0001,
    streaming: bool = True,
    pred_suffix: str = "",
) -> pd.DataFrame:
    """Run Personalised FL: base strategy + local fine-tuning.

    Step 1: Run base strategy (FedAdam/FedAvg/FedProx) to get global model
    Step 2: Each building fine-tunes the global model locally

    Args:
        n_buildings: Number of FL clients
        num_rounds: FL rounds for global model
        fine_tune_epochs: Local fine-tuning epochs after FL
        horizons: Forecast horizons
        base_strategy: FL strategy for Step 1 ('fedadam', 'fedavg', 'fedprox')
        fedprox_mu: Proximal term strength (only used when base_strategy='fedprox')
        fine_tune_lr: Learning rate for local fine-tuning (Step 2)
        streaming: Use streaming FL (different data segment per round)

    Returns:
        DataFrame with per-building results.
    """
    if horizons is None:
        horizons = EVAL_HORIZONS

    building_ids = _get_or_create_building_ids(n_buildings)
    n_clients = len(building_ids)

    sample_data = load_client_data(0, building_ids, horizon=horizons[0])
    n_features = sample_data["n_features"]
    del sample_data

    all_results = []

    for horizon in horizons:
        print(f"\n  --- Personalised FL: Horizon t+{horizon} ---")

        mu = fedprox_mu if base_strategy == "fedprox" else 0.0
        strategy_map = {"fedadam": "fedadam", "fedyogi": "fedyogi",
                        "fedadagrad": "fedadagrad"}
        set_strategy_name(strategy_map.get(base_strategy, "fedavg"))
        # Step 1: Run base strategy to get global model
        configure_client(building_ids, horizon, n_features, fedprox_mu=mu,
                         streaming=streaming, num_rounds=num_rounds)
        configure_server(building_ids, horizon, n_features, num_rounds=num_rounds)
        strategy = create_strategy(n_features, building_ids, horizon)

        print(f"  Step 1: {base_strategy} ({num_rounds} rounds, {n_clients} clients)...")
        start = time.time()

        history = fl.simulation.start_simulation(
            client_fn=lambda cid: DHClient(partition_id=int(cid)),
            num_clients=n_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        print(f"  FL aggregation done in {time.time() - start:.0f}s")

        # Extract global model weights
        global_model = build_model(n_features)
        final_weights = get_latest_global_weights()
        if final_weights is not None:
            global_model.set_weights(final_weights)
            print(f"  Global model weights loaded successfully", flush=True)
        else:
            print(f"  WARNING: No global weights - using random init!", flush=True)

        # Step 2: Fine-tune for each building using SEPARATE adapt set
        print(f"  Step 2: Fine-tuning {n_clients} buildings ({fine_tune_epochs} epochs, lr={fine_tune_lr})...")

        for i, bid in enumerate(building_ids):
            try:
                data = load_client_data(i, building_ids, horizon=horizon)

                # Fine-tune model — start from global weights, lower LR
                local_model = build_model(n_features, learning_rate=fine_tune_lr)
                local_model.set_weights(global_model.get_weights())

                # Fine-tune on ADAPT set (never seen during FL rounds)
                local_model.fit(
                    data["X_adapt"], data["y_adapt"],
                    validation_data=(data["X_val"], data["y_val"]),
                    epochs=10,
                    batch_size=64,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        patience=2, restore_best_weights=True)],
                    verbose=0,
                )

                # Evaluate personalised model
                metrics = evaluate_client(
                    local_model, data["X_test"], data["y_test"]
                )

                if horizon == 1:
                    pred_dir = LOG_DIR / "predictions" / pred_suffix if pred_suffix else LOG_DIR / "predictions"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    y_pred = local_model.predict(data["X_test"], verbose=0).flatten()
                    np.savez(pred_dir / f"personalised_{base_strategy}_{bid}_h{horizon}.npz",
                             y_actual=data["y_test"],
                             y_pred=y_pred)

                all_results.append({
                    "scenario": "personalised",
                    "n_buildings": n_clients,
                    "horizon": horizon,
                    "building_id": bid,
                    "status": "success",
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "n_test": metrics["n_test"],
                    "num_rounds": num_rounds,
                    "fine_tune_epochs": fine_tune_epochs,
                })

                tf.keras.backend.clear_session()
                del local_model
                gc.collect()

            except Exception as e:
                all_results.append({
                    "scenario": "personalised",
                    "n_buildings": n_clients,
                    "horizon": horizon,
                    "building_id": bid,
                    "status": "error",
                    "error": str(e),
                })

        del global_model
        tf.keras.backend.clear_session()
        gc.collect()

    return pd.DataFrame(all_results)


# ── Scalability Experiment ──────────────────────────────────

def run_scalability_experiment(
    scales: list[int] = None,
    num_rounds: int = 30,
    horizon: int = 1,
    base_strategy: str = "fedadam",
    fedprox_mu: float = 0.0,
    fine_tune_lr: float = 0.0001,
) -> dict[str, pd.DataFrame]:
    """Run FedAvg + FedAdam + Personalised FL at different client scales.

    This is Contribution #3: how does FL performance scale
    with the number of participating buildings?

    Args:
        scales: List of client counts. Default: [20, 50, 100, 200, 250, 500, 988]
        num_rounds: FL rounds per scale
        horizon: Single horizon for scalability analysis (t+1 only)
        base_strategy: Base strategy for personalised FL
        fedprox_mu: Proximal term strength for FedProx base
        fine_tune_lr: Learning rate for personalised fine-tuning

    Returns:
        Dict mapping scenario name to DataFrame with results at each scale.
    """
    if scales is None:
        max_buildings = len(get_building_ids())
        scales = [20, 50, 100, 200, 250, 500, max_buildings]
        scales = [s for s in scales if s <= max_buildings]

    results_fedavg = []
    results_fedadam = []
    results_personalised = []

    for n in scales:
        print(f"\n{'='*50}")
        print(f"SCALABILITY: K = {n} buildings, horizon t+{horizon}")
        print(f"{'='*50}")

        # FedAvg
        print(f"\n  >> FedAvg (K={n})")
        _log_gpu()
        df_avg = run_fl_experiment(
            scenario="fedavg",
            n_buildings=n,
            num_rounds=num_rounds,
            horizons=[horizon],
        )
        df_avg["scale_k"] = n
        results_fedavg.append(df_avg)

        # FedAdam
        print(f"\n  >> FedAdam (K={n})")
        _log_gpu()
        df_adam = run_fl_experiment(
            scenario="fedadam",
            n_buildings=n,
            num_rounds=num_rounds,
            horizons=[horizon],
        )
        df_adam["scale_k"] = n
        results_fedadam.append(df_adam)

        # Personalised FL
        print(f"\n  >> Personalised FL (K={n})")
        _log_gpu()
        df_pers = run_personalised_fl(
            n_buildings=n,
            num_rounds=num_rounds,
            horizons=[horizon],
            base_strategy=base_strategy,
            fedprox_mu=fedprox_mu,
            fine_tune_lr=fine_tune_lr,
        )
        df_pers["scale_k"] = n
        results_personalised.append(df_pers)

        # Print summary
        for label, df in [("FedAvg", df_avg), ("FedAdam", df_adam), ("Personalised", df_pers)]:
            df_ok = df[df["status"] == "success"]
            if len(df_ok) > 0:
                print(f"  K={n} {label}: MAE={df_ok['mae'].median():.3f}, "
                      f"R²={df_ok['r2'].median():.3f}")

    return {
        "fedavg": pd.concat(results_fedavg, ignore_index=True),
        "fedadam": pd.concat(results_fedadam, ignore_index=True),
        "personalised": pd.concat(results_personalised, ignore_index=True),
    }


# ── Local MLP (no FL) ───────────────────────────────────────

def run_local_mlp(
    n_buildings: int,
    horizons: list[int] = None,
    epochs: int = 50,
    pred_suffix: str = "",
) -> pd.DataFrame:
    """Train independent Local MLP per building (no federation).

    Matches the same building IDs as FL experiments for fair comparison.

    Args:
        n_buildings: Number of buildings to evaluate.
        horizons: Forecast horizons. Default: all 4.
        epochs: Max training epochs per building (early stopping used).

    Returns:
        DataFrame with per-building, per-horizon results.
    """
    if horizons is None:
        horizons = EVAL_HORIZONS

    building_ids = _get_or_create_building_ids(n_buildings)
    n_clients = len(building_ids)

    sample_data = load_client_data(0, building_ids, horizon=horizons[0])
    n_features = sample_data["n_features"]
    del sample_data

    all_results = []

    for horizon in horizons:
        print(f"\n  --- Local MLP: Horizon t+{horizon} ---")

        for i, bid in enumerate(building_ids):
            try:
                data = load_client_data(i, building_ids, horizon=horizon)

                model = build_model(n_features)
                model.fit(
                    data["X_train"], data["y_train"],
                    validation_data=(data["X_val"], data["y_val"]),
                    epochs=epochs,
                    batch_size=64,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        patience=5, restore_best_weights=True)],
                    verbose=0,
                )

                metrics = evaluate_client(model, data["X_test"], data["y_test"])

                if horizon == 1:
                    pred_dir = LOG_DIR / "predictions" / pred_suffix if pred_suffix else LOG_DIR / "predictions"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    y_pred = model.predict(data["X_test"], verbose=0).flatten()
                    np.savez(pred_dir / f"local_mlp_{bid}_h{horizon}.npz",
                             y_actual=data["y_test"],
                             y_pred=y_pred)

                all_results.append({
                    "scenario": "local_mlp",
                    "n_buildings": n_clients,
                    "horizon": horizon,
                    "building_id": bid,
                    "status": "success",
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "n_test": metrics["n_test"],
                })

                tf.keras.backend.clear_session()
                del model
                gc.collect()

            except Exception as e:
                all_results.append({
                    "scenario": "local_mlp",
                    "n_buildings": n_clients,
                    "horizon": horizon,
                    "building_id": bid,
                    "status": "error",
                    "error": str(e),
                })

        if i % 50 == 0 and i > 0:
            print(f"    Processed {i+1}/{n_clients} buildings...")

    return pd.DataFrame(all_results)


# ── Report ──────────────────────────────────────────────────

def generate_fl_report(df: pd.DataFrame, scenario: str, elapsed: float):
    """Print and save FL experiment report."""
    df_ok = df[df["status"] == "success"]

    print(f"\n{'='*70}")
    print(f"FL RESULTS — {scenario.upper()}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    if len(df_ok) == 0:
        print("  NO SUCCESSFUL RESULTS")
        return

    print(f"\n  Buildings: {df_ok['building_id'].nunique()}")
    print(f"  Horizons:  {sorted(df_ok['horizon'].unique())}")

    print(f"\n  {'Horizon':<10} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Buildings':>10}")
    print(f"  {'-'*46}")

    for h in sorted(df_ok["horizon"].unique()):
        hdf = df_ok[df_ok["horizon"] == h]
        print(f"  t+{h:<7} {hdf['mae'].median():>8.3f} "
              f"{hdf['rmse'].median():>8.3f} "
              f"{hdf['r2'].median():>8.3f} "
              f"{hdf['building_id'].nunique():>10}")

    print(f"{'='*70}")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run FL experiments")
    parser.add_argument("--scenario", type=str, required=True,
                        choices=["fedavg", "fedadam", "fedyogi", "fedadagrad",
                                 "fedprox", "personalised", "scalability",
                                 "local_mlp", "all"],
                        help="FL scenario to run")
    parser.add_argument("--n", type=int, default=250,
                        help="Number of buildings (clients)")
    parser.add_argument("--rounds", type=int, default=30,
                        help="Number of FL rounds")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Single horizon (default: all 4)")
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mu", type=float, default=0.01,
                        help="FedProx proximal term strength")
    parser.add_argument("--fine-tune-lr", type=float, default=0.0001,
                        help="Learning rate for personalised fine-tuning (default: 0.0001)")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Use streaming FL (different data per round)")
    parser.add_argument("--static", action="store_true",
                        help="Use static FL (same data every round)")
    parser.add_argument("--base-strategy", type=str, default="fedadam",
                        choices=["fedavg", "fedadam", "fedprox"],
                        help="Base strategy for personalised FL (default: fedadam)")
    parser.add_argument("--suffix", type=str, default="v2",
                        help="Suffix appended to output filenames (default: v2)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (for Aalborg benchmark)")
    parser.add_argument("--building-ids-file", type=str, default=None,
                        help="File with building IDs (one per line); "
                             "overrides --n and default ID discovery")
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

    if args.data_dir:
        from dasfl.task import set_data_dir
        set_data_dir(args.data_dir)

    if args.building_ids_file:
        global _BUILDING_IDS_FILE
        _BUILDING_IDS_FILE = args.building_ids_file

    horizons = [args.horizon] if args.horizon else EVAL_HORIZONS

    print("=" * 60)
    print(f"DAS-FL PROJECT — FEDERATED LEARNING EXPERIMENTS")
    print(f"Scenario: {args.scenario}")
    print(f"Buildings: {args.n}")
    print(f"Rounds: {args.rounds}")
    print(f"Horizons: {horizons}")
    print("=" * 60)

    _log_gpu()
    start_time = time.time()

    use_streaming = not args.static  # streaming by default

    if args.scenario == "fedavg":
        df = run_fl_experiment(
            "fedavg", args.n, args.rounds, horizons,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            streaming=use_streaming,
            pred_suffix=args.pred_suffix,
        )
        df.to_csv(LOG_DIR / f"fl_fedavg_final_{args.suffix}.csv", index=False)
        generate_fl_report(df, "fedavg", time.time() - start_time)

    elif args.scenario == "fedadam":
        df = run_fl_experiment(
            "fedadam", args.n, args.rounds, horizons,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            streaming=use_streaming,
            pred_suffix=args.pred_suffix,
        )
        df.to_csv(LOG_DIR / f"fl_fedadam_final_{args.suffix}.csv", index=False)
        generate_fl_report(df, "fedadam", time.time() - start_time)

    elif args.scenario == "fedyogi":
        df = run_fl_experiment(
            "fedyogi", args.n, args.rounds, horizons,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            streaming=use_streaming,
            pred_suffix=args.pred_suffix,
        )
        df.to_csv(LOG_DIR / f"fl_fedyogi_final_{args.suffix}.csv", index=False)
        generate_fl_report(df, "fedyogi", time.time() - start_time)

    elif args.scenario == "fedadagrad":
        df = run_fl_experiment(
            "fedadagrad", args.n, args.rounds, horizons,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            streaming=use_streaming,
            pred_suffix=args.pred_suffix,
        )
        df.to_csv(LOG_DIR / f"fl_fedadagrad_final_{args.suffix}.csv", index=False)
        generate_fl_report(df, "fedadagrad", time.time() - start_time)

    elif args.scenario == "fedprox":
        df = run_fl_experiment(
            "fedprox", args.n, args.rounds, horizons,
            fedprox_mu=args.mu,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            pred_suffix=args.pred_suffix,
        )
        df.to_csv(LOG_DIR / f"fl_fedprox_final_{args.suffix}.csv", index=False)
        generate_fl_report(df, "fedprox", time.time() - start_time)

    elif args.scenario == "personalised":
        df = run_personalised_fl(
            args.n, args.rounds, fine_tune_epochs=10, horizons=horizons,
            base_strategy=args.base_strategy, fedprox_mu=args.mu,
            fine_tune_lr=args.fine_tune_lr,
            pred_suffix=args.pred_suffix,
        )
        df.to_csv(LOG_DIR / f"fl_personalised_final_{args.suffix}.csv", index=False)
        generate_fl_report(df, "personalised", time.time() - start_time)

    elif args.scenario == "local_mlp":
        df = run_local_mlp(args.n, horizons=horizons, pred_suffix=args.pred_suffix)
        csv_name = f"local_mlp_matched_results_{args.n}_{args.suffix}.csv" if args.suffix else f"local_mlp_matched_results_{args.n}.csv"
        df.to_csv(LOG_DIR / csv_name, index=False)
        generate_fl_report(df, "local_mlp", time.time() - start_time)

    elif args.scenario == "scalability":
        results = run_scalability_experiment(
            num_rounds=args.rounds, horizon=args.horizon or 1,
            base_strategy=args.base_strategy, fedprox_mu=args.mu,
            fine_tune_lr=args.fine_tune_lr,
        )
        for scenario_name, df in results.items():
            df.to_csv(LOG_DIR / f"fl_scalability_{scenario_name}_{args.suffix}.csv", index=False)
            generate_fl_report(df, f"scalability_{scenario_name}", time.time() - start_time)

    elif args.scenario == "all":
        results = {}
        for scenario in ["fedavg", "fedadam", "personalised"]:
            print(f"\n{'#'*60}")
            print(f"  RUNNING: {scenario}")
            print(f"{'#'*60}")
            _log_gpu()

            if scenario == "personalised":
                df = run_personalised_fl(
                    args.n, args.rounds, horizons=horizons,
                    base_strategy=args.base_strategy, fedprox_mu=args.mu,
                    fine_tune_lr=args.fine_tune_lr,
                    pred_suffix=args.pred_suffix,
                )
            elif scenario == "fedadam":
                df = run_fl_experiment(
                    scenario, args.n, args.rounds, horizons,
                    streaming=use_streaming,
                    pred_suffix=args.pred_suffix,
                )
            else:
                df = run_fl_experiment(
                    scenario, args.n, args.rounds, horizons,
                    streaming=use_streaming,
                    pred_suffix=args.pred_suffix,
                )

            df.to_csv(LOG_DIR / f"fl_{scenario}_final_{args.suffix}.csv", index=False)
            results[scenario] = df
            generate_fl_report(df, scenario, time.time() - start_time)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL FL EXPERIMENTS COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to: {LOG_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
