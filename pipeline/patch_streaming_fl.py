"""
DAS-FL Project — Streaming FL Data Loading Patch
Adds streaming (rolling window) data loading to task.py.

Run from project root:
    python pipeline/patch_streaming_fl.py

Changes:
    - Adds load_client_data_streaming() to dasfl/task.py
    - Each FL round gets a unique time segment
    - Test set is always the final segment
    - Supports both static (original) and streaming modes
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def patch_task_py():
    """Add streaming data loading to task.py."""
    filepath = PROJECT_ROOT / "dasfl" / "task.py"
    with open(filepath, "r") as f:
        content = f.read()

    if "load_client_data_streaming" in content:
        print("  task.py: streaming functions already present")
        return

    streaming_code = '''

# ── Streaming FL data loading ───────────────────────────────
# Each FL round gets a different time segment, simulating
# real deployment where new data arrives periodically.
# This extends the thesis approach (Badgujar 2025) to scale.

def prepare_streaming_rounds(
    partition_id: int,
    building_ids: list[str],
    horizon: int = 1,
    num_rounds: int = 30,
    test_ratio: float = 0.15,
) -> dict:
    """Pre-compute all round data segments for one building.

    Splits the training period into num_rounds segments.
    Each segment becomes the training data for one FL round.
    The test set is the final portion (same for all rounds).

    Timeline:
        [seg_1][seg_2]...[seg_N][  TEST  ]
        Round 1 trains on seg_1
        Round 2 trains on seg_2
        ...
        Round N trains on seg_N
        All rounds evaluate on TEST

    For validation, each segment uses its last 15% as val.

    Args:
        partition_id: Client index.
        building_ids: List of building IDs.
        horizon: Forecast horizon.
        num_rounds: Number of FL rounds (= number of segments).
        test_ratio: Fraction reserved for testing.

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

    # Reserve test set (final portion)
    n_test = int(n * test_ratio)
    n_train_total = n - n_test

    X_test = X.iloc[n_train_total:].values.astype(np.float32)
    y_test = y.iloc[n_train_total:].values.astype(np.float32)

    # Split training period into num_rounds segments
    segment_size = n_train_total // num_rounds
    if segment_size < 100:
        # If segments too small, use fewer rounds
        num_rounds = max(1, n_train_total // 200)
        segment_size = n_train_total // num_rounds

    # Compute normalization from ALL training data (not per-segment)
    X_train_all = X.iloc[:n_train_total].values.astype(np.float32)
    mean = np.mean(X_train_all, axis=0)
    std = np.std(X_train_all, axis=0)
    std[std < 1e-8] = 1.0

    # Normalize test set
    X_test_norm = (X_test - mean) / std

    # Create per-round segments
    rounds_data = []
    for r in range(num_rounds):
        seg_start = r * segment_size
        seg_end = min(seg_start + segment_size, n_train_total)

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
        partition_id, building_ids, horizon, num_rounds, test_ratio
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

'''

    # Insert before the model section
    marker = "# ── Model ───────────────────────────────────────────────────"
    if marker in content:
        content = content.replace(marker, streaming_code + "\n" + marker)
    else:
        # Fallback: insert before build_model
        content = content.replace("def build_model(", streaming_code + "\ndef build_model(")

    with open(filepath, "w") as f:
        f.write(content)
    print("  task.py: Added streaming data loading functions")


def patch_client_app():
    """Update client_app.py to support streaming mode."""
    filepath = PROJECT_ROOT / "dasfl" / "client_app.py"
    with open(filepath, "r") as f:
        content = f.read()

    if "load_client_data_streaming" in content:
        print("  client_app.py: streaming support already present")
        return

    # Add import
    old_import = """from dasfl.task import (
    load_client_data,
    build_model,
    train_client,
    evaluate_client,
    get_building_ids,
)"""

    new_import = """from dasfl.task import (
    load_client_data,
    load_client_data_streaming,
    build_model,
    train_client,
    evaluate_client,
    get_building_ids,
)"""

    content = content.replace(old_import, new_import)

    # Add streaming flag to global state
    old_global = """_fedprox_mu: float = 0.0  # 0.0 = FedAvg, >0 = FedProx"""
    new_global = """_fedprox_mu: float = 0.0  # 0.0 = FedAvg, >0 = FedProx
_streaming: bool = False
_num_rounds: int = 30"""

    content = content.replace(old_global, new_global)

    # Update configure_client
    old_configure = """def configure_client(
    building_ids: list[str],
    horizon: int,
    n_features: int,
    fedprox_mu: float = 0.0,
):
    \"\"\"Set global client configuration before FL run.\"\"\"
    global _building_ids, _horizon, _n_features, _fedprox_mu
    _building_ids = building_ids
    _horizon = horizon
    _n_features = n_features
    _fedprox_mu = fedprox_mu"""

    new_configure = """def configure_client(
    building_ids: list[str],
    horizon: int,
    n_features: int,
    fedprox_mu: float = 0.0,
    streaming: bool = False,
    num_rounds: int = 30,
):
    \"\"\"Set global client configuration before FL run.\"\"\"
    global _building_ids, _horizon, _n_features, _fedprox_mu
    global _streaming, _num_rounds
    _building_ids = building_ids
    _horizon = horizon
    _n_features = n_features
    _fedprox_mu = fedprox_mu
    _streaming = streaming
    _num_rounds = num_rounds"""

    content = content.replace(old_configure, new_configure)

    # Update _ensure_data to support streaming
    old_ensure = """    def _ensure_data(self):
        \"\"\"Lazy-load data on first use.\"\"\"
        if self.data is None:
            # Use configured list, or auto-discover (for Ray workers)
            bids = _building_ids if _building_ids else get_building_ids()
            horizon = _horizon if _horizon else 1
            self.data = load_client_data(
                self.partition_id,
                bids,
                horizon=horizon,
            )"""

    new_ensure = """    def _load_round_data(self, round_idx: int = 0):
        \"\"\"Load data for a specific round (streaming) or all data (static).\"\"\"
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
        \"\"\"Lazy-load data. In streaming mode, reload each round.\"\"\"
        if _streaming:
            # Always reload in streaming mode (different data per round)
            self._load_round_data(round_idx)
        elif self.data is None:
            self._load_round_data(round_idx)"""

    content = content.replace(old_ensure, new_ensure)

    # Update fit() to pass round_idx
    old_fit_ensure = """        self._ensure_data()

        # Build or reuse model
        if self.model is None:
            self.model = build_model(self.data["n_features"])

        # Set global weights
        self.model.set_weights(parameters)

        # Get config
        epochs = config.get("local_epochs", 5)
        batch_size = config.get("batch_size", 64)
        lr = config.get("learning_rate", 0.001)"""

    new_fit_ensure = """        # Get round index from server config
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
        lr = config.get("learning_rate", 0.001)"""

    content = content.replace(old_fit_ensure, new_fit_ensure)

    # Update evaluate() to pass round_idx
    old_eval_ensure = """        self._ensure_data()

        if self.model is None:
            self.model = build_model(self.data["n_features"])

        self.model.set_weights(parameters)"""

    new_eval_ensure = """        server_round = config.get("server_round", 1)
        round_idx = max(0, server_round - 1)
        self._ensure_data(round_idx)

        if self.model is None:
            self.model = build_model(self.data["n_features"])

        self.model.set_weights(parameters)"""

    content = content.replace(old_eval_ensure, new_eval_ensure)

    with open(filepath, "w") as f:
        f.write(content)
    print("  client_app.py: Updated for streaming FL support")


def patch_run_fl():
    """Update run_fl.py to support streaming mode."""
    filepath = PROJECT_ROOT / "pipeline" / "run_fl.py"
    with open(filepath, "r") as f:
        content = f.read()

    if "streaming" in content and "configure_client" in content and "_streaming" not in content:
        print("  run_fl.py: streaming support may already be present")

    # Add streaming parameter to run_fl_experiment
    old_sig = """def run_fl_experiment(
    scenario: str,
    n_buildings: int,
    num_rounds: int = 50,
    horizons: list[int] = None,
    fedprox_mu: float = 0.01,
    local_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> pd.DataFrame:"""

    new_sig = """def run_fl_experiment(
    scenario: str,
    n_buildings: int,
    num_rounds: int = 50,
    horizons: list[int] = None,
    fedprox_mu: float = 0.01,
    local_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    streaming: bool = True,
) -> pd.DataFrame:"""

    content = content.replace(old_sig, new_sig)

    # Update configure_client call to pass streaming
    old_config_client = """        configure_client(building_ids, horizon, n_features, fedprox_mu=mu)"""
    new_config_client = """        configure_client(
            building_ids, horizon, n_features,
            fedprox_mu=mu, streaming=streaming, num_rounds=num_rounds,
        )"""

    content = content.replace(old_config_client, new_config_client)

    # Add --streaming flag to argparser
    old_argparse = """    parser.add_argument("--mu", type=float, default=0.01,
                        help="FedProx proximal term strength")"""

    new_argparse = """    parser.add_argument("--mu", type=float, default=0.01,
                        help="FedProx proximal term strength")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Use streaming FL (different data per round)")
    parser.add_argument("--static", action="store_true",
                        help="Use static FL (same data every round)")"""

    content = content.replace(old_argparse, new_argparse)

    # Add streaming logic in main
    old_main_fedavg = """    if args.scenario == "fedavg":
        df = run_fl_experiment(
            "fedavg", args.n, args.rounds, horizons,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )"""

    new_main_fedavg = """    use_streaming = not args.static  # streaming by default

    if args.scenario == "fedavg":
        df = run_fl_experiment(
            "fedavg", args.n, args.rounds, horizons,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            streaming=use_streaming,
        )"""

    content = content.replace(old_main_fedavg, new_main_fedavg)

    with open(filepath, "w") as f:
        f.write(content)
    print("  run_fl.py: Added streaming FL support")


if __name__ == "__main__":
    print("=" * 60)
    print("ADDING STREAMING FL DATA LOADING")
    print("=" * 60)

    print("\n[1/3] Patching dasfl/task.py...")
    patch_task_py()

    print("\n[2/3] Patching dasfl/client_app.py...")
    patch_client_app()

    print("\n[3/3] Patching pipeline/run_fl.py...")
    patch_run_fl()

    print("\n" + "=" * 60)
    print("STREAMING FL PATCH COMPLETE")
    print("=" * 60)
    print("""
Usage:
  # Streaming FL (default — different data each round):
  python -u pipeline/run_fl.py --scenario fedavg --n 20 --rounds 30 --horizon 1

  # Static FL (same data every round, for comparison):
  python -u pipeline/run_fl.py --scenario fedavg --n 20 --rounds 30 --horizon 1 --static

  # The streaming flag is ON by default
""")