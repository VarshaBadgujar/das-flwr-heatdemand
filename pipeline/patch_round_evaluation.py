"""
DAS-FL Project — Per-Round Evaluation Patch
Updates client_app.py to evaluate BOTH global and local
models after each FL round, matching the thesis pattern.

This enables:
  - Convergence curves (MAE vs rounds)
  - Global vs local model comparison
  - Per-client tracking across rounds
  - The key FL figures for the paper

Run from project root:
    python pipeline/patch_round_evaluation.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def patch_client_app():
    """Update client_app to save local weights and evaluate both models."""
    filepath = PROJECT_ROOT / "dasfl" / "client_app.py"
    with open(filepath, "r") as f:
        content = f.read()

    if "local_weights" in content and "mae_local" in content:
        print("  client_app.py: dual evaluation already present")
        return

    # 1. Add CSV logging imports and function (matching thesis pattern)
    old_imports = """import flwr as fl
from flwr.client import NumPyClient, ClientApp"""

    new_imports = """import csv
import flwr as fl
from flwr.client import NumPyClient, ClientApp"""

    content = content.replace(old_imports, new_imports)

    # 2. Add logging functions after imports
    logging_code = '''

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

'''

    # Insert after the FedProxLoss class definition marker
    marker = "# ── FedProx proximal loss"
    if marker in content:
        content = content.replace(marker, logging_code + "\n" + marker)
    else:
        # Insert after imports
        content = content.replace(
            "from dasfl.task import",
            logging_code + "\nfrom dasfl.task import"
        )

    # 3. Update fit() to save local weights (like thesis)
    old_fit_return = """        return (
            self.model.get_weights(),
            self.data["n_train"],
            {
                "building_id": self.data["building_id"],
                "train_mae": float(metrics["train_mae"]),
                "val_mae": float(metrics["val_mae"]),
            },
        )"""

    new_fit_return = """        # Save local weights BEFORE returning to server (thesis pattern)
        # This allows evaluate() to compare global vs local
        self.local_weights = [w.copy() for w in self.model.get_weights()]

        return (
            self.model.get_weights(),
            self.data["n_train"],
            {
                "building_id": self.data["building_id"],
                "train_mae": float(metrics["train_mae"]),
                "val_mae": float(metrics["val_mae"]),
            },
        )"""

    content = content.replace(old_fit_return, new_fit_return)

    # 4. Replace evaluate() with dual evaluation (global + local)
    old_evaluate = """    def evaluate(self, parameters, config):
        \"\"\"Evaluate global model on local test data.

        Args:
            parameters: Global model weights to evaluate.
            config: Evaluation config.

        Returns:
            Loss, number of test samples, metrics dict.
        \"\"\"
        server_round = config.get("server_round", 1)
        round_idx = max(0, server_round - 1)
        self._ensure_data(round_idx)

        if self.model is None:
            self.model = build_model(self.data["n_features"])

        self.model.set_weights(parameters)

        metrics = evaluate_client(
            self.model,
            self.data["X_test"],
            self.data["y_test"],
        )

        return (
            float(metrics["loss"]),
            self.data["n_test"],
            {
                "building_id": self.data["building_id"],
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
            },
        )"""

    new_evaluate = """    def evaluate(self, parameters, config):
        \"\"\"Evaluate BOTH global and local models on test data.

        Matches thesis pattern (Badgujar 2025):
        1. Evaluate global model (weights from server)
        2. Evaluate local model (weights from fit, before aggregation)
        3. Log both to CSV for convergence analysis

        Returns global model metrics to server.
        \"\"\"
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
        if hasattr(self, "local_weights") and self.local_weights is not None:
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
        )"""

    content = content.replace(old_evaluate, new_evaluate)

    # 5. Initialize local_weights in __init__
    old_init = """    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.data = None
        self.model = None"""

    new_init = """    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.data = None
        self.model = None
        self.local_weights = None"""

    content = content.replace(old_init, new_init)

    with open(filepath, "w") as f:
        f.write(content)
    print("  client_app.py: Added dual evaluation (global + local) per round")


def add_convergence_plot_script():
    """Create a script to plot FL convergence from per-round logs."""
    filepath = PROJECT_ROOT / "pipeline" / "plot_fl_convergence.py"

    script = '''"""
Plot FL convergence curves from per-round client evaluation logs.
Creates Figure 7 for the paper: MAE vs FL rounds (global and local).

Usage:
    python pipeline/plot_fl_convergence.py
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

LOG_DIR = PROJECT_ROOT / "logs" / "fl_metrics"
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]


def load_all_client_logs():
    """Load all per-client evaluation CSVs into one DataFrame."""
    files = sorted(LOG_DIR.glob("client_eval_*.csv"))
    if not files:
        print("No FL metric logs found. Run FL experiments first.")
        return None

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} client logs, {len(df)} total records")
    return df


def plot_convergence(df: pd.DataFrame, save_path: Path):
    """Plot MAE convergence: global vs local, across rounds."""
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("FL Convergence — Global vs Local Model",
                 fontsize=14, fontweight="bold")

    # (a) Median MAE across buildings per round
    ax = axes[0]
    for model_type, color, label in [
        ("global", "#2196F3", "Global model (after aggregation)"),
        ("local", "#FF9800", "Local model (before aggregation)"),
    ]:
        subset = df[df["model_type"] == model_type]
        if len(subset) == 0:
            continue
        grouped = subset.groupby("round")["mae"].agg(["median", "quantile"])
        medians = subset.groupby("round")["mae"].median()
        q25 = subset.groupby("round")["mae"].quantile(0.25)
        q75 = subset.groupby("round")["mae"].quantile(0.75)

        rounds = medians.index
        ax.plot(rounds, medians.values, color=color, lw=2, label=label, marker="o", markersize=4)
        ax.fill_between(rounds, q25.values, q75.values, color=color, alpha=0.15)

    ax.set_xlabel("FL Round")
    ax.set_ylabel("MAE (kWh/h)")
    ax.set_title("(a) Median MAE Across Buildings")
    ax.legend(fontsize=9)

    # (b) Per-building MAE trajectories (sample 10 buildings)
    ax = axes[1]
    global_df = df[df["model_type"] == "global"]
    buildings = global_df["building_id"].unique()
    sample = buildings[:min(10, len(buildings))]

    for bid in sample:
        bdf = global_df[global_df["building_id"] == bid]
        ax.plot(bdf["round"], bdf["mae"], alpha=0.5, lw=1)

    # Add median line
    medians = global_df.groupby("round")["mae"].median()
    ax.plot(medians.index, medians.values, color="black", lw=3,
            label="Median (all buildings)", zorder=10)

    ax.set_xlabel("FL Round")
    ax.set_ylabel("MAE (kWh/h)")
    ax.set_title("(b) Per-Building Global MAE Trajectories")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_global_vs_local_comparison(df: pd.DataFrame, save_path: Path):
    """Plot global vs local MAE per building at the final round."""
    final_round = df["round"].max()
    final = df[df["round"] == final_round]

    global_final = final[final["model_type"] == "global"].set_index("building_id")["mae"]
    local_final = final[final["model_type"] == "local"].set_index("building_id")["mae"]

    # Merge
    comparison = pd.DataFrame({
        "global_mae": global_final,
        "local_mae": local_final,
    }).dropna()

    if len(comparison) == 0:
        print("No comparison data available")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(comparison["local_mae"], comparison["global_mae"],
               alpha=0.6, s=30, color="#2196F3")

    max_val = max(comparison["global_mae"].max(), comparison["local_mae"].max())
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="Equal line")

    below = (comparison["global_mae"] < comparison["local_mae"]).sum()
    total = len(comparison)

    ax.set_xlabel("Local Model MAE (kWh/h)")
    ax.set_ylabel("Global Model MAE (kWh/h)")
    ax.set_title(f"Global vs Local Model (Round {final_round})\\n"
                 f"Global better for {below}/{total} buildings ({100*below/total:.0f}%)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    df = load_all_client_logs()
    if df is not None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plot_convergence(df, FIG_DIR / "fig_fl_convergence.png")
        plot_global_vs_local_comparison(df, FIG_DIR / "fig_fl_global_vs_local.png")
'''

    with open(filepath, "w") as f:
        f.write(script)
    print("  Created: pipeline/plot_fl_convergence.py")


if __name__ == "__main__":
    print("=" * 60)
    print("ADDING PER-ROUND DUAL EVALUATION")
    print("=" * 60)

    print("\n[1/2] Patching dasfl/client_app.py...")
    patch_client_app()

    print("\n[2/2] Creating convergence plot script...")
    add_convergence_plot_script()

    print("\n" + "=" * 60)
    print("PATCH COMPLETE")
    print("=" * 60)
    print("""
After FL runs complete, generate convergence figures:
    python pipeline/plot_fl_convergence.py

This creates:
    fig_fl_convergence.png    — MAE vs rounds (global + local)
    fig_fl_global_vs_local.png — scatter: which buildings benefit from FL
""")