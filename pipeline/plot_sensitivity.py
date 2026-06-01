"""
DAS-FL Project — Sensitivity Analysis Figures
Creates publication-quality figures from tuning results.

Figure 1: Learning rate sensitivity (4 strategies)
Figure 2: Global vs Personalised comparison
Figure 3: Strategy ranking heatmap

Run: python pipeline/plot_sensitivity.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "das-fl-paper" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 200,
})

COLORS = {
    "FedAvg": "#2196F3",
    "FedAdam": "#FF9800",
    "FedYogi": "#4CAF50",
    "FedAdagrad": "#9C27B0",
}


def fig_lr_sensitivity():
    """Plot MAE vs client learning rate for all strategies."""
    print("  Generating: Learning Rate Sensitivity...")

    lrs = [0.0001, 0.001, 0.01]
    lr_labels = ["1e-4", "1e-3", "1e-2"]

    data = {
        "FedAvg":     [0.915, 0.808, 1.270],
        "FedAdam":    [1.655, 0.886, 1.201],
        "FedYogi":    [0.935, 0.846, 1.146],
        "FedAdagrad": [0.850, 0.942, 1.306],
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, maes in data.items():
        ax.plot(lr_labels, maes, marker="o", lw=2.5, markersize=8,
                color=COLORS[strategy], label=strategy)

        # Annotate best
        best_idx = np.argmin(maes)
        ax.annotate(f"{maes[best_idx]:.3f}",
                    xy=(lr_labels[best_idx], maes[best_idx]),
                    xytext=(5, -15), textcoords="offset points",
                    fontsize=8, color=COLORS[strategy], fontweight="bold")

    ax.set_xlabel("Client Learning Rate ($\\eta_l$)")
    ax.set_ylabel("Median MAE (kWh/h)")
    ax.set_title("Client Learning Rate Sensitivity\n(20 buildings, 10 rounds, t+1, streaming FL)")
    ax.legend(fontsize=10)
    ax.axhline(y=0.649, color="gray", ls="--", lw=1, alpha=0.7, label="Local XGBoost")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_lr_sensitivity.png", dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_lr_sensitivity.png")
    plt.close()


def fig_epochs_sensitivity():
    """Plot MAE vs local epochs for all strategies."""
    print("  Generating: Local Epochs Sensitivity...")

    epochs = ["5", "10"]

    data = {
        "FedAvg":     [0.821, 0.777],
        "FedAdam":    [0.909, 0.791],
        "FedYogi":    [0.840, 0.847],
        "FedAdagrad": [0.916, 0.918],
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(epochs))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, (strategy, maes) in enumerate(data.items()):
        bars = ax.bar(x + offsets[i] * width, maes, width,
                      label=strategy, color=COLORS[strategy], alpha=0.85)
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{mae:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Local Epochs per FL Round")
    ax.set_ylabel("Median MAE (kWh/h)")
    ax.set_title("Local Epochs with Early Stopping (patience=2)\n"
                 "(20 buildings, 10 rounds, t+1, streaming FL)")
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend(fontsize=9)
    ax.axhline(y=0.649, color="gray", ls="--", lw=1, alpha=0.7)
    ax.text(1.3, 0.645, "Local XGBoost", fontsize=8, color="gray")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_epochs_sensitivity.png", dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_epochs_sensitivity.png")
    plt.close()


def fig_global_vs_personalised():
    """Bar chart: global model vs personalised for each strategy."""
    print("  Generating: Global vs Personalised...")

    strategies = ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad"]

    global_mae = [0.777, 0.791, 0.840, 0.850]
    pers_mae = [0.677, 0.660, 0.681, 0.661]
    improvement = [round((g - p) / g * 100, 1) for g, p in zip(global_mae, pers_mae)]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(strategies))
    width = 0.3

    bars1 = ax.bar(x - width/2, global_mae, width, label="Global Model",
                   color=[COLORS[s] for s in strategies], alpha=0.5, edgecolor="black", lw=0.5)
    bars2 = ax.bar(x + width/2, pers_mae, width, label="Personalised FL",
                   color=[COLORS[s] for s in strategies], alpha=0.9, edgecolor="black", lw=0.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, color="gray")
    for bar, imp in zip(bars2, improvement):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}\n({imp}%↓)", ha="center", fontsize=8,
                fontweight="bold")

    # Reference lines
    ax.axhline(y=0.649, color="blue", ls="--", lw=1.5, alpha=0.6)
    ax.text(3.5, 0.635, "Local XGBoost (0.649)", fontsize=8, color="blue")

    ax.axhline(y=0.498, color="green", ls="--", lw=1.5, alpha=0.6)
    ax.text(3.5, 0.485, "Centralised XGB (0.498)", fontsize=8, color="green")

    ax.set_xlabel("FL Aggregation Strategy")
    ax.set_ylabel("Median MAE (kWh/h)")
    ax.set_title("Effect of Local Fine-Tuning on FL Performance\n"
                 "(20 buildings, 10 rounds, t+1, streaming FL)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0.4, 0.95)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_global_vs_personalised.png", dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_global_vs_personalised.png")
    plt.close()


def fig_strategy_heatmap():
    """Heatmap of all tuning results."""
    print("  Generating: Strategy Heatmap...")

    # All MAE results organized
    configs = [
        "epochs=5", "epochs=10",
        "lr=1e-4", "lr=1e-3", "lr=1e-2",
        "Personalised"
    ]

    data = {
        "FedAvg":     [0.821, 0.777, 0.915, 0.808, 1.270, 0.677],
        "FedAdam":    [0.909, 0.791, 1.655, 0.886, 1.201, 0.660],
        "FedYogi":    [0.840, 0.847, 0.935, 0.846, 1.146, 0.681],
        "FedAdagrad": [0.916, 0.918, 0.850, 0.942, 1.306, 0.661],
    }

    strategies = list(data.keys())
    matrix = np.array([data[s] for s in strategies])

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0.5, vmax=1.3)

    # Annotate cells
    for i in range(len(strategies)):
        for j in range(len(configs)):
            val = matrix[i, j]
            color = "white" if val > 1.0 else "black"
            weight = "bold" if val == min(matrix[:, j]) else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight=weight)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=11)

    ax.set_title("MAE Across All Configurations\n"
                 "(Green = better, Red = worse. Bold = best per column)",
                 fontsize=12)

    plt.colorbar(im, ax=ax, label="MAE (kWh/h)", shrink=0.8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_sensitivity_heatmap.png", dpi=200, bbox_inches="tight")
    print(f"    Saved: fig_sensitivity_heatmap.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING SENSITIVITY ANALYSIS FIGURES")
    print("=" * 60)

    fig_lr_sensitivity()
    fig_epochs_sensitivity()
    fig_global_vs_personalised()
    fig_strategy_heatmap()

    print(f"\n{'=' * 60}")
    print("ALL SENSITIVITY FIGURES GENERATED")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)