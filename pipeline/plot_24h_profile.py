"""
DAS-FL — 24-Hour Building Profile Plot
Creates a focused 24-hour actual vs predicted plot for a
single building, showing:
  - Diurnal consumption pattern (morning ramp, peak, night)
  - Prediction accuracy at each hour
  - Error shading between actual and predicted
  - Peak hour and night minimum annotations

For paper Section 5 — makes estimation task tangible.

Usage:
  # Auto-recommend best building
  PYTHONPATH=. python pipeline/plot_24h_profile.py

  # Single building plot
  PYTHONPATH=. python pipeline/plot_24h_profile.py --building-id 123456

  # Two-panel stacked figure (for paper)
  PYTHONPATH=. python pipeline/plot_24h_profile.py --building-id 123456 234567

  # Three or more panels
  PYTHONPATH=. python pipeline/plot_24h_profile.py --building-id 123456 234567 345678
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import sys
import os
import argparse
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dasfl.task import build_model, load_client_data

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
DATA_DIR = PROJECT_ROOT / config["paths"]["processed_data"]
FIG_DIR = PROJECT_ROOT / config["paths"]["paper_figures"]
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_fl_building_ids():
    """Load the 100 building IDs used in FL experiments."""
    fl_file = LOG_DIR / "fl_building_ids.txt"
    if fl_file.exists():
        return open(fl_file).read().strip().split("\n")
    else:
        print("ERROR: fl_building_ids.txt not found")
        sys.exit(1)


def find_recommended_building(building_ids, min_consumption=5.0):
    """Find the best building for 24h profile plot.

    Criteria (aligned with select_buildings.py):
    - Mean consumption > min_consumption (clear signal)
    - High diurnal variation (visible morning/evening pattern)
    - Good prediction accuracy (R² > 0.7, low MAE)

    Returns:
        tuple: (building_id, index_in_list)
    """
    results_path = LOG_DIR / "local_mlp_matched_results.csv"
    if not results_path.exists():
        print("  No results file found, using first building")
        return building_ids[0], 0

    df = pd.read_csv(results_path)
    df["building_id"] = df["building_id"].astype(str)
    h1 = df[(df["horizon"] == 1) & (df["status"] == "success")].copy()

    # Get mean consumption and diurnal variation per building
    candidates = []
    for bid in h1["building_id"].unique():
        pfile = Path(DATA_DIR) / f"{bid}.parquet"
        if not pfile.exists():
            continue

        bdf = pd.read_parquet(pfile)
        if "kwh" not in bdf.columns:
            continue

        mean_kwh = bdf["kwh"].mean()
        if mean_kwh < min_consumption:
            continue

        diurnal = 0.0
        if hasattr(bdf.index, "hour"):
            diurnal = bdf.groupby(bdf.index.hour)["kwh"].mean().std()

        row = h1[h1["building_id"] == bid].iloc[0]
        candidates.append({
            "building_id": bid,
            "mean_consumption": mean_kwh,
            "diurnal_variation": diurnal,
            "mae": row["mae"],
            "r2": row["r2"],
        })

    if not candidates:
        print(f"  No buildings with mean > {min_consumption} kWh/h, using first")
        return building_ids[0], 0

    cdf = pd.DataFrame(candidates)

    # Among candidates with good R², pick highest diurnal variation
    good_r2 = cdf[cdf["r2"] > 0.7]
    if len(good_r2) > 0:
        best = good_r2.sort_values("diurnal_variation", ascending=False).iloc[0]
    else:
        best = cdf.sort_values("r2", ascending=False).iloc[0]

    bid = best["building_id"]
    try:
        idx = building_ids.index(bid)
    except ValueError:
        idx = 0
        bid = building_ids[0]

    print(f"  Recommended: Building {bid} "
          f"(mean={best['mean_consumption']:.1f} kWh/h, "
          f"diurnal={best['diurnal_variation']:.3f}, "
          f"MAE={best['mae']:.3f}, R²={best['r2']:.3f})")
    return bid, idx


def find_typical_day(y_true, y_pred):
    """Find a typical winter day with clear diurnal pattern.

    Criteria:
    - Above-median daily consumption (winter, not summer)
    - No extreme spikes (max < 5x daily mean)
    - Clear variation (std > 0.5 kWh/h)
    - Good prediction tracking (low MAE relative to variation)

    Returns:
        int: day index (0-based)
    """
    n_days = len(y_true) // 24
    median_daily = np.median([
        y_true[d * 24:(d + 1) * 24].mean()
        for d in range(n_days)
        if len(y_true[d * 24:(d + 1) * 24]) == 24
    ])

    best_day = 0
    best_score = float("inf")

    for d in range(n_days):
        dt = y_true[d * 24:(d + 1) * 24]
        dp = y_pred[d * 24:(d + 1) * 24]
        if len(dt) < 24:
            continue

        dmean = dt.mean()
        dstd = dt.std()
        dmax = dt.max()
        dmae = np.mean(np.abs(dt - dp))

        # Filter: winter day, no spike, clear variation
        if dmean < median_daily:
            continue
        if dmax > 5 * dmean:
            continue
        if dstd < 0.5:
            continue

        # Score: prefer good tracking (low MAE/std ratio)
        score = dmae / max(dstd, 0.1)
        if score < best_score:
            best_score = score
            best_day = d

    return best_day


def generate_profile(building_ids, bid, idx):
    """Train model and extract 24h profile from test data."""
    print(f"  Training MLP for building {bid}...")
    data = load_client_data(idx, building_ids, horizon=1)

    model = build_model(data["n_features"], [64, 32], 0.2, 0.001)
    model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=30, batch_size=64, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True
        )],
    )

    y_pred = np.clip(model.predict(data["X_test"], verbose=0).flatten(), 0, None)
    y_true = data["y_test"]

    tf.keras.backend.clear_session()
    del model

    print(f"  Test set: {len(y_true)} hours ({len(y_true) // 24} days)")

    # Find typical day
    best_day = find_typical_day(y_true, y_pred)
    s, e = best_day * 24, best_day * 24 + 24
    yt, yp = y_true[s:e], y_pred[s:e]

    mae_24h = np.mean(np.abs(yt - yp))
    print(f"  Selected day {best_day}: mean={yt.mean():.1f} kWh/h, "
          f"max={yt.max():.1f}, MAE={mae_24h:.2f}")

    return {
        "y_true": yt,
        "y_pred": yp,
        "hours": np.arange(24),
        "mae": mae_24h,
        "day_index": best_day,
    }


def plot_profile(profile, bid, r2_value, save_path):
    """Create the 24-hour profile figure for a single building."""
    hours = profile["hours"]
    yt = profile["y_true"]
    yp = profile["y_pred"]

    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_panel(ax, yt, yp, hours, bid, profile, r2_value)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path.name}")
    plt.close()


def plot_multi_profile(profiles, bids, r2_values, save_path):
    """Create a stacked multi-panel figure for multiple buildings.

    Usage:
        --building-id 123456 234567
        Produces (a) Building 123456 on top, (b) Building 234567 below.
    """
    n = len(profiles)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    panels = "abcdefgh"
    for i, (ax, profile, bid, r2) in enumerate(zip(axes, profiles, bids, r2_values)):
        hours = profile["hours"]
        yt = profile["y_true"]
        yp = profile["y_pred"]
        panel_label = panels[i] if i < len(panels) else str(i + 1)
        _draw_panel(ax, yt, yp, hours, bid, profile, r2, panel=panel_label)

    # Only bottom panel gets x-axis label and tick labels
    axes[-1].set_xticks(range(0, 24, 2))
    axes[-1].set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)],
                              rotation=45, fontsize=9)
    axes[-1].set_xlabel("Hour of day", fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path.name}")
    plt.close()


def _draw_panel(ax, yt, yp, hours, bid, profile, r2_value, panel=None):
    """Draw a single panel — shared by single and multi-panel plots."""
    # Actual and predicted
    ax.plot(hours, yt, color="#2196F3", lw=2.5, marker="o", markersize=4,
            label="Actual consumption", zorder=3)
    ax.plot(hours, yp, color="#FF9800", lw=2.5, marker="s", markersize=3,
            ls="--", label="Personalised FL prediction", zorder=3)

    # Error shading
    ax.fill_between(hours, yt, yp, alpha=0.15, color="#FF9800",
                    label="Prediction error")

    # Annotate peak hour
    peak_h = np.argmax(yt)
    y_range = yt.max() - yt.min()
    ax.annotate(
        f"Peak: {yt[peak_h]:.1f} kWh/h",
        xy=(peak_h, yt[peak_h]),
        xytext=(min(peak_h + 3, 20), yt[peak_h] + y_range * 0.15),
        fontsize=9, fontweight="bold", color="#D32F2F",
        arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
    )

    # Annotate night minimum
    min_h = np.argmin(yt)
    ax.annotate(
        f"Night: {yt[min_h]:.1f} kWh/h",
        xy=(min_h, yt[min_h]),
        xytext=(min(min_h + 3, 20), yt[min_h] - y_range * 0.15),
        fontsize=9, color="#666666",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1),
    )

    # Formatting
    season_str = profile.get("season_label", "")
    prefix = f"({panel}) " if panel else ""
    season_part = f" ({season_str})" if season_str else ""
    ax.set_title(
        f"{prefix}Building {bid}{season_part} — "
        f"Mean: {yt.mean():.1f} kWh/h | "
        f"MAE: {profile['mae']:.2f} kWh/h | "
        f"R² = {r2_value:.3f}",
        fontsize=11
    )
    ax.set_ylabel("Heat consumption (kWh/h)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)],
                       rotation=45, fontsize=9)


def get_r2_from_results(bid):
    """Get R² value from results file for the figure title."""
    results_path = LOG_DIR / "local_mlp_matched_results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        df["building_id"] = df["building_id"].astype(str)
        row = df[(df["building_id"] == bid) &
                 (df["horizon"] == 1) &
                 (df["status"] == "success")]
        if len(row) > 0:
            return row.iloc[0]["r2"]
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAS-FL — 24-Hour Building Profile Plot"
    )
    parser.add_argument("--building-id", type=str, nargs="+", default=None,
                        help="One or more building IDs. "
                             "Single ID = single plot. "
                             "Multiple IDs = stacked multi-panel figure. "
                             "Default: auto-recommend.")
    parser.add_argument("--min-consumption", type=float, default=5.0,
                        help="Min mean consumption for auto-selection (default: 5.0)")
    args = parser.parse_args()

    print("=" * 60)
    print("DAS-FL — 24-HOUR BUILDING PROFILE")
    print("=" * 60)

    building_ids = load_fl_building_ids()
    print(f"\n  FL buildings: {len(building_ids)}")

    # Determine which buildings to plot
    if args.building_id:
        target_bids = args.building_id
        # Validate all IDs
        for bid in target_bids:
            if bid not in building_ids:
                print(f"\n  ERROR: Building {bid} not in FL building list")
                sys.exit(1)
        print(f"\n  Target buildings: {', '.join(target_bids)}")
    else:
        print(f"\n  Auto-selecting building "
              f"(min consumption={args.min_consumption} kWh/h)...")
        bid, _ = find_recommended_building(building_ids, args.min_consumption)
        target_bids = [bid]

    # Generate profiles for all target buildings
    profiles = []
    r2_values = []
    valid_bids = []

    for i, bid in enumerate(target_bids):
        idx = building_ids.index(bid)
        r2_val = get_r2_from_results(bid)
        print(f"\n[{i+1}/{len(target_bids)}] Building {bid} (index {idx}, R²={r2_val:.3f})")

        profile = generate_profile(building_ids, bid, idx)
        if profile is not None and len(profile["y_true"]) == 24:
            profiles.append(profile)
            r2_values.append(r2_val)
            valid_bids.append(bid)
        else:
            print(f"  WARNING: Could not generate profile for {bid}, skipping")

    if not profiles:
        print("\n  ERROR: No valid profiles generated")
        sys.exit(1)

    # Plot
    print(f"\n  Creating figure ({len(profiles)} panel{'s' if len(profiles)>1 else ''})...")
    save_path = FIG_DIR / "fig_24h_building_profile.png"

    if len(profiles) == 1:
        plot_profile(profiles[0], valid_bids[0], r2_values[0], save_path)
    else:
        plot_multi_profile(profiles, valid_bids, r2_values, save_path)

    print(f"\n{'=' * 60}")
    print("24-HOUR PROFILE COMPLETE")
    print(f"{'=' * 60}")