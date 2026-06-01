#!/usr/bin/env python3
"""
Per-Building Prediction Example Figures
=========================================================
Generates actual vs predicted time series for selected buildings,
loading saved predictions from logs/predictions/.

4 panels:
  (a) 1-week view: building where personalised FL beats both local MLP and global FedAdam
  (b) 1-week view: high-consumption building (B176) with FL improvement
  (c) 24-hour winter peak day zoom (highest peak) for B176
  (d) 24-hour winter day where personalised FL best forecasts morning peak for B176

Output: das-fl-paper/paper/figures/prediction_examples.pdf

Usage:
    cd ~/Desktop/varsha_projects/das-flwr-heatdemand
    python pipeline/generate_prediction_examples.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
PRED_DIR = LOG_DIR / "predictions" / "v5_portfolio"
FIG_DIR = PROJECT_ROOT / "das-fl-paper" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Try to load anonymisation
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from pipeline.anonymise_buildings import load_mapping, anon_id
    MAPPING = load_mapping()
except:
    MAPPING = {}
    def anon_id(bid, mapping=None):
        return f"B{int(bid)}"

EXCLUDE = [B001, B037, B238, B028]

# Springer LNCS
DOUBLE_COL = 6.7

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 8,
    'legend.fontsize': 7, 'figure.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05, 'axes.grid': True, 'grid.alpha': 0.3,
})

COLORS = {
    'actual': '#333333',
    'local_mlp': '#762A83',
    'fedadam_global': '#D73027',
    'personalised': '#1B7837',
}

METRIC_BOX_PROPS = dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.9, edgecolor='#ccc')


def load_predictions(scenario, bid, horizon=1):
    """Load saved predictions from logs/predictions/."""
    path = PRED_DIR / f"{scenario}_{bid}_h{horizon}.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    return data['y_actual'], data['y_pred']


def _get_test_timestamps(bid):
    """Return the DatetimeIndex for the test set of a building."""
    f = DATA_DIR / f"{int(bid)}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f, columns=['kwh'])
    n = len(df)
    n_train = int(n * 0.60)
    n_adapt = int(n * 0.15)
    n_val = int(n * 0.10)
    test_start = n_train + n_adapt + n_val
    return df.index[test_start:]


def select_buildings():
    """Hardcoded building selection for Figure 2 (v6, Apr 7, 2026).

    3-panel 1x3 layout. All buildings selected from
    logs/clean_prediction_candidates.txt after quality filtering.
    Each panel shows the first 168h of the test set.

    Panel (a): B156 winter (Jan 21-28, mean 4.4), FL wins
    Panel (b): B042 winter (Feb 12-19, mean 6.5), Local wins
    Panel (c): B205 autumn (Sep 2-9,  mean 1.2), comparable
    """
    buildings = {
        'panel_a': B156,  # winter, FL wins
        'panel_b': B042,  # winter, Local wins
    }

    print("Selected buildings (hardcoded for v6):", flush=True)
    for role, bid in buildings.items():
        print(f"  {role}: {bid}", flush=True)

    return buildings


def _plot_week_panel(ax, bid, title):
    """Plot a 1-week (168h) panel with all 4 lines. No per-panel legend."""
    y_actual, y_local = load_predictions('local_mlp', bid)
    _, y_fedadam = load_predictions('fedadam', bid)
    _, y_pers = load_predictions('personalised_fedadam', bid)

    if y_actual is None:
        ax.text(0.5, 0.5, f"Building {anon_id(bid, MAPPING)}: data not available",
                transform=ax.transAxes, ha='center')
        return

    n = min(168, len(y_actual))
    x = np.arange(n)
    actual = y_actual[:n]

    ax.plot(x, actual, color=COLORS['actual'], linewidth=1.2,
            linestyle='-', label='Actual')
    if y_local is not None:
        ax.plot(x, y_local[:n], color=COLORS['local_mlp'], linewidth=0.8,
                linestyle='--', label='Local MLP')
    if y_fedadam is not None:
        ax.plot(x, y_fedadam[:n], color=COLORS['fedadam_global'], linewidth=0.8,
                linestyle=':', label='FedAdam global')
    if y_pers is not None:
        ax.plot(x, y_pers[:n], color=COLORS['personalised'], linewidth=1.0,
                linestyle='-', label='Personalised FL')

    # Day labels
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tick_positions = np.arange(12, n, 24)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([day_names[j % 7] for j in range(len(tick_positions))])
    ax.set_xlim(0, n)

    # MAE metric box
    lines = []
    _f_mean = DATA_DIR / f"{bid}.parquet"
    if _f_mean.exists():
        _mean_kwh = pd.read_parquet(_f_mean, columns=['kwh'])['kwh'].mean()
        lines.append(f"Mean: {_mean_kwh:.1f} kWh/h")
    if y_local is not None:
        lines.append(f"Local MLP MAE: {np.mean(np.abs(actual - y_local[:n])):.2f}")
    if y_fedadam is not None:
        lines.append(f"FedAdam MAE: {np.mean(np.abs(actual - y_fedadam[:n])):.2f}")
    if y_pers is not None:
        lines.append(f"Personalised MAE: {np.mean(np.abs(actual - y_pers[:n])):.2f}")

    # Title (short, no mean — mean is in the metric box)
    ax.set_title(f"{title} — {anon_id(bid, MAPPING)}", fontsize=8, loc='left')

    ax.text(0.40, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=6,
            verticalalignment='top', horizontalalignment='center', bbox=METRIC_BOX_PROPS)
    ax.set_ylabel('kWh/h')


def _plot_winter_peak_day_panel(ax, bid):
    """Plot a 24-hour zoom on the winter day with highest peak consumption."""
    y_actual, y_local = load_predictions('local_mlp', bid)
    _, y_fedadam = load_predictions('fedadam', bid)
    _, y_pers = load_predictions('personalised_fedadam', bid)

    if y_actual is None or len(y_actual) < 24:
        ax.text(0.5, 0.5, 'Data not available', transform=ax.transAxes, ha='center')
        return

    # Get test set timestamps to find winter days
    test_ts = _get_test_timestamps(bid)
    if test_ts is None or len(test_ts) != len(y_actual):
        ax.text(0.5, 0.5, 'Timestamp alignment error', transform=ax.transAxes, ha='center')
        return

    # Find winter hours (Dec, Jan, Feb)
    winter_mask = test_ts.month.isin([12, 1, 2])
    if winter_mask.sum() < 24:
        ax.text(0.5, 0.5, 'No winter data in test set', transform=ax.transAxes, ha='center')
        return

    # Find the winter day with highest peak (max single-hour consumption)
    # Group by date, find the max within each day
    winter_indices = np.where(winter_mask)[0]
    winter_dates = test_ts[winter_mask].normalize().unique()

    best_peak = -1
    best_date = None
    best_start_idx = None
    for date in winter_dates:
        day_mask = (test_ts >= date) & (test_ts < date + pd.Timedelta(days=1))
        day_indices = np.where(day_mask)[0]
        if len(day_indices) != 24:
            continue
        day_peak = y_actual[day_indices].max()
        if day_peak > best_peak:
            best_peak = day_peak
            best_date = date
            best_start_idx = day_indices[0]

    if best_start_idx is None:
        ax.text(0.5, 0.5, 'No complete winter day found', transform=ax.transAxes, ha='center')
        return

    s = best_start_idx
    e = s + 24
    actual_day = y_actual[s:e]
    x = np.arange(24)

    ax.plot(x, actual_day, color=COLORS['actual'], linewidth=1.2,
            linestyle='-', label='Actual')
    if y_local is not None:
        ax.plot(x, y_local[s:e], color=COLORS['local_mlp'], linewidth=0.8,
                linestyle='--', label='Local MLP')
    if y_fedadam is not None:
        ax.plot(x, y_fedadam[s:e], color=COLORS['fedadam_global'], linewidth=0.8,
                linestyle=':', label='FedAdam global')
    if y_pers is not None:
        ax.plot(x, y_pers[s:e], color=COLORS['personalised'], linewidth=1.0,
                linestyle='-', label='Personalised FL')

    # Mark peak hour
    peak_hour = int(np.argmax(actual_day))
    ax.axvline(peak_hour, color='#999999', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.text(peak_hour + 0.4, 0.97, f'peak {peak_hour:02d}:00',
            fontsize=6, color='#666666', verticalalignment='top',
            transform=ax.get_xaxis_transform())

    # Hour labels in HH:00 format
    ax.set_xticks(np.arange(0, 24))
    ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], rotation=0, ha='center')
    ax.set_xlim(0, 23)

    # MAE for this 24h window
    lines = []
    if y_local is not None:
        lines.append(f"Local MLP MAE: {np.mean(np.abs(actual_day - y_local[s:e])):.2f}")
    if y_fedadam is not None:
        lines.append(f"FedAdam MAE: {np.mean(np.abs(actual_day - y_fedadam[s:e])):.2f}")
    if y_pers is not None:
        lines.append(f"Personalised MAE: {np.mean(np.abs(actual_day - y_pers[s:e])):.2f}")

    # Title with actual date
    date_str = best_date.strftime('%-d %b %Y')
    bid_label = anon_id(bid, MAPPING)
    ax.set_title(f"(c) Winter day forecast — {bid_label}, {date_str}",
                 fontsize=8, loc='left')

    ax.text(0.40, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=6,
            verticalalignment='top', horizontalalignment='center', bbox=METRIC_BOX_PROPS)
    ax.set_ylabel('kWh/h')
    ax.set_xlabel('Hour of day')


def _plot_transition_day_panel(ax, bid, target_date='2024-03-02'):
    """Plot a 24-hour transition season day — contrasts with winter peak in panel (c)."""
    y_actual, y_local = load_predictions('local_mlp', bid)
    _, y_fedadam = load_predictions('fedadam', bid)
    _, y_pers = load_predictions('personalised_fedadam', bid)

    if y_actual is None or y_pers is None:
        ax.text(0.5, 0.5, 'No predictions found', transform=ax.transAxes, ha='center')
        return

    # Find the target date in test set
    f = DATA_DIR / f"{bid}.parquet"
    df = pd.read_parquet(f)
    n = len(df)
    test_start = int(n * 0.85)
    test_dates = df.index[test_start:test_start + len(y_actual)]

    import datetime
    target = pd.Timestamp(target_date)
    day_mask = (test_dates.date == target.date())
    day_indices = np.where(day_mask)[0]

    if len(day_indices) != 24:
        ax.text(0.5, 0.5, f'Date {target_date} not found (got {len(day_indices)} hours)',
                transform=ax.transAxes, ha='center')
        return

    s = day_indices[0]
    e = s + 24
    actual_day = y_actual[s:e]
    x = np.arange(24)

    ax.plot(x, actual_day, color=COLORS['actual'], linewidth=1.2,
            linestyle='-', label='Actual')
    if y_local is not None:
        ax.plot(x, y_local[s:e], color=COLORS['local_mlp'], linewidth=0.8,
                linestyle='--', label='Local MLP')
    if y_fedadam is not None:
        ax.plot(x, y_fedadam[s:e], color=COLORS['fedadam_global'], linewidth=0.8,
                linestyle=':', label='FedAdam global')
    ax.plot(x, y_pers[s:e], color=COLORS['personalised'], linewidth=1.0,
            linestyle='-', label='Personalised FL')

    # Hour labels
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], rotation=0, ha='center')
    ax.set_xlim(0, 23)

    # MAE metric box
    lines = []
    if y_local is not None:
        lines.append(f"Local MLP MAE: {np.mean(np.abs(actual_day - y_local[s:e])):.2f}")
    if y_fedadam is not None:
        lines.append(f"FedAdam MAE: {np.mean(np.abs(actual_day - y_fedadam[s:e])):.2f}")
    lines.append(f"Personalised MAE: {np.mean(np.abs(actual_day - y_pers[s:e])):.2f}")

    bid_label = anon_id(bid, MAPPING)
    ax.set_title(f"(d) Spring transition — {bid_label}, 2 Mar 2024",
                 fontsize=8, loc='left')
    ax.text(0.40, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=6,
            verticalalignment='top', horizontalalignment='center', bbox=METRIC_BOX_PROPS)
    ax.set_ylabel('kWh/h')
    ax.set_xlabel('Hour of day')


def plot_prediction_examples(buildings):
    """Generate the 2-panel prediction example figure (v6, Apr 7).

    1x2 horizontal layout. Each panel is a 1-week (168h) view of
    the first test-set week. Supports the 50/50 selective deployment
    finding without a third illustrative panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharex=False)
    axes = axes.flatten()

    bid_a = int(buildings['panel_a'])
    bid_b = int(buildings['panel_b'])

    print(f"\nPanel (a): building {bid_a} ({anon_id(bid_a, MAPPING)}) - winter, FL wins...", flush=True)
    _plot_week_panel(axes[0], bid_a, '(a) Winter, FL wins')

    print(f"\nPanel (b): building {bid_b} ({anon_id(bid_b, MAPPING)}) - winter, Local wins...", flush=True)
    _plot_week_panel(axes[1], bid_b, '(b) Winter, Local wins')

    # Shared legend below the last panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               ncol=4,
               fontsize=8,
               frameon=True,
               edgecolor='#ccc')

    fig.subplots_adjust(hspace=0.45, wspace=0.28)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    outpath = FIG_DIR / "prediction_examples.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"\nSaved: {outpath} + .png", flush=True)


def main():
    print("=" * 60, flush=True)
    print("GENERATING PREDICTION EXAMPLE FIGURES", flush=True)
    print("=" * 60, flush=True)

    print("\nSelecting buildings...", flush=True)
    buildings = select_buildings()

    print("\nLoading predictions and plotting...", flush=True)
    plot_prediction_examples(buildings)

    print(f"\n{'=' * 60}", flush=True)
    print("DONE", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
