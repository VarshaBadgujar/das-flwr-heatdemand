#!/usr/bin/env python3
"""
Outlier Building Analysis
=======================================================
Analyses the 4 seasonal-mismatch buildings that are also
top peak contributors. Documents why they're outliers and
their operational significance.

Buildings: B001, B037, B238, B028

Output:
  - logs/outlier_analysis_report.txt  (full text report)
  - logs/outlier_analysis_stats.csv   (per-building stats)
  - das-fl-paper/paper/figures/outlier_*.pdf (figures)

Usage:
    cd ~/Desktop/varsha_projects/das-flwr-heatdemand
    python pipeline/analyze_outliers.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.anonymise_buildings import load_mapping, anon_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
FIG_DIR = PROJECT_ROOT / "das-fl-paper" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTLIERS = [B001, B037, B238, B028]

# Springer LNCS widths
SINGLE_COL = 4.8
DOUBLE_COL = 6.7

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05, 'axes.grid': True, 'grid.alpha': 0.3,
})

COLORS = {
    'outlier':'#D73027', 'normal': '#4575B4', 'train': '#2166AC',
    'adapt': '#92C5DE', 'val': '#FDDBC7', 'test': '#D6604D',
    'winter': '#2166AC', 'summer': '#F4A582',
}


def load_all_buildings(fl_ids):
    """Load consumption series for all FL buildings."""
    series = {}
    for bid in fl_ids:
        f = DATA_DIR / f"{int(bid)}.parquet"
        if f.exists():
            series[bid] = pd.read_parquet(f, columns=['kwh'])['kwh']
    return series


def main():
    mapping = load_mapping()

    report = []
    report.append("=" * 70)
    report.append("OUTLIER BUILDING ANALYSIS — DAS-FL Paper")
    report.append(f"Buildings: {[anon_id(b, mapping) for b in OUTLIERS]}")
    report.append("=" * 70)

    # Load data
    fl_ids = pd.read_csv(LOG_DIR / "fl_building_ids_250.txt", header=None)[0].tolist()
    all_series = load_all_buildings(fl_ids)
    print(f"Loaded {len(all_series)} buildings", flush=True)

    # Portfolio
    port_all = pd.DataFrame(all_series).sum(axis=1).dropna()
    port_clean = pd.DataFrame(
        {k: v for k, v in all_series.items() if k not in OUTLIERS}
    ).sum(axis=1).dropna()

    # ================================================================
    # 1. PORTFOLIO SHARE
    # ================================================================
    outlier_total = sum(all_series[b].mean() for b in OUTLIERS if b in all_series)
    all_total = port_all.mean()

    report.append(f"\n{'='*50}")
    report.append("1. PORTFOLIO SHARE")
    report.append(f"{'='*50}")
    report.append(f"  Portfolio mean consumption:  {all_total:.1f} kWh/h (250 buildings)")
    report.append(f"  4 outlier buildings:         {outlier_total:.1f} kWh/h "
                  f"({outlier_total/all_total*100:.1f}% of portfolio)")
    report.append(f"  Remaining 246 buildings:     {port_clean.mean():.1f} kWh/h "
                  f"({port_clean.mean()/all_total*100:.1f}%)")

    # ================================================================
    # 2. PEAK CONTRIBUTION
    # ================================================================
    p95 = port_all.quantile(0.95)
    peak_hours = port_all[port_all > p95].index
    total_peak = port_all.reindex(peak_hours).mean()
    outlier_peak = sum(all_series[b].reindex(peak_hours).mean() for b in OUTLIERS if b in all_series)

    report.append(f"\n{'='*50}")
    report.append("2. PEAK CONTRIBUTION (during P95 network peak hours)")
    report.append(f"{'='*50}")
    report.append(f"  Total peak load:    {total_peak:.1f} kWh/h")
    report.append(f"  4 outliers share:   {outlier_peak:.1f} kWh/h "
                  f"({outlier_peak/total_peak*100:.1f}% of network peak)")
    report.append(f"  Peak hours count:   {len(peak_hours)}")

    # ================================================================
    # 3. PER-BUILDING DETAIL
    # ================================================================
    report.append(f"\n{'='*50}")
    report.append("3. PER-BUILDING DETAIL")
    report.append(f"{'='*50}")
    report.append(f"  {'Building':<12} {'Mean':>7} {'Max':>7} {'Peak':>7} "
                  f"{'Share':>7} {'W/S':>6} {'Years':>6} {'Train→':>10} {'Test→':>10} {'Ratio':>6}")
    report.append(f"  {'-'*85}")

    stats_list = []
    for bid in OUTLIERS:
        if bid not in all_series:
            continue
        s = all_series[bid]
        n = len(s)
        train_end = int(n * 0.60)
        adapt_end = train_end + int(n * 0.15)
        val_end = adapt_end + int(n * 0.10)
        test_start = val_end

        train_mean = s.iloc[:train_end].mean()
        adapt_mean = s.iloc[train_end:adapt_end].mean()
        test_mean = s.iloc[test_start:].mean()
        ratio = test_mean / train_mean if train_mean > 0 else float('inf')

        peak_mean = s.reindex(peak_hours).mean()
        peak_share = peak_mean / total_peak * 100

        winter = s[s.index.month.isin([12, 1, 2])].mean()
        summer = s[s.index.month.isin([6, 7, 8])].mean()
        ws_ratio = winter / summer if summer > 0 else float('inf')

        years = n / 8760

        stats = {
            'building_id': bid,
            'total_hours': n,
            'years': round(years, 1),
            'mean_kwh': round(s.mean(), 2),
            'max_kwh': round(s.max(), 1),
            'std_kwh': round(s.std(), 2),
            'peak_mean_kwh': round(peak_mean, 2),
            'peak_share_pct': round(peak_share, 1),
            'winter_mean': round(winter, 2),
            'summer_mean': round(summer, 2),
            'winter_summer_ratio': round(ws_ratio, 1),
            'train_mean': round(train_mean, 2),
            'adapt_mean': round(adapt_mean, 2),
            'test_mean': round(test_mean, 2),
            'test_train_ratio': round(ratio, 2),
            'train_end_date': str(s.index[train_end - 1].strftime('%Y-%m-%d')),
            'test_start_date': str(s.index[test_start].strftime('%Y-%m-%d')),
        }
        stats_list.append(stats)

        report.append(f"  {anon_id(bid, mapping):<12} {s.mean():>7.1f} {s.max():>7.1f} {peak_mean:>7.1f} "
                      f"{peak_share:>6.1f}% {ws_ratio:>5.1f}x {years:>5.1f}y "
                      f"{s.index[train_end-1].strftime('%Y-%m'):>10} "
                      f"{s.index[test_start].strftime('%Y-%m'):>10} {ratio:>5.2f}x")

    # ================================================================
    # 4. WHY THEY'RE OUTLIERS — SEASONAL MISMATCH
    # ================================================================
    report.append(f"\n{'='*50}")
    report.append("4. SEASONAL MISMATCH EXPLANATION")
    report.append(f"{'='*50}")
    for st in stats_list:
        bid = st['building_id']
        report.append(f"\n  Building {anon_id(bid, mapping)}:")
        report.append(f"    Data span: {st['years']} years ({st['total_hours']} hours)")
        report.append(f"    60% train ends: {st['train_end_date']}")
        report.append(f"    Test starts:    {st['test_start_date']}")
        report.append(f"    Train mean:     {st['train_mean']:.2f} kWh/h")
        report.append(f"    Test mean:      {st['test_mean']:.2f} kWh/h")
        report.append(f"    Ratio:          {st['test_train_ratio']:.2f}x")

        if st['test_train_ratio'] > 1.3:
            report.append(f"    ⚠ REGIME CHANGE: test consumption {st['test_train_ratio']:.0%} of train")
        if st['years'] < 2:
            report.append(f"    ⚠ SHORT DATA: only {st['years']} years — "
                          f"insufficient seasonal coverage in 60% train")

    # ================================================================
    # 5. ML PERFORMANCE (all scenarios)
    # ================================================================
    report.append(f"\n{'='*50}")
    report.append("5. ML PERFORMANCE ON OUTLIER BUILDINGS (t+1)")
    report.append(f"{'='*50}")

    result_files = {
        'Local MLP': 'local_mlp_matched_results_250.csv',
        'Local XGB': 'local_baseline_results_v2.csv',
        'Centr XGB': 'centralised_xgboost_results_v2.csv',
        'Centr MLP': 'centralised_mlp_results_v2.csv',
        'FedAvg': 'fl_fedavg_final_v4.csv',
        'FedAdam': 'fl_fedadam_final_v4.csv',
        'FedProx': 'fl_fedprox_final_v4.csv',
        'Pers FedAdam': 'fl_personalised_final_fedadam_v4.csv',
        'Pers FedProx': 'fl_personalised_final_fedprox_v4.csv',
    }

    # Build header line
    header = f"  {'Scenario':<16}"
    for bid in OUTLIERS:
        header += f" {anon_id(bid, mapping):>10}"
    report.append(header)
    report.append(f"  {'-'*56}")

    for name, fname in result_files.items():
        fpath = LOG_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        h1 = df[df['horizon'] == 1]
        line = f"  {name:<16}"
        for bid in OUTLIERS:
            bdf = h1[h1['building_id'] == bid]
            if len(bdf) > 0:
                line += f" {bdf['mae'].values[0]:>10.3f}"
            else:
                line += f" {'N/A':>10}"
        report.append(line)

    # ================================================================
    # 6. PEAK TIMING — WITH vs WITHOUT OUTLIERS
    # ================================================================
    report.append(f"\n{'='*50}")
    report.append("6. DOES REMOVING OUTLIERS CHANGE PEAK PATTERNS?")
    report.append(f"{'='*50}")

    p95_clean = port_clean.quantile(0.95)
    peak_all = port_all[port_all > p95]
    peak_clean = port_clean[port_clean > p95_clean]

    for label, peaks in [("All 250 buildings", peak_all),
                         ("246 buildings (no outliers)", peak_clean)]:
        morning = peaks[(peaks.index.hour >= 6) & (peaks.index.hour <= 10)]
        evening = peaks[(peaks.index.hour >= 16) & (peaks.index.hour <= 21)]
        peak_hour = peaks.groupby(peaks.index.hour).size().idxmax()
        winter_pct = peaks[peaks.index.month.isin([12, 1, 2])].shape[0] / len(peaks) * 100

        report.append(f"\n  {label}:")
        report.append(f"    Peak hour:      {peak_hour}:00")
        report.append(f"    Morning share:  {len(morning)/len(peaks)*100:.1f}%")
        report.append(f"    Evening share:  {len(evening)/len(peaks)*100:.1f}%")
        report.append(f"    Winter share:   {winter_pct:.1f}%")

    # ================================================================
    # 7. COMPARISON WITH NORMAL BUILDINGS
    # ================================================================
    report.append(f"\n{'='*50}")
    report.append("7. OUTLIERS vs NORMAL BUILDINGS")
    report.append(f"{'='*50}")

    normal_ids = [b for b in all_series if b not in OUTLIERS]
    normal_means = [all_series[b].mean() for b in normal_ids]
    normal_stds = [all_series[b].std() for b in normal_ids]
    normal_peaks = [all_series[b].reindex(peak_hours).mean() for b in normal_ids]

    outlier_means = [all_series[b].mean() for b in OUTLIERS if b in all_series]
    outlier_stds = [all_series[b].std() for b in OUTLIERS if b in all_series]
    outlier_peaks_list = [all_series[b].reindex(peak_hours).mean() for b in OUTLIERS if b in all_series]

    report.append(f"\n  {'Metric':<25} {'Outliers (4)':>15} {'Normal (246)':>15} {'Ratio':>8}")
    report.append(f"  {'-'*65}")
    report.append(f"  {'Mean consumption':.<25} {np.mean(outlier_means):>15.1f} {np.mean(normal_means):>15.1f} "
                  f"{np.mean(outlier_means)/np.mean(normal_means):>7.1f}x")
    report.append(f"  {'Std consumption':.<25} {np.mean(outlier_stds):>15.1f} {np.mean(normal_stds):>15.1f} "
                  f"{np.mean(outlier_stds)/np.mean(normal_stds):>7.1f}x")
    report.append(f"  {'Peak contribution':.<25} {np.mean(outlier_peaks_list):>15.1f} {np.mean(normal_peaks):>15.1f} "
                  f"{np.mean(outlier_peaks_list)/np.mean(normal_peaks):>7.1f}x")
    report.append(f"  {'Max consumption':.<25} {np.max([all_series[b].max() for b in OUTLIERS]):>15.1f} "
                  f"{np.max([all_series[b].max() for b in normal_ids]):>15.1f}")

    # ================================================================
    # SAVE REPORT
    # ================================================================
    report_text = '\n'.join(report)
    report_path = LOG_DIR / "outlier_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(report_text, flush=True)
    print(f"\nReport saved: {report_path}", flush=True)

    # Save stats CSV
    stats_df = pd.DataFrame(stats_list)
    stats_path = LOG_DIR / "outlier_analysis_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Stats saved:  {stats_path}", flush=True)

    # ================================================================
    # FIGURES
    # ================================================================
    print("\nGenerating outlier figures...", flush=True)

    # Figure A: Consumption timeline with split boundaries
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    split_colors = {
        'train': '#2166AC', 'adapt': '#66BD63',
        'val': '#FEE08B', 'test': '#D73027',
    }

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.5), sharex=False)
    for i, bid in enumerate(OUTLIERS):
        ax = axes[i // 2][i % 2]
        s = all_series[bid]
        n = len(s)
        train_end = int(n * 0.60)
        adapt_end = train_end + int(n * 0.15)
        val_end = adapt_end + int(n * 0.10)

        # Plot weekly rolling mean for clarity
        weekly = s.resample('W').mean()
        ax.plot(weekly.index, weekly.values, color='#333', linewidth=0.8, zorder=3)

        # Shade splits with stronger alpha
        ax.axvspan(s.index[0], s.index[train_end - 1],
                   alpha=0.20, color=split_colors['train'])
        ax.axvspan(s.index[train_end], s.index[adapt_end - 1],
                   alpha=0.25, color=split_colors['adapt'])
        ax.axvspan(s.index[adapt_end], s.index[val_end - 1],
                   alpha=0.25, color=split_colors['val'])
        ax.axvspan(s.index[val_end], s.index[-1],
                   alpha=0.20, color=split_colors['test'])

        ratio = stats_list[i]['test_train_ratio']
        ax.set_title(f"{anon_id(bid, mapping)}  —  "
                     f"test/train = {ratio:.2f}×, {stats_list[i]['years']}y data",
                     fontsize=8, fontweight='bold')
        ax.set_ylabel('kWh/h', fontsize=7)
        ax.tick_params(labelsize=6)

        # Fix x-axis: yearly ticks, short date format
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=0)

    # Shared legend at bottom
    legend_elements = [
        Patch(facecolor=split_colors['train'], alpha=0.35, label='Train (60%)'),
        Patch(facecolor=split_colors['adapt'], alpha=0.40, label='Adapt (15%)'),
        Patch(facecolor=split_colors['val'], alpha=0.40, label='Val (10%)'),
        Patch(facecolor=split_colors['test'], alpha=0.35, label='Test (15%)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=7, frameon=True, framealpha=0.9, edgecolor='none',
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    outpath = FIG_DIR / "outlier_timelines.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)

    # Figure B: Outlier vs normal comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    # (a) Mean consumption
    categories = ['Mean\nconsumption', 'Peak\nconsumption', 'Std\nconsumption']
    outlier_vals = [np.mean(outlier_means), np.mean(outlier_peaks_list), np.mean(outlier_stds)]
    normal_vals = [np.mean(normal_means), np.mean(normal_peaks), np.mean(normal_stds)]

    x = np.arange(len(categories))
    w = 0.35
    ax1.bar(x - w / 2, outlier_vals, w, color=COLORS['outlier'], alpha=0.8, label='Outliers (4)')
    ax1.bar(x + w / 2, normal_vals, w, color=COLORS['normal'], alpha=0.8, label='Normal (246)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=7)
    ax1.set_ylabel('kWh/h')
    ax1.legend(fontsize=7)
    ax1.set_title('(a) Consumption comparison', fontsize=9)

    for i in range(len(categories)):
        ratio = outlier_vals[i] / normal_vals[i] if normal_vals[i] > 0 else 0
        y = max(outlier_vals[i], normal_vals[i]) + 0.5
        ax1.text(x[i], y, f"{ratio:.1f}×", ha='center', fontsize=7, fontweight='bold', color='#555')

    # (b) Train vs test consumption ratio
    bids_anon = [anon_id(bid, mapping) for bid in OUTLIERS]
    train_means = [st['train_mean'] for st in stats_list]
    test_means = [st['test_mean'] for st in stats_list]

    x2 = np.arange(len(bids_anon))
    ax2.bar(x2 - w / 2, train_means, w, color=COLORS['train'], alpha=0.8, label='Train mean')
    ax2.bar(x2 + w / 2, test_means, w, color=COLORS['test'], alpha=0.8, label='Test mean')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(bids_anon, fontsize=7)
    ax2.set_ylabel('kWh/h')
    ax2.legend(fontsize=7)
    ax2.set_title('(b) Train vs test regime change', fontsize=9)

    for i in range(len(bids_anon)):
        ratio = stats_list[i]['test_train_ratio']
        y = max(train_means[i], test_means[i]) + 0.5
        ax2.text(x2[i], y, f"{ratio:.2f}×", ha='center', fontsize=7, fontweight='bold',
                 color=COLORS['outlier'])

    fig.tight_layout()
    outpath = FIG_DIR / "outlier_comparison.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)

    print(f"\nDone. All outputs in {LOG_DIR} and {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()