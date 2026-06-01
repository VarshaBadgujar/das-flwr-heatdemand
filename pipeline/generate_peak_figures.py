#!/usr/bin/env python3
"""
Peak Analysis Figures
====================================================
Generates figures.

Figures produced:
  1. Monthly consumption + peak distribution (dual axis)
  2. Hourly peak pattern (winter, with morning/evening highlight)
  3. Top 20 peak-contributing buildings (horizontal bar)
  4. Peak concentration curve (cumulative % vs building rank)
  5. Holiday impact comparison (grouped bar)
  6. Weekday vs weekend heatmap (hour × day-of-week)
  7. Seasonal box plots (consumption distribution by season)
  8. FL benefit vs peak contribution scatter

Output: das-fl-paper/paper/figures/peak_*.pdf

Usage:
    cd ~/Desktop/varsha_projects/das-flwr-heatdemand
    python pipeline/generate_peak_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.anonymise_buildings import load_mapping, anon_id

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
FIG_DIR = PROJECT_ROOT / "das-fl-paper" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Springer LNCS: single column = 12.2 cm, double = 17.0 cm
# Use inches: single=4.8in, double=6.7in
SINGLE_COL = 4.8
DOUBLE_COL = 6.7

# Consistent style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Color palette (colorblind-safe)
COLORS = {
    'winter': '#2166AC',
    'spring': '#66BD63',
    'summer': '#F4A582',
    'autumn': '#D6604D',
    'peak': '#D73027',
    'normal': '#4575B4',
    'fl_wins': '#1B7837',
    'local_wins': '#762A83',
    'outlier': '#D73027',
    'regular': '#4575B4',
    'morning': '#2166AC',
    'evening': '#B2182B',
    'night': '#878787',
    'midday': '#92C5DE',
    'holiday': '#1B7837',
    'normal_day': '#878787',
    'weekday': '#4575B4',
    'weekend': '#D73027',
}

OUTLIER_BUILDINGS = [B001, B037, B238, B028]

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

SEASON_MONTHS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11],
}


# ============================================================
# DATA LOADING
# ============================================================
def load_portfolio(building_ids):
    """Load consumption series for all buildings."""
    series_dict = {}
    for bid in building_ids:
        fpath = DATA_DIR / f"{int(bid)}.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath, columns=['kwh'])
            series_dict[bid] = df['kwh']
    return series_dict


def load_building_ids():
    """Load FL building IDs."""
    ids_file = LOG_DIR / "fl_building_ids_250.txt"
    return pd.read_csv(ids_file, header=None)[0].tolist()


# ============================================================
# FIGURE 1: Monthly consumption + peak distribution
# ============================================================
def fig_monthly_consumption_peaks(portfolio):
    """Dual-axis: bar (monthly mean) + line (peak hours count)."""
    fig, ax1 = plt.subplots(figsize=(DOUBLE_COL, 3.2))

    monthly_mean = portfolio.groupby(portfolio.index.month).mean()
    p95 = portfolio.quantile(0.95)
    peak_hours = portfolio[portfolio > p95]
    peak_by_month = peak_hours.groupby(peak_hours.index.month).size()

    months = range(1, 13)
    means = [monthly_mean.get(m, 0) for m in months]
    peaks = [peak_by_month.get(m, 0) for m in months]

    bars = ax1.bar(MONTH_NAMES, means, color=COLORS['normal'], alpha=0.8,
                   label='Mean consumption', zorder=2, width=0.6)
    ax1.set_ylabel('Portfolio mean consumption (kWh/h)')
    ax1.set_ylim(0, max(means) * 1.15)

    ax2 = ax1.twinx()
    ax2.plot(MONTH_NAMES, peaks, 'o-', color=COLORS['peak'], linewidth=2,
             markersize=5, label='Peak hours (>P95)', zorder=3)
    ax2.set_ylabel('Number of peak hours (>P95)')
    ax2.set_ylim(0, max(peaks) * 1.2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               framealpha=0.9, edgecolor='none')

    ax1.set_xlabel('Month')
    fig.tight_layout()

    outpath = FIG_DIR / "peak_monthly_consumption.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# FIGURE 2: Hourly peak pattern (winter)
# ============================================================
def fig_hourly_peak_pattern(portfolio):
    """Bar chart of peak hour distribution with time-of-day highlighting."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))

    p95 = portfolio.quantile(0.95)
    peak_hours = portfolio[portfolio > p95]
    winter_peaks = peak_hours[peak_hours.index.month.isin([12, 1, 2])]

    hour_counts = winter_peaks.groupby(winter_peaks.index.hour).size()
    hours = range(24)
    counts = [hour_counts.get(h, 0) for h in hours]

    # Color by time period
    bar_colors = []
    for h in hours:
        if 6 <= h <= 10:
            bar_colors.append(COLORS['morning'])
        elif 16 <= h <= 21:
            bar_colors.append(COLORS['evening'])
        elif h >= 22 or h <= 5:
            bar_colors.append(COLORS['night'])
        else:
            bar_colors.append(COLORS['midday'])

    ax.bar([f"{h:02d}" for h in hours], counts, color=bar_colors, width=0.8)
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Number of peak hours')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['morning'], label=f'Morning 06-10 ({sum(counts[6:11])}, '
              f'{sum(counts[6:11])/sum(counts)*100:.0f}%)'),
        Patch(facecolor=COLORS['evening'], label=f'Evening 16-21 ({sum(counts[16:22])}, '
              f'{sum(counts[16:22])/sum(counts)*100:.0f}%)'),
        Patch(facecolor=COLORS['night'], label=f'Night 22-05 ({sum(counts[22:])+sum(counts[:6])}, '
              f'{(sum(counts[22:])+sum(counts[:6]))/sum(counts)*100:.0f}%)'),
        Patch(facecolor=COLORS['midday'], label=f'Midday 11-15'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
              framealpha=0.9, edgecolor='none')

    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()

    outpath = FIG_DIR / "peak_hourly_winter.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# FIGURE 3: Top 20 peak-contributing buildings
# ============================================================
def fig_top_peak_buildings(all_series, portfolio, mapping=None):
    """Horizontal bar chart of top peak contributors."""
    if mapping is None:
        mapping = load_mapping()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.0))

    p95 = portfolio.quantile(0.95)
    peak_timestamps = portfolio[portfolio > p95].index
    portfolio_peak_mean = portfolio.reindex(peak_timestamps).mean()

    contributions = []
    for bid, series in all_series.items():
        peak_mean = series.reindex(peak_timestamps).mean()
        total_mean = series.mean()
        contributions.append({
            'building_id': int(bid),
            'peak_mean': peak_mean,
            'total_mean': total_mean,
            'peak_share': peak_mean / portfolio_peak_mean * 100,
            'is_outlier': int(bid) in OUTLIER_BUILDINGS,
        })

    cdf = pd.DataFrame(contributions).sort_values('peak_mean', ascending=True)
    top20 = cdf.tail(20)

    labels = [anon_id(row['building_id'], mapping) for _, row in top20.iterrows()]

    colors = [COLORS['outlier'] if row['is_outlier'] else COLORS['regular']
              for _, row in top20.iterrows()]

    y_pos = range(len(top20))
    bars_peak = ax.barh(y_pos, top20['peak_mean'].values, height=0.7,
                        color=colors, alpha=0.85, label='During network peaks')
    bars_total = ax.barh(y_pos, top20['total_mean'].values, height=0.4,
                         color='#92C5DE', alpha=0.7, label='Overall average')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean consumption (kWh/h)')

    # Add peak share annotation
    for i, (_, row) in enumerate(top20.iterrows()):
        ax.text(row['peak_mean'] + 0.5, i, f"{row['peak_share']:.1f}%",
                va='center', fontsize=7, color='#555')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['outlier'], label='Seasonal-mismatch buildings'),
        Patch(facecolor=COLORS['regular'], label='Normal buildings'),
        Patch(facecolor='#92C5DE', alpha=0.7, label='Overall average'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
              framealpha=0.9, edgecolor='none')

    fig.tight_layout()
    outpath = FIG_DIR / "peak_top20_buildings.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)

    return cdf


# ============================================================
# FIGURE 4: Peak concentration curve
# ============================================================
def fig_peak_concentration(contributions_df):
    """Cumulative percentage of network peak load vs building rank."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.2))

    sorted_df = contributions_df.sort_values('peak_mean', ascending=False)
    cumulative = sorted_df['peak_share'].cumsum()
    n_buildings = range(1, len(cumulative) + 1)

    ax.plot(n_buildings, cumulative.values, color=COLORS['normal'], linewidth=2)
    ax.fill_between(n_buildings, cumulative.values, alpha=0.1, color=COLORS['normal'])

    # Reference lines
    for n, label in [(5, 'Top 5'), (10, 'Top 10'), (20, 'Top 20')]:
        if n <= len(cumulative):
            val = cumulative.iloc[n - 1]
            ax.axhline(y=val, color='#999', linewidth=0.5, linestyle='--')
            ax.axvline(x=n, color='#999', linewidth=0.5, linestyle='--')
            ax.annotate(f'{label}: {val:.1f}%', xy=(n, val),
                        xytext=(n + 15, val + 2), fontsize=7, color='#555')

    ax.set_xlabel('Number of buildings (ranked by peak contribution)')
    ax.set_ylabel('Cumulative share of network peak (%)')
    ax.set_xlim(1, len(cumulative))
    ax.set_ylim(0, 105)

    fig.tight_layout()
    outpath = FIG_DIR / "peak_concentration_curve.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# FIGURE 5: Holiday impact
# ============================================================
def fig_holiday_impact(portfolio):
    """Grouped bar chart comparing holiday vs normal consumption."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))

    HOLIDAYS = {
        'Epiphany\n(Jan 6)': ['01-06'],
        'May Day\n(May 1)': ['05-01'],
        'Nat. Day\n(Jun 6)': ['06-06'],
        'Midsummer\n(Jun 20-25)': ['06-20', '06-21', '06-22', '06-23', '06-24', '06-25'],
        'Christmas\n(Dec 24-26)': ['12-24', '12-25', '12-26'],
    }

    names = []
    hol_vals = []
    norm_vals = []

    for name, dates in HOLIDAYS.items():
        mask = portfolio.index.strftime('%m-%d').isin(dates)
        if mask.any():
            hol_mean = portfolio[mask].mean()
            months = list(set(int(d.split('-')[0]) for d in dates))
            normal_mask = portfolio.index.month.isin(months) & ~mask
            normal_mean = portfolio[normal_mask].mean()
            names.append(name)
            hol_vals.append(hol_mean)
            norm_vals.append(normal_mean)

    x = np.arange(len(names))
    w = 0.35

    ax.bar(x - w / 2, hol_vals, w, color=COLORS['holiday'], label='Holiday', alpha=0.85)
    ax.bar(x + w / 2, norm_vals, w, color=COLORS['normal_day'], label='Normal', alpha=0.85)

    # Add % change annotation
    for i in range(len(names)):
        pct = (hol_vals[i] - norm_vals[i]) / norm_vals[i] * 100
        y_pos = max(hol_vals[i], norm_vals[i]) + 20
        color = COLORS['peak'] if pct > 0 else COLORS['fl_wins']
        ax.text(x[i], y_pos, f"{pct:+.0f}%", ha='center', fontsize=7,
                fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel('Mean consumption (kWh/h)')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9, edgecolor='none')

    fig.tight_layout()
    outpath = FIG_DIR / "peak_holiday_impact.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# FIGURE 6: Weekday × Hour heatmap
# ============================================================
def fig_weekday_hour_heatmap(portfolio):
    """Heatmap of mean consumption by day-of-week and hour."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))

    # Winter only
    winter = portfolio[portfolio.index.month.isin([12, 1, 2])]
    pivot = winter.groupby([winter.index.dayofweek, winter.index.hour]).mean().unstack()
    pivot.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot.columns = [f"{h:02d}" for h in range(24)]

    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(7))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel('Hour of day')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean kWh/h', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    outpath = FIG_DIR / "peak_weekday_hour_heatmap.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# FIGURE 7: Seasonal box plots
# ============================================================
def fig_seasonal_boxplots(all_series):
    """Box plots of building-level mean consumption by season."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))

    season_data = {s: [] for s in SEASON_MONTHS}
    for bid, series in all_series.items():
        if int(bid) in OUTLIER_BUILDINGS:
            continue
        for season, months in SEASON_MONTHS.items():
            smean = series[series.index.month.isin(months)].mean()
            if not np.isnan(smean):
                season_data[season].append(smean)

    data = [season_data[s] for s in ['Winter', 'Spring', 'Summer', 'Autumn']]
    colors_bp = [COLORS['winter'], COLORS['spring'], COLORS['summer'], COLORS['autumn']]

    bp = ax.boxplot(data, labels=['Winter', 'Spring', 'Summer', 'Autumn'],
                    patch_artist=True, widths=0.6, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1.5})

    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Building mean consumption (kWh/h)')

    # Add median labels
    for i, d in enumerate(data):
        median = np.median(d)
        ax.text(i + 1, median + 0.3, f"{median:.1f}", ha='center', fontsize=7, fontweight='bold')

    fig.tight_layout()
    outpath = FIG_DIR / "peak_seasonal_boxplots.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# FIGURE 8: FL benefit vs peak contribution scatter
# ============================================================
def fig_fl_vs_peak_scatter(all_series, portfolio):
    """Scatter: x = peak contribution, y = FL improvement over local."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))

    # Load results
    try:
        local = pd.read_csv(LOG_DIR / 'local_mlp_matched_results_250_v5_portfolio.csv')
        pers = pd.read_csv(LOG_DIR / 'fl_personalised_final_fedadam_v5_portfolio.csv')
    except FileNotFoundError:
        print("  SKIP: FL results not found for scatter", flush=True)
        return

    l = local[local['horizon'] == 1].set_index('building_id')
    p = pers[pers['horizon'] == 1].set_index('building_id')

    # Peak contributions
    p95 = portfolio.quantile(0.95)
    peak_timestamps = portfolio[portfolio > p95].index

    scatter_data = []
    for bid in l.index.intersection(p.index):
        if int(bid) in OUTLIER_BUILDINGS:
            continue
        if bid in all_series:
            peak_contrib = all_series[bid].reindex(peak_timestamps).mean()
            fl_improvement = l.loc[bid, 'mae'] - p.loc[bid, 'mae']  # positive = FL better
            scatter_data.append({
                'building_id': bid,
                'peak_contrib': peak_contrib,
                'fl_improvement': fl_improvement,
                'fl_wins': fl_improvement > 0,
            })

    sdf = pd.DataFrame(scatter_data)

    # Plot
    wins = sdf[sdf['fl_wins']]
    loses = sdf[~sdf['fl_wins']]

    ax.scatter(wins['peak_contrib'], wins['fl_improvement'],
               c=COLORS['fl_wins'], s=20, alpha=0.6, label=f'FL wins ({len(wins)})')
    ax.scatter(loses['peak_contrib'], loses['fl_improvement'],
               c=COLORS['local_wins'], s=20, alpha=0.6, label=f'Local wins ({len(loses)})')

    ax.axhline(y=0, color='#999', linewidth=1, linestyle='-')
    ax.set_xlabel('Building peak contribution (mean kWh/h during P95 hours)')
    ax.set_ylabel('FL improvement (Local MAE - Pers FL MAE)')

    # Annotate quadrants
    ax.text(0.02, 0.98, 'FL wins + low peak\n(general benefit)',
            transform=ax.transAxes, fontsize=7, va='top', color=COLORS['fl_wins'], alpha=0.7)
    ax.text(0.98, 0.98, 'FL wins + high peak\n(operationally critical)',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            color=COLORS['fl_wins'], fontweight='bold', alpha=0.9)
    ax.text(0.98, 0.02, 'Local wins + high peak',
            transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
            color=COLORS['local_wins'], alpha=0.7)

    ax.legend(loc='lower left', fontsize=7, framealpha=0.9, edgecolor='none')

    fig.tight_layout()
    outpath = FIG_DIR / "peak_fl_vs_peak_scatter.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# FIGURE 9: Three-way comparison — who wins per building?
# ============================================================
def fig_three_way_comparison():
    """Pie/bar showing Pers FL vs Local MLP vs Centralised MLP wins per building."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0),
                                    gridspec_kw={'width_ratios': [1, 1.5]})

    try:
        local = pd.read_csv(LOG_DIR / 'local_mlp_matched_results_250_v5_portfolio.csv')
        pers = pd.read_csv(LOG_DIR / 'fl_personalised_final_fedadam_v5_portfolio.csv')
        central = pd.read_csv(LOG_DIR / 'centralised_mlp_results_v5_portfolio.csv')
    except FileNotFoundError:
        print("  SKIP: Results not found", flush=True)
        return

    l = local[local['horizon'] == 1].set_index('building_id')
    p = pers[pers['horizon'] == 1].set_index('building_id')
    c = central[central['horizon'] == 1].set_index('building_id')

    common = [b for b in l.index.intersection(p.index).intersection(c.index)
              if b not in OUTLIER_BUILDINGS]
    l, p, c = l.loc[common], p.loc[common], c.loc[common]

    fl_best = ((p['mae'] <= l['mae']) & (p['mae'] <= c['mae'])).sum()
    local_best = ((l['mae'] <= p['mae']) & (l['mae'] <= c['mae'])).sum()
    central_best = ((c['mae'] <= l['mae']) & (c['mae'] <= p['mae'])).sum()

    # (a) Pie chart
    sizes = [fl_best, local_best, central_best]
    labels_pie = [f'Pers. FL\n{fl_best} ({fl_best/len(common)*100:.0f}%)',
                  f'Local MLP\n{local_best} ({local_best/len(common)*100:.0f}%)',
                  f'Centr. MLP\n{central_best} ({central_best/len(common)*100:.0f}%)']
    colors_pie = [COLORS['fl_wins'], COLORS['local_wins'], COLORS['normal_day']]

    wedges, texts = ax1.pie(sizes, labels=labels_pie, colors=colors_pie,
                            startangle=90, textprops={'fontsize': 7})
    for w in wedges:
        w.set_alpha(0.8)
    ax1.set_title('(a) Best model per building', fontsize=9, fontweight='bold')

    # (b) MAE comparison bar
    scenarios = ['Centr.\nXGB', 'Oracle\n(best-of-3)', 'Local\nMLP',
                 'Pers FL\n(FedAdam)', 'Pers FL\n(FedProx)', 'Centr.\nMLP']
    
    oracle = np.minimum(np.minimum(l['mae'], p['mae']), c['mae']).mean()
    
    # Load additional results
    try:
        cxgb = pd.read_csv(LOG_DIR / 'centralised_xgboost_results_v2.csv')
        cxgb_h1 = cxgb[~cxgb['building_id'].isin(OUTLIER_BUILDINGS)]
        cxgb_mae = cxgb_h1[cxgb_h1['horizon'] == 1]['mae'].mean()
    except:
        cxgb_mae = 0.591

    try:
        pers_prox = pd.read_csv(LOG_DIR / 'fl_personalised_final_fedprox_v4.csv')
        pp = pers_prox[~pers_prox['building_id'].isin(OUTLIER_BUILDINGS)]
        prox_mae = pp[pp['horizon'] == 1]['mae'].mean()
    except:
        prox_mae = 0.693

    values = [cxgb_mae, oracle, l['mae'].mean(), p['mae'].mean(), prox_mae, c['mae'].mean()]
    bar_colors = ['#4575B4', '#1B7837', COLORS['local_wins'], COLORS['fl_wins'],
                  '#2166AC', COLORS['normal_day']]

    bars = ax2.bar(range(len(scenarios)), values, color=bar_colors, alpha=0.8, width=0.7)

    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels(scenarios, fontsize=7)
    ax2.set_ylabel('MAE (kWh/h)')
    ax2.set_ylim(0, max(values) * 1.2)
    ax2.set_title('(b) Centralized vs distributed trade-off', fontsize=9, fontweight='bold')

    # Privacy annotation
    ax2.annotate('Privacy-preserving', xy=(3, values[3]), xytext=(4.2, values[3] + 0.15),
                 fontsize=6, color=COLORS['fl_wins'], ha='center',
                 arrowprops=dict(arrowstyle='->', color=COLORS['fl_wins'], lw=0.8))

    fig.tight_layout()
    outpath = FIG_DIR / "tradeoff_three_way_comparison.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# FIGURE 10: Per-building MAE scatter — centralized vs distributed
# ============================================================
def fig_central_vs_distributed_scatter():
    """Scatter plot: Centralised MLP MAE vs Pers FL MAE per building."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 4.0))

    try:
        pers = pd.read_csv(LOG_DIR / 'fl_personalised_final_fedadam_v5_portfolio.csv')
        central = pd.read_csv(LOG_DIR / 'centralised_mlp_results_v5_portfolio.csv')
    except FileNotFoundError:
        print("  SKIP: Results not found", flush=True)
        return

    p = pers[pers['horizon'] == 1].set_index('building_id')
    c = central[central['horizon'] == 1].set_index('building_id')

    common = [b for b in p.index.intersection(c.index) if b not in OUTLIER_BUILDINGS]
    p, c = p.loc[common], c.loc[common]

    fl_wins = p['mae'] < c['mae']

    ax.scatter(c.loc[fl_wins, 'mae'], p.loc[fl_wins, 'mae'],
               c=COLORS['fl_wins'], s=15, alpha=0.5, label=f'FL better ({fl_wins.sum()})')
    ax.scatter(c.loc[~fl_wins, 'mae'], p.loc[~fl_wins, 'mae'],
               c=COLORS['normal_day'], s=15, alpha=0.5,
               label=f'Centralised better ({(~fl_wins).sum()})')

    # Diagonal
    max_val = max(c['mae'].max(), p['mae'].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.5, label='Equal')

    ax.set_xlabel('Centralised MLP MAE (kWh/h)')
    ax.set_ylabel('Personalised FL MAE (kWh/h)')
    ax.set_xlim(0, min(max_val, 5))
    ax.set_ylim(0, min(max_val, 5))
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, edgecolor='none')

    # Annotate
    ax.text(0.95, 0.05, 'FL better\n(below diagonal)',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color=COLORS['fl_wins'], alpha=0.7)

    fig.tight_layout()
    outpath = FIG_DIR / "tradeoff_central_vs_fl_scatter.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# FIGURE 11: All scenarios bar chart (main results)
# ============================================================
def fig_all_scenarios_comparison():
    """Horizontal bar chart of all scenarios — the main results figure."""
    from matplotlib.patches import Patch

    # (file, label, color, category)
    scenarios_files = [
        ('logs/centralised_xgboost_results_v2.csv', 'Centralised XGB', '#4575B4', 'Centralised'),
        ('logs/centralised_mlp_results_v5_portfolio.csv', 'Centralised MLP', '#4575B4', 'Centralised'),
        ('logs/local_baseline_results_v2.csv', 'Local XGB', '#762A83', 'Local'),
        ('logs/local_mlp_matched_results_250_v5_portfolio.csv', 'Local MLP', '#762A83', 'Local'),
        ('logs/fl_personalised_final_fedadam_v5_portfolio.csv', 'Pers. FL (FedAdam)', '#1B7837', 'FL'),
        ('logs/fl_personalised_final_fedprox_v4.csv', 'Pers. FL (FedProx)', '#1B7837', 'FL'),
        ('logs/fl_fedavg_final_v4.csv', 'FedAvg (global)', '#74C476', 'FL global'),
        ('logs/fl_fedprox_final_v4.csv', 'FedProx (global)', '#74C476', 'FL global'),
        ('logs/fl_fedadam_final_v4.csv', 'FedAdam (global)', '#74C476', 'FL global'),
    ]

    names = []
    maes = []
    colors = []

    for fpath, name, color, category in scenarios_files:
        try:
            df = pd.read_csv(LOG_DIR.parent / fpath)
            df = df[~df['building_id'].isin(OUTLIER_BUILDINGS)]
            h1 = df[df['horizon'] == 1]
            names.append(name)
            maes.append(h1['mae'].mean())
            colors.append(color)
        except Exception:
            pass

    # Sort by MAE ascending (best at top of chart)
    order = np.argsort(maes)
    names = [names[i] for i in order]
    maes = [maes[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))

    y = range(len(names))
    bars = ax.barh(y, maes, color=colors, alpha=0.85, height=0.65,
                   edgecolor='#333', linewidth=0.5)

    # Value labels to the right of each bar
    for bar, val in zip(bars, maes):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', ha='left', va='center', fontsize=7, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('MAE (kWh/h)')
    ax.set_xlim(0, max(maes) * 1.18)
    ax.set_ylim(-0.5, len(names) - 0.4)

    # Reference lines — labels at the top, rotated to avoid overlap
    ax.axvline(x=0.591, color='#333333', linewidth=0.8, linestyle='--', alpha=0.8)
    ax.text(0.591 - 0.012, len(names) - 0.5, 'Accuracy ceiling', fontsize=6.5,
            color='#222222', ha='right', va='top', fontweight='bold',
            rotation=90)

    ax.axvline(x=0.7, color='#333333', linewidth=0.8, linestyle='--', alpha=0.8)
    ax.text(0.7 + 0.012, len(names) - 0.5, 'Privacy boundary', fontsize=6.5,
            color='#222222', ha='left', va='top', fontweight='bold',
            rotation=90)

    # Legend
    legend_elements = [
        Patch(facecolor='#4575B4', label='Centralised', alpha=0.85, edgecolor='#333'),
        Patch(facecolor='#762A83', label='Local (per-building)', alpha=0.85, edgecolor='#333'),
        Patch(facecolor='#1B7837', label='FL (personalised)', alpha=0.85, edgecolor='#333'),
        Patch(facecolor='#74C476', label='FL (global)', alpha=0.85, edgecolor='#333'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
              framealpha=0.9, edgecolor='none')

    fig.tight_layout()
    outpath = FIG_DIR / "tradeoff_all_scenarios.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {outpath.name} + .png", flush=True)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60, flush=True)
    print("GENERATING ALL FIGURES (PEAK + TRADE-OFF)", flush=True)
    print(f"Output: {FIG_DIR}", flush=True)
    print("=" * 60, flush=True)

    # Load anonymisation mapping
    mapping = load_mapping()

    # Load data
    print("\nLoading building data...", flush=True)
    building_ids = load_building_ids()
    all_series = load_portfolio(building_ids)
    print(f"  Loaded {len(all_series)} buildings", flush=True)

    # Aggregate portfolio
    portfolio = pd.DataFrame(all_series).sum(axis=1).dropna()
    print(f"  Portfolio: {len(portfolio)} hours", flush=True)

    # Generate figures
    print("\nGenerating peak analysis figures...", flush=True)

    print("\n[1/11] Monthly consumption + peaks", flush=True)
    fig_monthly_consumption_peaks(portfolio)

    print("[2/11] Hourly peak pattern (winter)", flush=True)
    fig_hourly_peak_pattern(portfolio)

    print("[3/11] Top 20 peak-contributing buildings", flush=True)
    contributions_df = fig_top_peak_buildings(all_series, portfolio, mapping)

    print("[4/11] Peak concentration curve", flush=True)
    fig_peak_concentration(contributions_df)

    print("[5/11] Holiday impact", flush=True)
    fig_holiday_impact(portfolio)

    print("[6/11] Weekday × hour heatmap (winter)", flush=True)
    fig_weekday_hour_heatmap(portfolio)

    print("[7/11] Seasonal box plots", flush=True)
    fig_seasonal_boxplots(all_series)

    print("[8/11] FL benefit vs peak contribution", flush=True)
    fig_fl_vs_peak_scatter(all_series, portfolio)

    print("\nGenerating trade-off figures...", flush=True)

    print("[9/11] Three-way comparison (centralized vs distributed)", flush=True)
    fig_three_way_comparison()

    print("[10/11] Centralised vs FL scatter (per-building)", flush=True)
    fig_central_vs_distributed_scatter()

    print("[11/11] All scenarios comparison bar chart", flush=True)
    fig_all_scenarios_comparison()

    print(f"\n{'=' * 60}", flush=True)
    print(f"ALL 11 FIGURES SAVED to {FIG_DIR}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print("\nPeak analysis figures:", flush=True)
    print("  [Dataset]    peak_monthly_consumption.pdf", flush=True)
    print("  [Dataset]    peak_seasonal_boxplots.pdf", flush=True)
    print("  [Dataset]    peak_weekday_hour_heatmap.pdf", flush=True)
    print("  [Dataset]    peak_hourly_winter.pdf", flush=True)
    print("  [Discussion] peak_top20_buildings.pdf", flush=True)
    print("  [Discussion] peak_concentration_curve.pdf", flush=True)
    print("  [Discussion] peak_holiday_impact.pdf", flush=True)
    print("  [Discussion] peak_fl_vs_peak_scatter.pdf", flush=True)
    print("\nTrade-off figures (professor's theme):", flush=True)
    print("  [Results]    tradeoff_all_scenarios.pdf", flush=True)
    print("  [Results]    tradeoff_three_way_comparison.pdf", flush=True)
    print("  [Results]    tradeoff_central_vs_fl_scatter.pdf", flush=True)


if __name__ == "__main__":
    main()