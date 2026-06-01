#!/usr/bin/env python3
"""
Peak and Seasonality Analysis
===============================================
Analyses all 250 FL buildings for:
1. Seasonal consumption patterns
2. Hourly peak identification (winter morning/evening peaks)
3. Weekday vs weekend patterns
4. Portfolio-level network peak analysis
5. Swedish holiday impact
6. Top peak-contributing buildings
7. Peak concentration analysis (which buildings drive network peaks)

Output:
  - logs/peak_analysis_results.csv (per-building stats)
  - logs/portfolio_peak_analysis.txt (summary report)
  - Console output with key findings

Usage:
  cd ~/Desktop/varsha_projects/das-flwr-heatdemand
  python pipeline/analyze_peaks.py
  python pipeline/analyze_peaks.py --n 50  # first 50 buildings only
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.anonymise_buildings import load_mapping, anon_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

# Swedish public holidays (month-day format)
SWEDISH_HOLIDAYS = {
    'New Year': ['01-01'],
    'Epiphany': ['01-06'],
    'May Day': ['05-01'],
    'National Day': ['06-06'],
    'Midsummer (approx)': ['06-20', '06-21', '06-22', '06-23', '06-24', '06-25'],
    'Christmas': ['12-24', '12-25', '12-26'],
    'New Year Eve': ['12-31'],
}

SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11],
}


def load_building_ids(ids_file=None):
    """Load FL building IDs."""
    if ids_file and os.path.exists(ids_file):
        return pd.read_csv(ids_file, header=None)[0].tolist()
    return None


def analyze_building(bid, data_dir):
    """Analyse one building's consumption patterns."""
    fpath = os.path.join(data_dir, f'{int(bid)}.parquet')
    if not os.path.exists(fpath):
        return None

    df = pd.read_parquet(fpath, columns=['kwh'])

    stats = {'building_id': int(bid)}

    # Basic stats
    stats['mean_kwh'] = df['kwh'].mean()
    stats['max_kwh'] = df['kwh'].max()
    stats['std_kwh'] = df['kwh'].std()
    stats['total_hours'] = len(df)

    # Seasonal means
    for season_name, months in SEASONS.items():
        season_data = df[df.index.month.isin(months)]['kwh']
        stats[f'{season_name.lower()}_mean'] = season_data.mean() if len(season_data) > 0 else np.nan

    # Winter/Summer ratio
    if stats['summer_mean'] > 0:
        stats['winter_summer_ratio'] = stats['winter_mean'] / stats['summer_mean']
    else:
        stats['winter_summer_ratio'] = np.nan

    # Hourly peaks (winter only)
    winter = df[df.index.month.isin([12, 1, 2])]
    if len(winter) > 100:
        hourly = winter.groupby(winter.index.hour)['kwh'].mean()
        stats['winter_peak_hour'] = int(hourly.idxmax())
        stats['winter_peak_kwh'] = hourly.max()
        stats['winter_offpeak_hour'] = int(hourly.idxmin())
        stats['winter_offpeak_kwh'] = hourly.min()
        stats['winter_peak_ratio'] = hourly.max() / hourly.min() if hourly.min() > 0 else np.nan

    # Weekday vs weekend
    weekday = df[df.index.dayofweek < 5]['kwh'].mean()
    weekend = df[df.index.dayofweek >= 5]['kwh'].mean()
    stats['weekday_mean'] = weekday
    stats['weekend_mean'] = weekend
    stats['weekday_weekend_ratio'] = weekday / weekend if weekend > 0 else np.nan

    return stats, df['kwh']


def portfolio_analysis(all_series, building_stats_df, output_file, mapping=None):
    """Network-level peak analysis across all buildings."""
    report = []
    report.append("=" * 70)
    report.append("PORTFOLIO PEAK AND SEASONALITY ANALYSIS")
    report.append(f"Buildings analysed: {len(all_series)}")
    report.append("=" * 70)

    # Aggregate portfolio consumption
    portfolio = pd.DataFrame(all_series).sum(axis=1)
    portfolio = portfolio.dropna()

    # Basic portfolio stats
    report.append(f"\n--- Portfolio consumption ---")
    report.append(f"  Mean:  {portfolio.mean():.1f} kWh/h")
    report.append(f"  Max:   {portfolio.max():.1f} kWh/h")
    report.append(f"  P95:   {portfolio.quantile(0.95):.1f} kWh/h")
    report.append(f"  P99:   {portfolio.quantile(0.99):.1f} kWh/h")

    # Peak hours (>P95)
    p95 = portfolio.quantile(0.95)
    peak_hours = portfolio[portfolio > p95]
    report.append(f"  Peak hours (>P95): {len(peak_hours)}")

    # 1. SEASONAL PATTERN
    report.append(f"\n--- Seasonal portfolio consumption ---")
    for season_name, months in SEASONS.items():
        season = portfolio[portfolio.index.month.isin(months)]
        if len(season) > 0:
            report.append(f"  {season_name:<8}: mean={season.mean():.1f}, "
                          f"max={season.max():.1f}, "
                          f"std={season.std():.1f}")

    # 2. MONTHLY PATTERN
    report.append(f"\n--- Monthly consumption ---")
    monthly = portfolio.groupby(portfolio.index.month).agg(['mean', 'max'])['mean']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m in range(1, 13):
        if m in monthly.index:
            val = monthly[m]
            bar = '█' * int(val / portfolio.mean() * 20)
            report.append(f"  {month_names[m-1]}: {val:>8.1f} {bar}")

    # 3. PEAK HOURS BY MONTH
    report.append(f"\n--- Peak hours (>P95) by month ---")
    peak_months = peak_hours.groupby(peak_hours.index.month).size()
    for m in range(1, 13):
        count = peak_months.get(m, 0)
        pct = count / len(peak_hours) * 100 if len(peak_hours) > 0 else 0
        bar = '█' * int(pct)
        report.append(f"  {month_names[m-1]}: {count:>5} ({pct:>5.1f}%) {bar}")

    # 4. PEAK HOURS BY HOUR OF DAY
    report.append(f"\n--- Peak hours by time of day ---")
    peak_hod = peak_hours.groupby(peak_hours.index.hour).size()
    for h in range(24):
        count = peak_hod.get(h, 0)
        pct = count / len(peak_hours) * 100 if len(peak_hours) > 0 else 0
        bar = '█' * int(pct)
        report.append(f"  {h:>2}:00  {count:>4} ({pct:>5.1f}%) {bar}")

    # 5. WEEKDAY VS WEEKEND PEAKS
    weekday_peaks = peak_hours[peak_hours.index.dayofweek < 5]
    weekend_peaks = peak_hours[peak_hours.index.dayofweek >= 5]
    report.append(f"\n--- Peak distribution weekday vs weekend ---")
    report.append(f"  Weekday peaks: {len(weekday_peaks)} "
                  f"({len(weekday_peaks)/len(peak_hours)*100:.1f}%)")
    report.append(f"  Weekend peaks: {len(weekend_peaks)} "
                  f"({len(weekend_peaks)/len(peak_hours)*100:.1f}%)")

    # 6. SWEDISH HOLIDAYS
    report.append(f"\n--- Swedish holiday impact on consumption ---")
    for name, dates in SWEDISH_HOLIDAYS.items():
        mask = portfolio.index.strftime('%m-%d').isin(dates)
        if mask.any():
            hol_mean = portfolio[mask].mean()
            months = list(set(int(d.split('-')[0]) for d in dates))
            normal_mask = portfolio.index.month.isin(months) & ~mask
            normal_mean = portfolio[normal_mask].mean()
            if normal_mean > 0:
                diff_pct = (hol_mean - normal_mean) / normal_mean * 100
                report.append(f"  {name:<25}: {hol_mean:>8.1f} vs {normal_mean:>8.1f} "
                              f"({diff_pct:>+6.1f}%)")

    # 7. TOP PEAK CONTRIBUTORS
    report.append(f"\n--- Top 20 peak-contributing buildings ---")
    report.append(f"  (Ranked by mean consumption during network peak hours)")

    # Find each building's contribution during network peak hours
    peak_timestamps = peak_hours.index
    contributions = []
    for bid, series in all_series.items():
        peak_consumption = series.reindex(peak_timestamps).mean()
        total_mean = series.mean()
        contributions.append({
            'building_id': int(bid),
            'peak_mean': peak_consumption,
            'total_mean': total_mean,
            'peak_share_pct': peak_consumption / portfolio.reindex(peak_timestamps).mean() * 100 
                if portfolio.reindex(peak_timestamps).mean() > 0 else 0,
        })

    contrib_df = pd.DataFrame(contributions).sort_values('peak_mean', ascending=False)
    report.append(f"  {'Rank':<5} {'Building':<12} {'Peak Mean':>10} {'Total Mean':>11} "
                  f"{'Peak Share':>11} {'Peak/Avg':>9}")
    for i, (_, row) in enumerate(contrib_df.head(20).iterrows()):
        ratio = row['peak_mean'] / row['total_mean'] if row['total_mean'] > 0 else 0
        label = anon_id(row['building_id'], mapping)
        report.append(f"  {i+1:<5} {label:<12} "
                      f"{row['peak_mean']:>10.2f} {row['total_mean']:>11.2f} "
                      f"{row['peak_share_pct']:>10.1f}% {ratio:>8.2f}x")

    # Peak concentration
    top5_share = contrib_df.head(5)['peak_share_pct'].sum()
    top10_share = contrib_df.head(10)['peak_share_pct'].sum()
    top20_share = contrib_df.head(20)['peak_share_pct'].sum()
    report.append(f"\n  Peak concentration:")
    report.append(f"    Top 5 buildings:  {top5_share:.1f}% of network peak")
    report.append(f"    Top 10 buildings: {top10_share:.1f}% of network peak")
    report.append(f"    Top 20 buildings: {top20_share:.1f}% of network peak")

    # 8. MORNING vs EVENING PEAKS (winter)
    winter_peak = peak_hours[peak_hours.index.month.isin([12, 1, 2])]
    if len(winter_peak) > 0:
        morning = winter_peak[(winter_peak.index.hour >= 6) & (winter_peak.index.hour <= 10)]
        evening = winter_peak[(winter_peak.index.hour >= 16) & (winter_peak.index.hour <= 21)]
        night = winter_peak[(winter_peak.index.hour >= 22) | (winter_peak.index.hour <= 5)]
        report.append(f"\n--- Winter peak timing ---")
        report.append(f"  Morning (06-10): {len(morning)} "
                      f"({len(morning)/len(winter_peak)*100:.1f}%)")
        report.append(f"  Evening (16-21): {len(evening)} "
                      f"({len(evening)/len(winter_peak)*100:.1f}%)")
        report.append(f"  Night (22-05):   {len(night)} "
                      f"({len(night)/len(winter_peak)*100:.1f}%)")

    # Save report
    report_text = '\n'.join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)
    print(report_text)

    return contrib_df


def main():
    parser = argparse.ArgumentParser(description="Peak and seasonality analysis")
    parser.add_argument("--n", type=int, default=None,
                        help="Analyse first N buildings (default: all 250)")
    parser.add_argument("--building-ids-file", type=str,
                        default=str(LOG_DIR / "fl_building_ids_250.txt"))
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("PEAK AND SEASONALITY ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    # Load anonymisation mapping
    mapping = load_mapping()

    # Load building IDs
    building_ids = load_building_ids(args.building_ids_file)
    if building_ids is None:
        print("ERROR: Building IDs file not found", flush=True)
        sys.exit(1)

    if args.n:
        building_ids = building_ids[:args.n]
    print(f"Analysing {len(building_ids)} buildings...", flush=True)

    # Analyse each building
    all_stats = []
    all_series = {}
    for i, bid in enumerate(building_ids):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(building_ids)} done...", flush=True)
        result = analyze_building(bid, DATA_DIR)
        if result is not None:
            stats, series = result
            all_stats.append(stats)
            all_series[bid] = series

    # Save per-building stats
    stats_df = pd.DataFrame(all_stats)
    stats_path = LOG_DIR / "peak_analysis_results.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nPer-building stats saved to: {stats_path}", flush=True)

    # Building-level summary
    print(f"\n--- Building-level summary ---", flush=True)
    print(f"  Mean winter/summer ratio: {stats_df['winter_summer_ratio'].mean():.1f}x", flush=True)
    print(f"  Mean peak hour: {stats_df['winter_peak_hour'].mode().values[0]}:00", flush=True)
    print(f"  Mean weekday/weekend ratio: {stats_df['weekday_weekend_ratio'].mean():.2f}x", flush=True)

    # Portfolio analysis
    report_path = LOG_DIR / "portfolio_peak_analysis.txt"
    contrib_df = portfolio_analysis(all_series, stats_df, report_path, mapping)

    # Save contributions
    contrib_path = LOG_DIR / "peak_contributions.csv"
    contrib_df.to_csv(contrib_path, index=False)
    print(f"\nPeak contributions saved to: {contrib_path}", flush=True)
    print(f"Full report saved to: {report_path}", flush=True)


if __name__ == "__main__":
    main()