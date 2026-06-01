"""
DAS-FL — Building Selection method
Scans all 100 FL buildings and ranks them by:
  - Mean hourly consumption (kWh/h)
  - Prediction accuracy (MAE, R²)
  - Diurnal variation (std of hourly pattern)
  - Peak-to-base ratio

Outputs a ranked table to help select buildings for:
  - 24h profile plots
  - Demand forecast demos
  - Case study examples in the paper

Run: PYTHONPATH=. python pipeline/select_buildings.py

Options:
  --min-consumption 5.0   Filter buildings above this mean kWh/h
  --top 10                Show top N buildings
  --sort-by consumption   Sort by: consumption, mae, r2, variation
  --export                Save results to CSV
"""

import pandas as pd
import numpy as np
import yaml
import sys
import os
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
DATA_DIR = PROJECT_ROOT / config["paths"]["processed_data"]


def load_fl_building_ids():
    """Load the 100 building IDs used in FL experiments."""
    fl_file = LOG_DIR / "fl_building_ids.txt"
    if fl_file.exists():
        return open(fl_file).read().strip().split("\n")
    else:
        print("ERROR: fl_building_ids.txt not found")
        sys.exit(1)


def get_building_stats(building_ids):
    """Compute statistics for each building from raw data and results."""
    print(f"  Scanning {len(building_ids)} buildings...")

    # Load prediction results
    results_files = {
        "local_mlp": LOG_DIR / "local_mlp_matched_results.csv",
        "local_xgb": LOG_DIR / "local_baseline_results_matched100.csv",
        "personalised": LOG_DIR / "fl_personalised_final.csv",
    }

    results = {}
    for name, path in results_files.items():
        if path.exists():
            df = pd.read_csv(path)
            df["building_id"] = df["building_id"].astype(str)
            results[name] = df
            print(f"    Loaded {name}: {len(df)} rows")
        else:
            print(f"    {name}: not found ({path.name})")

    # Build stats for each building
    rows = []
    for i, bid in enumerate(building_ids):
        stats = {"building_id": bid, "index": i}

        # --- Raw data stats ---
        pfile = DATA_DIR / f"{bid}.parquet"
        if pfile.exists():
            bdf = pd.read_parquet(pfile)

            if "kwh" in bdf.columns:
                kwh = bdf["kwh"]
                stats["mean_consumption"] = round(kwh.mean(), 2)
                stats["median_consumption"] = round(kwh.median(), 2)
                stats["max_consumption"] = round(kwh.max(), 2)
                stats["std_consumption"] = round(kwh.std(), 2)
                stats["total_hours"] = len(kwh)

                # Peak-to-base ratio (90th percentile / 10th percentile)
                p90 = kwh.quantile(0.90)
                p10 = kwh.quantile(0.10)
                stats["peak_to_base"] = round(p90 / max(p10, 0.01), 1)

                # Diurnal variation: std of mean hourly profile
                if "hour" in bdf.columns:
                    hourly_means = bdf.groupby("hour")["kwh"].mean()
                    stats["diurnal_variation"] = round(hourly_means.std(), 3)
                elif bdf.index.dtype == "datetime64[ns]" or hasattr(bdf.index, "hour"):
                    hourly_means = kwh.groupby(kwh.index.hour).mean()
                    stats["diurnal_variation"] = round(hourly_means.std(), 3)

                # Season check: winter mean vs summer mean
                if "month" in bdf.columns:
                    winter = bdf[bdf["month"].isin([12, 1, 2])]["kwh"].mean()
                    summer = bdf[bdf["month"].isin([6, 7, 8])]["kwh"].mean()
                    stats["winter_mean"] = round(winter, 2) if not np.isnan(winter) else None
                    stats["summer_mean"] = round(summer, 2) if not np.isnan(summer) else None
        else:
            stats["mean_consumption"] = None

        # --- Prediction accuracy stats (t+1) ---
        for name, df in results.items():
            bdf_r = df[(df["building_id"] == bid) &
                       (df["horizon"] == 1) &
                       (df["status"] == "success")]
            if len(bdf_r) > 0:
                row = bdf_r.iloc[0]
                stats[f"{name}_mae"] = round(row["mae"], 3)
                if "r2" in row:
                    stats[f"{name}_r2"] = round(row["r2"], 3)

        rows.append(stats)

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(building_ids)} scanned")

    return pd.DataFrame(rows)


def display_table(df, sort_by="mean_consumption", top=20, min_consumption=None):
    """Display a formatted table of building statistics."""

    # Filter
    if min_consumption is not None:
        df = df[df["mean_consumption"] >= min_consumption]
        print(f"\n  Filtered: {len(df)} buildings with mean consumption >= {min_consumption} kWh/h")

    # Sort
    sort_map = {
        "consumption": "mean_consumption",
        "mae": "local_mlp_mae",
        "r2": "local_mlp_r2",
        "variation": "diurnal_variation",
        "peak": "peak_to_base",
    }
    sort_col = sort_map.get(sort_by, sort_by)

    ascending = True if sort_by == "mae" else False
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")

    df_show = df.head(top)

    # Display
    print(f"\n{'='*100}")
    print(f"  TOP {min(top, len(df_show))} BUILDINGS (sorted by {sort_by})")
    print(f"{'='*100}")

    # Select columns to display
    display_cols = ["building_id", "index", "mean_consumption", "std_consumption",
                    "max_consumption", "peak_to_base"]

    # Add diurnal variation if available
    if "diurnal_variation" in df.columns:
        display_cols.append("diurnal_variation")

    # Add winter/summer if available
    if "winter_mean" in df.columns:
        display_cols.extend(["winter_mean", "summer_mean"])

    # Add prediction metrics
    for prefix in ["local_mlp", "personalised"]:
        mae_col = f"{prefix}_mae"
        r2_col = f"{prefix}_r2"
        if mae_col in df.columns:
            display_cols.append(mae_col)
        if r2_col in df.columns:
            display_cols.append(r2_col)

    # Filter to existing columns
    display_cols = [c for c in display_cols if c in df.columns]

    # Print header
    header = ""
    for col in display_cols:
        short = col.replace("mean_consumption", "mean_kwh") \
                   .replace("std_consumption", "std") \
                   .replace("max_consumption", "max") \
                   .replace("peak_to_base", "pk/base") \
                   .replace("diurnal_variation", "diurnal") \
                   .replace("winter_mean", "winter") \
                   .replace("summer_mean", "summer") \
                   .replace("local_mlp_mae", "mlp_mae") \
                   .replace("local_mlp_r2", "mlp_r2") \
                   .replace("personalised_mae", "pers_mae") \
                   .replace("personalised_r2", "pers_r2") \
                   .replace("building_id", "bldg_id")
        header += f"{short:>10}"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    # Print rows
    for _, row in df_show.iterrows():
        line = ""
        for col in display_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                line += f"{'---':>10}"
            elif isinstance(val, float):
                line += f"{val:>10.3f}"
            else:
                line += f"{str(val):>10}"
        print(f"  {line}")

    print(f"{'='*100}")

    # Recommendations
    print("\n  RECOMMENDATIONS FOR PAPER FIGURES:")
    print(f"  {'-'*50}")

    # Best for 24h profile: high consumption + good prediction + clear variation
    candidates = df[df["mean_consumption"] >= 5.0] if "mean_consumption" in df.columns else df
    if "diurnal_variation" in candidates.columns and len(candidates) > 0:
        candidates = candidates.sort_values("diurnal_variation", ascending=False)
        # Among top 10 by variation, pick best prediction accuracy
        top_varied = candidates.head(10)
        if "local_mlp_mae" in top_varied.columns:
            best_24h = top_varied.sort_values("local_mlp_mae").iloc[0]
            print(f"  24h profile:  Building {best_24h['building_id']} "
                  f"(mean={best_24h['mean_consumption']:.1f} kWh/h, "
                  f"diurnal={best_24h.get('diurnal_variation', 'N/A')}, "
                  f"MAE={best_24h.get('local_mlp_mae', 'N/A')})")

    # Best for demand demo: high consumption (shows the housing company value)
    if "mean_consumption" in df.columns and len(df) > 0:
        biggest = df.sort_values("mean_consumption", ascending=False).iloc[0]
        print(f"  Demand demo:  Building {biggest['building_id']} "
              f"(mean={biggest['mean_consumption']:.1f} kWh/h, "
              f"max={biggest.get('max_consumption', 'N/A')})")

    # Best for case study: median consumption + median accuracy
    if "mean_consumption" in df.columns and "local_mlp_mae" in df.columns and len(df) > 0:
        df_tmp = df.copy()
        df_tmp["combined_rank"] = (
            df_tmp["mean_consumption"].rank(pct=True).sub(0.5).abs() +
            df_tmp["local_mlp_mae"].rank(pct=True).sub(0.5).abs()
        )
        most_typical = df_tmp.sort_values("combined_rank").iloc[0]
        print(f"  Case study:   Building {most_typical['building_id']} "
              f"(mean={most_typical['mean_consumption']:.1f} kWh/h, "
              f"MAE={most_typical.get('local_mlp_mae', 'N/A')})")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAS-FL Building Selection Tool")
    parser.add_argument("--min-consumption", type=float, default=None,
                        help="Minimum mean consumption (kWh/h) filter")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N buildings (default: 20)")
    parser.add_argument("--sort-by", type=str, default="consumption",
                        choices=["consumption", "mae", "r2", "variation", "peak"],
                        help="Sort criteria (default: consumption)")
    parser.add_argument("--export", action="store_true",
                        help="Export full results to CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("DAS-FL — BUILDING SELECTION TOOL")
    print("=" * 60)

    building_ids = load_fl_building_ids()
    print(f"\n  FL buildings: {len(building_ids)}")

    # Compute stats
    print("\n[1/2] Computing building statistics...")
    stats_df = get_building_stats(building_ids)

    # Display
    print("\n[2/2] Ranking buildings...")
    result_df = display_table(
        stats_df,
        sort_by=args.sort_by,
        top=args.top,
        min_consumption=args.min_consumption,
    )

    # Export
    if args.export:
        out_path = LOG_DIR / "building_selection_stats.csv"
        stats_df.to_csv(out_path, index=False)
        print(f"\n  Exported: {out_path}")

    print(f"\n{'='*60}")
    print("BUILDING SELECTION COMPLETE")
    print(f"{'='*60}")
    print("""
Usage examples:
  # Show top 10 by consumption
  PYTHONPATH=. python pipeline/select_buildings.py --sort-by consumption --top 10

  # Filter to buildings > 5 kWh/h, sort by prediction accuracy
  PYTHONPATH=. python pipeline/select_buildings.py --min-consumption 5 --sort-by mae

  # Sort by diurnal variation (best for 24h profile plot)
  PYTHONPATH=. python pipeline/select_buildings.py --sort-by variation --top 10

  # Export all stats to CSV for further analysis
  PYTHONPATH=. python pipeline/select_buildings.py --export
""")