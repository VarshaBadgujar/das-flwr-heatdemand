"""
DAS-FL Project — Step 1: Build Features
Purpose:
    Process buildings through the full pipeline:
    load → clean → feature engineering → create targets → save.

    Processes buildings in parallel and saves feature-engineered
    DataFrames to data/processed/ as parquet files.

Output:
    - data/processed/{building_id}.parquet  (per building)
    - logs/feature_engineering_report.txt
    - logs/feature_summary.csv

Run from project root:
    python pipeline/build_features.py                  # all buildings
    python pipeline/build_features.py --n 20           # first 20 only
    python pipeline/build_features.py --buildings B001 REDACTED  # specific
"""

import pandas as pd
import numpy as np
import yaml
import sys
import argparse
import glob
import time
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import dasfl
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dasfl.data_utils import (
    load_outdoor_temperature,
    get_building_files,
    load_building,
    clean_building,
    build_features,
    create_target,
    prepare_building,
)

# ── Load config 
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RAW_DATA_PATH = PROJECT_ROOT / config["paths"]["raw_data"]
PROCESSED_PATH = PROJECT_ROOT / config["paths"]["processed_data"]
LOG_DIR = PROJECT_ROOT / config["paths"]["logs"]
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# Global: load weather data once
_outdoor_temp = None

def _get_outdoor_temp():
    """Load outdoor temperature data (cached after first call)."""
    global _outdoor_temp
    if _outdoor_temp is None:
        weather_path = PROJECT_ROOT / config["paths"].get("weather_file", "")
        if weather_path.exists():
            _outdoor_temp = load_outdoor_temperature(weather_path)
            print(f"  Loaded outdoor temperature: {len(_outdoor_temp)} hours")
        else:
            print(f"  WARNING: Weather file not found: {weather_path}")
            _outdoor_temp = False  # Mark as attempted but missing
    return _outdoor_temp if _outdoor_temp is not False else None


def process_single_building(filepath: Path) -> dict:
    """Process one building and save to parquet.

    Returns metadata dict with building stats.
    """
    building_id = filepath.stem
    out_path = PROCESSED_PATH / f"{building_id}.parquet"

    try:
        outdoor_temp = _get_outdoor_temp()
        df_feat, metadata = prepare_building(filepath, config, drop_na=True, outdoor_temp=outdoor_temp)

        if len(df_feat) == 0:
            return {**metadata, "status": "empty_after_processing", "error": None}

        # Save as parquet (much faster + smaller than pkl for tabular data)
        df_feat.to_parquet(out_path, index=True)

        return {**metadata, "status": "success", "error": None}

    except Exception as e:
        return {
            "building_id": building_id,
            "status": "error",
            "error": str(e),
            "n_raw": 0, "n_clean": 0, "n_features": 0,
            "n_final": 0, "n_dropped_na": 0,
            "date_min": None, "date_max": None,
            "n_columns": 0, "feature_names": [],
        }


def generate_report(results: list[dict], elapsed: float) -> str:
    """Generate a human-readable feature engineering report."""
    lines = []
    lines.append("=" * 70)
    lines.append("DAS-FL PROJECT — FEATURE ENGINEERING REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Processing time: {elapsed:.1f} seconds")
    lines.append("=" * 70)

    df_r = pd.DataFrame(results)
    n_total = len(df_r)
    n_success = (df_r["status"] == "success").sum()
    n_empty = (df_r["status"] == "empty_after_processing").sum()
    n_error = (df_r["status"] == "error").sum()

    lines.append(f"\n1. PROCESSING SUMMARY")
    lines.append(f"   Total buildings processed:  {n_total}")
    lines.append(f"   Successful:                 {n_success}")
    lines.append(f"   Empty after processing:     {n_empty}")
    lines.append(f"   Errors:                     {n_error}")

    if n_success > 0:
        df_ok = df_r[df_r["status"] == "success"]

        lines.append(f"\n2. DATA DIMENSIONS")
        lines.append(f"   Rows per building (median):   {df_ok['n_final'].median():.0f}")
        lines.append(f"   Rows per building (min):      {df_ok['n_final'].min():.0f}")
        lines.append(f"   Rows per building (max):      {df_ok['n_final'].max():.0f}")
        lines.append(f"   Total rows across buildings:  {df_ok['n_final'].sum():,.0f}")
        lines.append(f"   Features per building:        {df_ok['n_columns'].iloc[0]}")

        lines.append(f"\n3. DATA LOSS FROM PROCESSING")
        total_raw = df_ok["n_raw"].sum()
        total_final = df_ok["n_final"].sum()
        total_dropped = df_ok["n_dropped_na"].sum()
        pct_retained = total_final / total_raw * 100 if total_raw > 0 else 0
        lines.append(f"   Total raw rows:       {total_raw:,.0f}")
        lines.append(f"   Total final rows:     {total_final:,.0f}")
        lines.append(f"   Dropped (NaN/lag):    {total_dropped:,.0f}")
        lines.append(f"   Retention rate:       {pct_retained:.1f}%")

        lines.append(f"\n4. DATE RANGES")
        dates = df_ok["date_min"].dropna()
        if len(dates) > 0:
            lines.append(f"   Earliest start:  {dates.min()}")
            lines.append(f"   Latest end:      {df_ok['date_max'].dropna().max()}")

        lines.append(f"\n5. FEATURE LIST")
        # Get feature names from first successful building
        feat_names = df_ok.iloc[0]["feature_names"]
        target_cols = [f for f in feat_names if f.startswith("target_")]
        input_cols = [f for f in feat_names if not f.startswith("target_")]
        lines.append(f"   Input features ({len(input_cols)}):")
        # Group by type for readability
        raw = [f for f in input_cols if "_lag_" not in f
               and "_rolling_" not in f and "_sin" not in f
               and "_cos" not in f and "_diff_" not in f
               and "is_" not in f]
        lags = [f for f in input_cols if "_lag_" in f]
        rolling = [f for f in input_cols if "_rolling_" in f]
        temporal = [f for f in input_cols if "_sin" in f or "_cos" in f or "is_" in f]
        diffs = [f for f in input_cols if "_diff_" in f]

        lines.append(f"     Raw inputs ({len(raw)}):      {raw}")
        lines.append(f"     Lagged ({len(lags)}):          {lags[:5]} ...")
        lines.append(f"     Rolling ({len(rolling)}):      {rolling[:5]} ...")
        lines.append(f"     Temporal ({len(temporal)}):     {temporal}")
        lines.append(f"     Rate of change ({len(diffs)}): {diffs}")
        lines.append(f"   Target columns ({len(target_cols)}): {target_cols}")

    if n_error > 0:
        lines.append(f"\n6. ERRORS")
        for _, row in df_r[df_r["status"] == "error"].iterrows():
            lines.append(f"   {row['building_id']}: {row['error']}")

    lines.append(f"\n{'=' * 70}")
    lines.append("END OF REPORT")
    return "\n".join(lines)


# ── Main ───────

def main():
    parser = argparse.ArgumentParser(
        description="Build features for DAS-FL project buildings."
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Process only the first N buildings (for testing)."
    )
    parser.add_argument(
        "--buildings", nargs="+", type=str, default=None,
        help="Process specific building IDs (e.g., --buildings B001 REDACTED)."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DAS-FL PROJECT — STEP 1: BUILD FEATURES")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)

    # Check data path
    if not RAW_DATA_PATH.exists():
        print(f"\nERROR: Data path not found: {RAW_DATA_PATH}")
        sys.exit(1)

    # Determine which buildings to process
    if args.buildings:
        files = [RAW_DATA_PATH / f"{bid}.pkl" for bid in args.buildings]
        files = [f for f in files if f.exists()]
    else:
        files = get_building_files(RAW_DATA_PATH, config, "*.pkl")
        if args.n:
            files = files[:args.n]

    n_files = len(files)
    print(f"\n  Buildings to process: {n_files}")
    print(f"  Output directory: {PROCESSED_PATH}")

    # Process buildings
    print(f"\n[1/2] Processing buildings...")
    start_time = time.time()
    results = []

    for i, fpath in enumerate(files):
        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == n_files:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_files - i - 1) / rate if rate > 0 else 0
            print(f"  Processing {i+1}/{n_files}... "
                  f"({rate:.1f} buildings/sec, ETA: {eta:.0f}s)")

        result = process_single_building(fpath)
        results.append(result)

    elapsed = time.time() - start_time

    # Generate report
    print(f"\n[2/2] Generating report...")
    report = generate_report(results, elapsed)

    report_path = LOG_DIR / "feature_engineering_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Save summary CSV (without feature_names list for cleanliness)
    df_results = pd.DataFrame(results)
    csv_cols = [c for c in df_results.columns if c != "feature_names"]
    df_results[csv_cols].to_csv(LOG_DIR / "feature_summary.csv", index=False)

    print("\n" + report)

    # Quick validation: load one processed file and show shape
    success_files = list(PROCESSED_PATH.glob("*.parquet"))
    if success_files:
        sample = pd.read_parquet(success_files[0])
        print(f"\n  VALIDATION — Sample processed file: {success_files[0].name}")
        print(f"  Shape: {sample.shape}")
        print(f"  Columns: {list(sample.columns[:10])} ... ({len(sample.columns)} total)")
        print(f"  Date range: {sample.index.min()} to {sample.index.max()}")
        print(f"  Memory: {sample.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print(f"\n{'=' * 60}")
    print("FEATURE ENGINEERING COMPLETE")
    print(f"  Processed: {PROCESSED_PATH}/")
    print(f"  Report:    {report_path}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
