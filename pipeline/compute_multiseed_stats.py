#!/usr/bin/env python3
"""
compute_multiseed_stats.py — produce mean ± std summaries for camera-ready.

Reads per-seed result CSVs, applies the paper's standard filter
(horizon=1, status=success, drop 4 outlier buildings), and reports
mean / std / range across seeds. Designed for the SCAI 2026 paper.

Usage:
    python pipeline/compute_multiseed_stats.py \\
        --scenario centralised_mlp \\
        --files logs/centralised_mlp_results_v5_portfolio.csv \\
                logs/centralised_mlp_results_v5_seed43.csv \\
                logs/centralised_mlp_results_v5_seed44.csv \\
                logs/centralised_mlp_results_v5_seed45.csv \\
                logs/centralised_mlp_results_v5_seed46.csv

    # Or with glob:
    python pipeline/compute_multiseed_stats.py \\
        --scenario centralised_mlp \\
        --pattern "logs/centralised_mlp_results_v5_*.csv"

    # Multiple metrics:
    python pipeline/compute_multiseed_stats.py \\
        --scenario centralised_mlp \\
        --pattern "logs/centralised_mlp_results_v5_*.csv" \\
        --metrics mae r2

Notes:
    - The 4-outlier list (B001, B037, B238, B028) matches the
      paper's wilcoxon_test.py recipe.
    - Horizon filter is configurable via --horizon (default 1).
    - Output is a markdown table on stdout; optionally writes a JSON
      sidecar for downstream scripts.
"""

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.anonymise_buildings import real_ids

# Paper-standard outlier filter (mirrors pipeline/wilcoxon_test.py)
OUTLIER_BUILDINGS = {int(x) for x in real_ids("B001", "B037", "B238", "B028")}


def apply_paper_filter(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Apply the standard paper filter: horizon, status, outliers."""
    out = df.copy()
    if 'horizon' in out.columns:
        out = out[out['horizon'] == horizon]
    if 'status' in out.columns:
        out = out[out['status'] == 'success']
    if 'building_id' in out.columns:
        # Coerce both sides to string to dodge dtype mismatches
        outlier_strs = {str(b) for b in OUTLIER_BUILDINGS}
        out = out[~out['building_id'].astype(str).isin(outlier_strs)]
    return out


def label_from_filename(path: Path) -> str:
    """Extract a human-readable seed/run label from a filename."""
    stem = path.stem
    # Common patterns: ..._seed43, ..._v5_portfolio, ..._v5
    for marker in ['_seed', '_portfolio', '_v5']:
        if marker in stem:
            tail = stem.split(marker, 1)[1]
            if marker == '_seed':
                return f"seed={tail.strip('_')}"
            if marker == '_portfolio':
                return "seed=42 (portfolio)"
            if marker == '_v5':
                return "seed=42 (v5)"
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True,
                        help="Scenario name (e.g. centralised_mlp). Used in output only.")
    parser.add_argument("--files", nargs="*", default=[],
                        help="Explicit list of CSV files.")
    parser.add_argument("--pattern", default=None,
                        help="Glob pattern for CSV files (alternative to --files).")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forecast horizon to filter on (default 1).")
    parser.add_argument("--metrics", nargs="+", default=["mae"],
                        help="Metrics to summarise (default: mae). Common: mae r2 rmse mape.")
    parser.add_argument("--json-out", default=None,
                        help="Optional path to write JSON summary for downstream use.")
    args = parser.parse_args()

    # Resolve file list
    if args.pattern:
        files = sorted(Path(p) for p in glob.glob(args.pattern))
    else:
        files = [Path(f) for f in args.files]

    if not files:
        print("ERROR: no files matched. Provide --files or --pattern.", file=sys.stderr)
        sys.exit(1)

    # Per-file, per-metric stats
    per_seed = []
    print()
    print(f"Scenario: {args.scenario}  (horizon={args.horizon}, outliers dropped: {len(OUTLIER_BUILDINGS)})")
    print()
    header_metrics = "  ".join(f"{m:>10}" for m in args.metrics)
    print(f"{'Label':<25} {header_metrics}    {'N':>5}  Status")
    print("-" * (25 + len(header_metrics) + 20))

    for f in files:
        if not f.exists():
            print(f"{label_from_filename(f):<25} {'MISSING':>10}")
            continue
        df = pd.read_csv(f)
        df_filt = apply_paper_filter(df, horizon=args.horizon)

        row_metrics = {}
        for m in args.metrics:
            if m not in df_filt.columns:
                row_metrics[m] = None
            else:
                row_metrics[m] = df_filt[m].mean()

        per_seed.append({"label": label_from_filename(f), "file": str(f),
                         "n": len(df_filt), "metrics": row_metrics})

        values_str = "  ".join(
            f"{row_metrics[m]:>10.4f}" if row_metrics[m] is not None else f"{'—':>10}"
            for m in args.metrics
        )
        status = "OK" if len(df_filt) == 246 else f"WARN: N={len(df_filt)}"
        print(f"{label_from_filename(f):<25} {values_str}    {len(df_filt):>5}  {status}")

    print("-" * (25 + len(header_metrics) + 20))

    # Aggregate
    summary = {}
    for m in args.metrics:
        vals = [p["metrics"][m] for p in per_seed if p["metrics"][m] is not None]
        if len(vals) >= 2:
            summary[m] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals),
                "min": min(vals),
                "max": max(vals),
                "n_seeds": len(vals),
            }
        else:
            summary[m] = {"mean": vals[0] if vals else None, "std": None,
                          "min": None, "max": None, "n_seeds": len(vals)}

    # Aggregate line
    print()
    for m in args.metrics:
        s = summary[m]
        if s["std"] is not None:
            print(f"  {m}: mean ± std = {s['mean']:.4f} ± {s['std']:.4f}   "
                  f"range [{s['min']:.4f}, {s['max']:.4f}]   n={s['n_seeds']}")
        else:
            print(f"  {m}: single value = {s['mean']:.4f}   n={s['n_seeds']}")
    print()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"scenario": args.scenario,
                       "horizon": args.horizon,
                       "outliers_dropped": sorted(OUTLIER_BUILDINGS),
                       "per_seed": per_seed,
                       "summary": summary}, f, indent=2)
        print(f"JSON summary written to {args.json_out}")


if __name__ == "__main__":
    main()