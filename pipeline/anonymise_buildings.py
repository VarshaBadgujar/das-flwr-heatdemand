#!/usr/bin/env python3
"""
Building ID Anonymisation Mapping
===================================
Creates a deterministic mapping from real building IDs to 
anonymised labels (B001, B002, ...) for use in paper figures, 
tables, and reports.

Mapping rules:
  - 250 FL buildings sorted numerically → B001 to B250
  - Consistent across ALL scripts (shared JSON file)
  - Real IDs kept in code/CSVs for cross-reference
  - Anonymised IDs used ONLY in figures and paper text

Output:
  logs/building_id_mapping.json

Usage:
    # Generate mapping
    python pipeline/anonymise_buildings.py

    # Show mapping for specific buildings
    python pipeline/anonymise_buildings.py --lookup B001 B037 B238 B028

    # Show all mappings
    python pipeline/anonymise_buildings.py --show-all

    # In other scripts, import the helper:
    from pipeline.anonymise_buildings import load_mapping, anon_id
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
MAPPING_FILE = LOG_DIR / "building_id_mapping.json"
FL_IDS_FILE = LOG_DIR / "fl_building_ids_250.txt"

# Known outlier buildings (for highlighting in output)
OUTLIER_BUILDINGS = [B001, B037, B238, B028]


def create_mapping():
    """Create deterministic mapping: sorted real IDs → B001, B002, ..."""
    if not FL_IDS_FILE.exists():
        print(f"ERROR: {FL_IDS_FILE} not found", flush=True)
        sys.exit(1)

    fl_ids = pd.read_csv(FL_IDS_FILE, header=None)[0].tolist()
    fl_ids_sorted = sorted([int(b) for b in fl_ids])

    mapping = {}
    for i, bid in enumerate(fl_ids_sorted):
        mapping[str(bid)] = f"B{i + 1:03d}"

    # Save
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=2)

    return mapping


def load_mapping():
    """Load existing mapping from JSON file."""
    if not MAPPING_FILE.exists():
        print("Mapping not found — creating...", flush=True)
        return create_mapping()

    with open(MAPPING_FILE) as f:
        return json.load(f)


def anon_id(bid, mapping=None):
    """
    Convert a real building ID to anonymised label.
    
    Usage:
        label = anon_id(B001)          # → "B042" (example)
        label = anon_id(B001, mapping)  # with pre-loaded mapping
    """
    if mapping is None:
        mapping = load_mapping()
    return mapping.get(str(int(bid)), f"B{int(bid)}")


def reverse_lookup(anon_label, mapping=None):
    """Convert anonymised label back to real ID."""
    if mapping is None:
        mapping = load_mapping()
    reverse = {v: k for k, v in mapping.items()}
    return reverse.get(anon_label, None)


def main():
    parser = argparse.ArgumentParser(description="Building ID anonymisation")
    parser.add_argument("--lookup", nargs='+', type=int,
                        help="Look up anonymised IDs for specific buildings")
    parser.add_argument("--reverse", nargs='+', type=str,
                        help="Reverse lookup: B001 → real ID")
    parser.add_argument("--show-all", action='store_true',
                        help="Show all mappings")
    parser.add_argument("--regenerate", action='store_true',
                        help="Force regenerate mapping file")
    args = parser.parse_args()

    # Create or load mapping
    if args.regenerate or not MAPPING_FILE.exists():
        mapping = create_mapping()
        print(f"Mapping created: {MAPPING_FILE}", flush=True)
    else:
        mapping = load_mapping()
        print(f"Mapping loaded: {MAPPING_FILE}", flush=True)

    print(f"Total buildings: {len(mapping)}", flush=True)

    # Lookup specific buildings
    if args.lookup:
        print(f"\n=== LOOKUP ===", flush=True)
        for bid in args.lookup:
            label = mapping.get(str(bid), "NOT FOUND")
            is_outlier = " ← OUTLIER" if bid in OUTLIER_BUILDINGS else ""
            print(f"  {bid} → {label}{is_outlier}", flush=True)
        return

    # Reverse lookup
    if args.reverse:
        print(f"\n=== REVERSE LOOKUP ===", flush=True)
        for label in args.reverse:
            real_id = reverse_lookup(label, mapping)
            if real_id:
                is_outlier = " ← OUTLIER" if int(real_id) in OUTLIER_BUILDINGS else ""
                print(f"  {label} → {real_id}{is_outlier}", flush=True)
            else:
                print(f"  {label} → NOT FOUND", flush=True)
        return

    # Show all or just summary
    if args.show_all:
        print(f"\n=== ALL MAPPINGS ===", flush=True)
        for real_id, anon in sorted(mapping.items(), key=lambda x: x[1]):
            is_outlier = " ← OUTLIER" if int(real_id) in OUTLIER_BUILDINGS else ""
            print(f"  {anon} → {real_id}{is_outlier}", flush=True)
    else:
        # Default: show outliers + summary
        print(f"\n=== OUTLIER BUILDINGS ===", flush=True)
        for bid in OUTLIER_BUILDINGS:
            label = mapping.get(str(bid), "NOT FOUND")
            print(f"  {bid} → {label}", flush=True)

        print(f"\n=== RANGE ===", flush=True)
        labels = sorted(mapping.values())
        print(f"  First: {labels[0]} (real: {reverse_lookup(labels[0], mapping)})", flush=True)
        print(f"  Last:  {labels[-1]} (real: {reverse_lookup(labels[-1], mapping)})", flush=True)

        print(f"\n=== USAGE IN OTHER SCRIPTS ===", flush=True)
        print(f"  from pipeline.anonymise_buildings import load_mapping, anon_id", flush=True)
        print(f"  mapping = load_mapping()", flush=True)
        print(f"  label = anon_id(B001, mapping)  # → {mapping.get('B001', '?')}", flush=True)


if __name__ == "__main__":
    main()