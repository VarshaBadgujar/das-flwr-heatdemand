"""
DAS-FL Project — Apply Building Exclusion Patch
Run from project root:
    python pipeline/apply_exclusion_patch.py

This script:
1. Adds exclude_buildings to config.yaml
2. Adds get_building_files() to dasfl/data_utils.py
3. Updates build_features.py to use exclusion
4. Updates train_local_baseline.py to use exclusion
5. Removes excluded buildings from data/processed/
"""

import json
import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _load_excluded_buildings():
    """Excluded substation IDs are local-only: logs/excluded_substations.json.

    Returns a list of int IDs, or [] if the gitignored file is absent.
    """
    local_file = PROJECT_ROOT / "logs" / "excluded_substations.json"
    if not local_file.exists():
        return []
    with open(local_file) as f:
        data = json.load(f)
    return [int(b) for ids in data.values() for b in ids]


EXCLUDE_BUILDINGS = _load_excluded_buildings()

def patch_config():
    """Add exclude_buildings to config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    existing = config.get("data", {}).get("quality", {}).get("exclude_buildings", None)
    if existing is not None:
        print(f"  config.yaml: exclude_buildings already present: {existing}")
        return

    config["data"]["quality"]["exclude_buildings"] = EXCLUDE_BUILDINGS

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  config.yaml: Added exclude_buildings: {EXCLUDE_BUILDINGS}")


def patch_data_utils():
    """Add get_building_files() function to dasfl/data_utils.py."""
    filepath = PROJECT_ROOT / "dasfl" / "data_utils.py"
    with open(filepath, "r") as f:
        content = f.read()

    if "get_building_files" in content:
        print("  data_utils.py: get_building_files already present")
        return

    # Insert after the load_building function (find the marker)
    function_code = '''

def get_building_files(
    data_path: Path,
    config: dict,
    extension: str = "*.pkl",
) -> list[Path]:
    """Get list of building files, excluding blacklisted buildings.

    Reads exclude_buildings from config['data']['quality'] and
    filters them out. Use this everywhere instead of raw glob.

    Args:
        data_path: Directory containing building files.
        config: Project config dict (from config.yaml).
        extension: File glob pattern (e.g., '*.pkl' or '*.parquet').

    Returns:
        Sorted list of file paths with excluded buildings removed.
    """
    exclude = config.get("data", {}).get("quality", {}).get("exclude_buildings", [])
    exclude_ids = set(str(b) for b in exclude)

    files = sorted(data_path.glob(extension))
    before = len(files)
    files = [f for f in files if f.stem not in exclude_ids]
    after = len(files)

    if before != after:
        print(f"  Excluded {before - after} building(s): {exclude_ids}")

    return files

'''

    # Insert before "# ── 2. CLEANING"
    marker = "# ── 2. CLEANING"
    if marker in content:
        content = content.replace(marker, function_code + "\n" + marker)
    else:
        # Fallback: insert before clean_building function
        marker2 = "def clean_building("
        if marker2 in content:
            content = content.replace(marker2, function_code + "\n" + marker2)
        else:
            print("  WARNING: Could not find insertion point in data_utils.py")
            print("  Add get_building_files() manually")
            return

    with open(filepath, "w") as f:
        f.write(content)

    print("  data_utils.py: Added get_building_files()")


def patch_build_features():
    """Update build_features.py to use get_building_files."""
    filepath = PROJECT_ROOT / "pipeline" / "build_features.py"
    with open(filepath, "r") as f:
        content = f.read()

    if "get_building_files" in content:
        print("  build_features.py: already patched")
        return

    # Add import
    old_import = "from dasfl.data_utils import ("
    new_import = "from dasfl.data_utils import (\n    get_building_files,"
    if old_import in content:
        content = content.replace(old_import, new_import)

    # Replace file listing in main()
    old_listing = '        files = sorted(Path(RAW_DATA_PATH).glob("*.pkl"))'
    new_listing = '        files = get_building_files(RAW_DATA_PATH, config, "*.pkl")'
    if old_listing in content:
        content = content.replace(old_listing, new_listing)
    else:
        # Try alternative pattern
        old_listing2 = "        files = sorted(RAW_DATA_PATH.glob(\"*.pkl\"))"
        new_listing2 = "        files = get_building_files(RAW_DATA_PATH, config, \"*.pkl\")"
        if old_listing2 in content:
            content = content.replace(old_listing2, new_listing2)

    with open(filepath, "w") as f:
        f.write(content)

    print("  build_features.py: Patched to use get_building_files()")


def patch_train_baseline():
    """Update train_local_baseline.py to use get_building_files."""
    filepath = PROJECT_ROOT / "pipeline" / "train_local_baseline.py"
    with open(filepath, "r") as f:
        content = f.read()

    if "get_building_files" in content:
        print("  train_local_baseline.py: already patched")
        return

    # Add import
    old_import = "from dasfl.data_utils import chronological_split, get_feature_target_split"
    new_import = "from dasfl.data_utils import chronological_split, get_feature_target_split, get_building_files"
    if old_import in content:
        content = content.replace(old_import, new_import)

    # Replace file listing in main()
    old_listing = '        files = sorted(PROCESSED_PATH.glob("*.parquet"))'
    new_listing = '        files = get_building_files(PROCESSED_PATH, config, "*.parquet")'
    if old_listing in content:
        content = content.replace(old_listing, new_listing)

    with open(filepath, "w") as f:
        f.write(content)

    print("  train_local_baseline.py: Patched to use get_building_files()")


def remove_excluded_processed():
    """Remove excluded buildings from data/processed/."""
    processed = PROJECT_ROOT / "data" / "processed"
    removed = 0
    for bid in EXCLUDE_BUILDINGS:
        fpath = processed / f"{bid}.parquet"
        if fpath.exists():
            fpath.unlink()
            print(f"  Removed: {fpath.name}")
            removed += 1
    if removed == 0:
        print("  No excluded buildings found in data/processed/")
    else:
        remaining = len(list(processed.glob("*.parquet")))
        print(f"  Remaining processed files: {remaining}")


if __name__ == "__main__":
    print("=" * 60)
    print("APPLYING BUILDING EXCLUSION PATCH")
    print("=" * 60)

    print("\n[1/5] Patching config.yaml...")
    patch_config()

    print("\n[2/5] Patching dasfl/data_utils.py...")
    patch_data_utils()

    print("\n[3/5] Patching pipeline/build_features.py...")
    patch_build_features()

    print("\n[4/5] Patching pipeline/train_local_baseline.py...")
    patch_train_baseline()

    print("\n[5/5] Removing excluded buildings from processed data...")
    remove_excluded_processed()

    print("\n" + "=" * 60)
    print("PATCH COMPLETE")
    print(f"Excluded buildings: {EXCLUDE_BUILDINGS}")
    print("All pipeline scripts now filter via config.yaml")
    print("To add more exclusions later: edit configs/config.yaml → exclude_buildings")
    print("=" * 60)