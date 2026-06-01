"""
DAS-FL Project — Add Outdoor Temperature to Pipeline
Run from project root:
    python pipeline/add_weather_patch.py

This script:
1. Updates config.yaml to enable weather features
2. Adds load_outdoor_temperature() to dasfl/data_utils.py
3. Updates build_features() to merge outdoor temp
4. Updates prepare_building() to accept weather data
"""

import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def patch_config():
    """Enable weather features and add weather path to config."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Update include_weather
    config["model"]["features"]["include_weather"] = True

    # Add weather path
    if "weather_file" not in config["paths"]:
        config["paths"]["weather_file"] = "data/external/outdoor_temp_boras.csv"

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("  config.yaml: include_weather=True, weather_file path added")


def patch_data_utils():
    """Add weather loading and merge functions to data_utils.py."""
    filepath = PROJECT_ROOT / "dasfl" / "data_utils.py"
    with open(filepath, "r") as f:
        content = f.read()

    if "load_outdoor_temperature" in content:
        print("  data_utils.py: weather functions already present")
        return

    # ── 1. Add the weather functions before build_features ──
    weather_code = '''

# ── WEATHER DATA 

def load_outdoor_temperature(filepath: str | Path) -> pd.Series:
    """Load outdoor temperature from SMHI CSV file.

    Args:
        filepath: Path to the outdoor temperature CSV.
            Expected columns: Temperature, datetime

    Returns:
        pd.Series with DatetimeIndex and name 'outdoor_temp'.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Weather file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Parse datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError("Expected 'datetime' column in weather file")

    # Extract temperature series
    temp = df["Temperature"].copy()
    temp.name = "outdoor_temp"
    temp.index.name = "timestamp"

    # Sort and remove duplicates
    temp = temp.sort_index()
    temp = temp[~temp.index.duplicated(keep="first")]

    return temp


def merge_weather(
    df: pd.DataFrame,
    outdoor_temp: pd.Series,
    interpolate_limit: int = 6,
) -> pd.DataFrame:
    """Merge outdoor temperature into a building DataFrame.

    Uses nearest-hour matching via reindex, then interpolates
    small gaps.

    Args:
        df: Building DataFrame with DatetimeIndex.
        outdoor_temp: Series from load_outdoor_temperature().
        interpolate_limit: Max hours to interpolate weather gaps.

    Returns:
        DataFrame with 'outdoor_temp' column added.
    """
    df = df.copy()

    # Reindex weather to match building timestamps
    matched = outdoor_temp.reindex(df.index, method="nearest", tolerance=pd.Timedelta("2h"))

    # Interpolate remaining small gaps
    matched = matched.interpolate(method="linear", limit=interpolate_limit)

    df["outdoor_temp"] = matched

    return df

'''

    # Insert before the build_features function
    marker = "def build_features("
    if marker in content:
        content = content.replace(marker, weather_code + "\n" + marker)
    else:
        print("  WARNING: Could not find build_features() — add weather functions manually")
        return

    # ── 2. Update build_features to include outdoor temp features ──
    # Add outdoor_temp lags and rolling stats if present
    old_return = '''    return df_feat'''

    new_return = '''    # Weather features (if outdoor_temp is present)
    if "outdoor_temp" in df_feat.columns:
        # Lagged outdoor temp
        for lag in [1, 3, 6, 24]:
            df_feat[f"outdoor_temp_lag_{lag}"] = df_feat["outdoor_temp"].shift(lag)
        # Rolling outdoor temp stats
        for window in [6, 24]:
            df_feat[f"outdoor_temp_rolling_mean_{window}"] = (
                df_feat["outdoor_temp"].rolling(window=window, min_periods=max(1, window // 2)).mean()
            )
        # Rate of change of outdoor temp
        df_feat["outdoor_temp_diff_1"] = df_feat["outdoor_temp"].diff(1)
        df_feat["outdoor_temp_diff_24"] = df_feat["outdoor_temp"].diff(24)

    return df_feat'''

    # Find the last "return df_feat" in build_features
    # We need to be careful to only replace the one inside build_features
    # Find the function and its return
    import re
    # Find build_features function body and replace its return
    func_match = re.search(r'(def build_features\(.*?\n(?:.*\n)*?)(    return df_feat)', content)
    if func_match:
        old_section = func_match.group(0)
        new_section = old_section.replace("    return df_feat", new_return, 1)
        content = content.replace(old_section, new_section, 1)
    else:
        print("  WARNING: Could not find return in build_features()")

    # ── 3. Update prepare_building to accept and merge weather ──
    old_prepare_sig = '''def prepare_building(
    filepath: str | Path,
    config: dict,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, dict]:'''

    new_prepare_sig = '''def prepare_building(
    filepath: str | Path,
    config: dict,
    drop_na: bool = True,
    outdoor_temp: pd.Series = None,
) -> tuple[pd.DataFrame, dict]:'''

    if old_prepare_sig in content:
        content = content.replace(old_prepare_sig, new_prepare_sig)

    # Add weather merge after cleaning, before features
    old_features_call = '''    # Features
    feat_cfg = config["model"]["features"]
    df_feat = build_features('''

    new_features_call = '''    # Merge weather data if provided
    if outdoor_temp is not None:
        df_clean = merge_weather(df_clean, outdoor_temp)

    # Features
    feat_cfg = config["model"]["features"]
    df_feat = build_features('''

    if old_features_call in content:
        content = content.replace(old_features_call, new_features_call, 1)

    # ── 4. Update input_columns in prepare_building to include outdoor_temp ──
    old_input = '''        input_columns=["kwh", "m3h", "ft", "rt", "dt"],'''
    new_input = '''        input_columns=["kwh", "m3h", "ft", "rt", "dt"] + (["outdoor_temp"] if outdoor_temp is not None else []),'''

    if old_input in content:
        content = content.replace(old_input, new_input, 1)

    with open(filepath, "w") as f:
        f.write(content)

    print("  data_utils.py: Added load_outdoor_temperature(), merge_weather(), updated build_features() and prepare_building()")


def patch_build_features_script():
    """Update pipeline/build_features.py to load and pass weather data."""
    filepath = PROJECT_ROOT / "pipeline" / "build_features.py"
    with open(filepath, "r") as f:
        content = f.read()

    if "load_outdoor_temperature" in content:
        print("  build_features.py: already patched for weather")
        return

    # Add import
    old_import = "from dasfl.data_utils import ("
    new_import = "from dasfl.data_utils import (\n    load_outdoor_temperature,"
    if old_import in content:
        content = content.replace(old_import, new_import, 1)

    # Add weather loading in process_single_building
    old_process = '''def process_single_building(filepath: Path) -> dict:
    """Process one building and save to parquet.

    Returns metadata dict with building stats.
    """
    building_id = filepath.stem
    out_path = PROCESSED_PATH / f"{building_id}.parquet"

    try:
        df_feat, metadata = prepare_building(filepath, config, drop_na=True)'''

    new_process = '''# Global: load weather data once
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
        df_feat, metadata = prepare_building(filepath, config, drop_na=True, outdoor_temp=outdoor_temp)'''

    if old_process in content:
        content = content.replace(old_process, new_process)
    else:
        print("  WARNING: Could not find process_single_building() — patch manually")

    with open(filepath, "w") as f:
        f.write(content)

    print("  build_features.py: Patched to load and pass outdoor temperature")


if __name__ == "__main__":
    print("=" * 60)
    print("ADDING OUTDOOR TEMPERATURE TO PIPELINE")
    print("=" * 60)

    print("\n[1/3] Patching config.yaml...")
    patch_config()

    print("\n[2/3] Patching dasfl/data_utils.py...")
    patch_data_utils()

    print("\n[3/3] Patching pipeline/build_features.py...")
    patch_build_features_script()

    print("\n" + "=" * 60)
    print("PATCH COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Clear old processed files:  rm data/processed/*.parquet")
    print("  2. Rebuild features:           python pipeline/build_features.py --n 5")
    print("  3. Verify new columns include outdoor_temp features")
    print("  4. Full rebuild:               python pipeline/build_features.py")
    print("  5. Rerun baseline:             python pipeline/train_local_baseline.py")
    print("=" * 60)