"""
DAS-FL Project — Data Utilities

Shared module for data loading, cleaning, and feature engineering.
Used by both pipeline/ scripts and dasfl/ Flower app.

Features created:
    - Lagged values (kwh, m3h, ft, rt, dt)
    - Rolling statistics (mean, max, std)
    - Temporal encoding (sin/cos for hour, day-of-week, month)
    - Rate of change
    - Weekend/holiday flags

Usage:
    from dasfl.data_utils import load_building, clean_building, build_features

"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# ── 1. LOADING 

def load_building(filepath: str | Path) -> pd.DataFrame:
    """Load a single building pkl file.

    Args:
        filepath: Path to .pkl file.

    Returns:
        DataFrame with DatetimeIndex named 'timestamp'.

    Raises:
        ValueError: If index is not DatetimeIndex.
        FileNotFoundError: If file doesn't exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Building file not found: {filepath}")

    df = pd.read_pickle(filepath)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex, got {type(df.index)}")
    df.index.name = "timestamp"
    df = df.sort_index()
    return df


# ── 2. CLEANING

def _load_local_excluded_ids() -> set:
    """Load locally-stored excluded substation IDs (gitignored).

    The real substation IDs are kept out of version control in
    logs/excluded_substations.json (see configs/config.yaml). The file maps a
    reason -> list of ID strings, e.g. {"disconnected_meter": ["123456"]}.
    Returns a set of ID strings; an empty set if the file is absent.
    """
    local_file = Path(__file__).resolve().parent.parent / "logs" / "excluded_substations.json"
    if not local_file.exists():
        return set()
    with open(local_file) as f:
        data = json.load(f)
    return {str(b) for ids in data.values() for b in ids}


def get_building_files(data_path: Path, config: dict, extension: str = "*.pkl") -> list[Path]:
    """Get list of building files, excluding blacklisted buildings.

    Args:
        data_path: Directory containing building files.
        config: Project config dict.
        extension: File pattern to match.

    Returns:
        Sorted list of file paths, excluding blacklisted buildings.
    """
    exclude = config.get("data", {}).get("quality", {}).get("exclude_buildings", [])
    exclude_ids = set(str(b) for b in exclude)
    exclude_ids |= _load_local_excluded_ids()

    files = sorted(data_path.glob(extension))
    files = [f for f in files if f.stem not in exclude_ids]
    return files

def clean_building(
    df: pd.DataFrame,
    kwh_sentinel: float = 99999.9,
    kwh_max_valid: float = 500.0,
    dt_min_valid: float = 0.0,
    negative_dt_strategy: str = "clip",
    interpolate_limit: int = 6,
) -> pd.DataFrame:
    """Clean a single building's data.

    Steps:
        1. Remove sentinel values (kwh >= kwh_sentinel) → NaN
        2. Remove extreme outliers (kwh > kwh_max_valid) → NaN
        3. Remove negative kwh → NaN
        4. Handle negative delta-T based on strategy
        5. Interpolate short gaps (up to interpolate_limit hours)
        6. Ensure hourly frequency (resample if needed)

    Args:
        df: Raw building DataFrame.
        kwh_sentinel: Sentinel value to remove (e.g., 99999.9).
        kwh_max_valid: Max plausible kwh per hour.
        dt_min_valid: Minimum valid delta-T (typically 0).
        negative_dt_strategy: How to handle negative dt.
            - "clip": Set negative dt to 0 (conservative)
            - "nan": Replace negative dt rows with NaN (aggressive)
            - "keep": Do nothing (wait for domain expert input)
        interpolate_limit: Max consecutive NaN hours to interpolate.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Step 1: Sentinel removal
    sentinel_mask = df["kwh"] >= kwh_sentinel
    df.loc[sentinel_mask, "kwh"] = np.nan

    # Step 2: Extreme outlier removal
    outlier_mask = df["kwh"] > kwh_max_valid
    df.loc[outlier_mask, "kwh"] = np.nan

    # Step 3: Negative kwh
    neg_kwh_mask = df["kwh"] < 0
    df.loc[neg_kwh_mask, "kwh"] = np.nan

    # Step 4: Negative delta-T handling
    if negative_dt_strategy == "clip":
        df["dt"] = df["dt"].clip(lower=dt_min_valid)
    elif negative_dt_strategy == "nan":
        neg_dt_mask = df["dt"] < dt_min_valid
        for col in ["kwh", "m3h", "ft", "rt", "dt"]:
            if col in df.columns:
                df.loc[neg_dt_mask, col] = np.nan
    elif negative_dt_strategy == "keep":
        pass  # Do nothing — decide after workshop
    else:
        raise ValueError(f"Unknown negative_dt_strategy: {negative_dt_strategy}")

    # Step 5: Ensure hourly frequency
    freq = pd.infer_freq(df.index)
    if freq != "h":
        df = df.resample("h").first()

    # Step 6: Interpolate short gaps
    for col in ["kwh", "m3h", "ft", "rt", "dt"]:
        if col in df.columns:
            df[col] = df[col].interpolate(
                method="linear", limit=interpolate_limit
            )

    return df


# ── 3. FEATURE ENGINEERING 

def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged features for specified columns.

    Args:
        df: DataFrame with time series data.
        columns: Column names to create lags for.
        lags: List of lag values in hours (e.g., [1, 24, 168]).

    Returns:
        DataFrame with added lag columns named '{col}_lag_{lag}'.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: list[int],
    stats: list[str] = None,
) -> pd.DataFrame:
    """Add rolling window statistics.

    Args:
        df: DataFrame with time series data.
        column: Column to compute rolling stats on.
        windows: Window sizes in hours (e.g., [6, 24, 168]).
        stats: Statistics to compute. Default: ["mean", "max", "std"].

    Returns:
        DataFrame with added columns named '{col}_rolling_{stat}_{window}'.
    """
    if stats is None:
        stats = ["mean", "max", "std"]

    df = df.copy()
    for window in windows:
        rolling = df[column].rolling(window=window, min_periods=max(1, window // 2))
        for stat in stats:
            col_name = f"{column}_rolling_{stat}_{window}"
            if stat == "mean":
                df[col_name] = rolling.mean()
            elif stat == "max":
                df[col_name] = rolling.max()
            elif stat == "std":
                df[col_name] = rolling.std()
            elif stat == "min":
                df[col_name] = rolling.min()
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical temporal encoding features.

    Creates sin/cos encoding for:
        - Hour of day (period = 24)
        - Day of week (period = 7)
        - Month of year (period = 12)
    Plus binary flags: is_weekend

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with temporal feature columns added.
    """
    df = df.copy()
    idx = df.index

    # Hour of day (0-23)
    hour = idx.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week (0=Monday, 6=Sunday)
    dow = idx.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Month of year (1-12)
    month = idx.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Binary flags
    df["is_weekend"] = (dow >= 5).astype(int)

    return df


def add_rate_of_change(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Add first-difference (rate of change) features.

    Args:
        df: DataFrame with time series data.
        columns: Columns to compute rate of change for.

    Returns:
        DataFrame with '{col}_diff_1' columns added.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_diff_1"] = df[col].diff(1)
    return df




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


def build_features(
    df: pd.DataFrame,
    input_columns: list[str] = None,
    kwh_lags: list[int] = None,
    other_lags: list[int] = None,
    rolling_windows: list[int] = None,
    include_temporal: bool = True,
    include_rate_of_change: bool = True,
) -> pd.DataFrame:
    """Full feature engineering pipeline for one building.

    This is the main entry point. Takes a cleaned DataFrame
    and returns a feature-rich DataFrame ready for modelling.

    Args:
        df: Cleaned building DataFrame (output of clean_building).
        input_columns: Raw input feature columns.
            Default: ["kwh", "m3h", "ft", "rt", "dt"]
        kwh_lags: Lag values for kwh.
            Default: [1, 2, 3, 6, 12, 24, 48, 168]
        other_lags: Lag values for non-kwh input features.
            Default: [1, 24]
        rolling_windows: Window sizes for rolling stats on kwh.
            Default: [6, 12, 24, 168]
        include_temporal: Whether to add temporal encoding.
        include_rate_of_change: Whether to add diff features.

    Returns:
        DataFrame with all features. NaN rows from lagging
        are NOT dropped — caller decides how to handle them.
    """
    if input_columns is None:
        input_columns = ["kwh", "m3h", "ft", "rt", "dt"]
    if kwh_lags is None:
        kwh_lags = [1, 2, 3, 6, 12, 24, 48, 168]
    if other_lags is None:
        other_lags = [1, 24]
    if rolling_windows is None:
        rolling_windows = [6, 12, 24, 168]

    # Start with input columns only
    available = [c for c in input_columns if c in df.columns]
    df_feat = df[available].copy()

    # Lagged kwh features (most important for time-series)
    df_feat = add_lag_features(df_feat, columns=["kwh"], lags=kwh_lags)

    # Lagged features for other inputs (lighter — just 1h and 24h)
    other_cols = [c for c in available if c != "kwh"]
    df_feat = add_lag_features(df_feat, columns=other_cols, lags=other_lags)

    # Rolling statistics on kwh
    df_feat = add_rolling_features(
        df_feat, column="kwh", windows=rolling_windows,
        stats=["mean", "max", "std"],
    )

    # Temporal encoding
    if include_temporal:
        df_feat = add_temporal_features(df_feat)

    # Rate of change
    if include_rate_of_change:
        df_feat = add_rate_of_change(df_feat, columns=["kwh", "dt"])

    # Weather features (if outdoor_temp is present)
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

    return df_feat


# ── 4. TRAIN/VAL/TEST SPLIT

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series chronologically (no shuffling).

    Args:
        df: Feature DataFrame (must be sorted by time).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# ── 5. TARGET CREATION

def create_target(
    df: pd.DataFrame,
    target_col: str = "kwh",
    horizons: list[int] = None,
) -> pd.DataFrame:
    """Create prediction target columns for specified horizons.

    For each horizon h, creates target column 'target_kwh_t+{h}'
    which is the kwh value h hours in the future.

    Args:
        df: Feature DataFrame.
        target_col: Column to predict.
        horizons: Forecast horizons in hours. Default: [1, 6, 24, 168].

    Returns:
        DataFrame with target columns added.
    """
    if horizons is None:
        horizons = [1, 6, 24, 168]

    df = df.copy()
    for h in horizons:
        df[f"target_{target_col}_t+{h}"] = df[target_col].shift(-h)

    return df


# ── 6. FULL PIPELINE

def prepare_building(
    filepath: str | Path,
    config: dict,
    drop_na: bool = True,
    outdoor_temp: pd.Series = None,
) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: load → clean → features → targets.

    This is the convenience function that runs the entire
    data preparation pipeline for a single building.

    Args:
        filepath: Path to building .pkl file.
        config: Project config dict (from config.yaml).

    Returns:
        Tuple of (prepared_df, metadata_dict).
        metadata_dict contains building_id, date range, row counts, etc.
    """
    building_id = Path(filepath).stem

    # Load
    df_raw = load_building(filepath)
    n_raw = len(df_raw)

    # Clean
    q = config["data"]["quality"]
    df_clean = clean_building(
        df_raw,
        kwh_sentinel=q["kwh_sentinel"],
        kwh_max_valid=q["kwh_max_valid"],
        dt_min_valid=q["dt_min_valid"],
        negative_dt_strategy="clip",  # conservative default
        interpolate_limit=6,
    )
    n_clean = len(df_clean)

    # Merge weather data if provided
    if outdoor_temp is not None:
        df_clean = merge_weather(df_clean, outdoor_temp)

    # Features
    feat_cfg = config["model"]["features"]
    df_feat = build_features(
        df_clean,
        input_columns=["kwh", "m3h", "ft", "rt", "dt"] + (["outdoor_temp"] if outdoor_temp is not None else []),
        kwh_lags=feat_cfg["lagged_kwh"],
        rolling_windows=feat_cfg["rolling_windows"],
        include_temporal=feat_cfg["temporal_encoding"],
    )

    # Targets
    horizons = config["model"]["baseline"]["eval_horizons"]
    df_feat = create_target(df_feat, target_col="kwh", horizons=horizons)

    # Drop NaN rows (from lagging and target shifting)
    n_before_drop = len(df_feat)
    if drop_na:
        df_feat = df_feat.dropna()
    n_final = len(df_feat)

    metadata = {
        "building_id": building_id,
        "n_raw": n_raw,
        "n_clean": n_clean,
        "n_features": n_before_drop,
        "n_final": n_final,
        "n_dropped_na": n_before_drop - n_final,
        "date_min": str(df_feat.index.min()) if n_final > 0 else None,
        "date_max": str(df_feat.index.max()) if n_final > 0 else None,
        "n_columns": len(df_feat.columns),
        "feature_names": list(df_feat.columns),
    }

    return df_feat, metadata


def get_feature_target_split(
    df: pd.DataFrame,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split prepared DataFrame into feature matrix X and target y.

    Args:
        df: Output of prepare_building or build_features + create_target.
        horizon: Which target horizon to use (e.g., 1 for t+1).

    Returns:
        Tuple of (X, y) where X is features and y is the target series.
    """
    target_col = f"target_kwh_t+{horizon}"
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available targets: {[c for c in df.columns if c.startswith('target_')]}"
        )

    # Target columns and raw kwh (current value leaks future in some setups)
    exclude_cols = [c for c in df.columns if c.startswith("target_")]
    # Keep 'kwh' lagged versions but remove current 'kwh' to prevent leakage
    # Current kwh at time t is fine as feature when predicting t+h (h >= 1)
    # But raw input columns (kwh, m3h, ft, rt, dt) at time t are valid features

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]

    return X, y
