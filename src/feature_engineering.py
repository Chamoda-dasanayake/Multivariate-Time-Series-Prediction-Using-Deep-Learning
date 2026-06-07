"""
feature_engineering.py
======================

Feature construction and selection for the Appliances Energy Prediction task.

The functions here turn the cleaned sensor frame into a richer feature matrix:

* **Time-based features** - hour, day-of-week, month, weekend flag, ``NSM``
  (seconds since midnight) plus cyclical (sin/cos) encodings so the network sees
  the wrap-around at midnight / end-of-week.
* **Rolling-window statistics** - 1-hour and 3-hour moving averages / std of the
  target to expose short-term momentum.
* **Lagged features** - past ``Appliances`` values at lags suggested by the
  autocorrelation function.
* **Interaction features** - temperature x humidity products that capture comfort
  effects driving appliance use.
* **Feature selection** - Random-Forest importance ranking.

All feature creation is causal (only past/at-time information is used), which keeps
the pipeline leakage-free for forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Appliances"


# --------------------------------------------------------------------------- #
# Time-based features
# --------------------------------------------------------------------------- #
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar / clock features derived from the DatetimeIndex.

    Includes the ``NSM``, ``WeekStatus`` and ``Day_of_week`` fields referenced in
    the assessment brief (reconstructed from the timestamp) plus cyclical encodings.
    """
    df = df.copy()
    idx = df.index

    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek          # Monday=0 .. Sunday=6
    df["month"] = idx.month
    df["day"] = idx.day
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["WeekStatus"] = np.where(df["is_weekend"] == 1, "Weekend", "Weekday")
    df["Day_of_week"] = idx.dayofweek
    df["NSM"] = idx.hour * 3600 + idx.minute * 60 + idx.second  # seconds since midnight

    # Cyclical encodings so 23:50 and 00:00 are "close".
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# --------------------------------------------------------------------------- #
# Rolling-window statistics
# --------------------------------------------------------------------------- #
def add_rolling_features(
    df: pd.DataFrame, target: str = TARGET, windows: tuple[int, ...] = (6, 18)
) -> pd.DataFrame:
    """Add rolling mean / std of the target.

    ``windows`` are expressed in number of 10-minute steps: 6 -> 1 hour,
    18 -> 3 hours. ``shift(1)`` ensures the statistic only uses *past* values
    (no leakage of the current target into its own predictor).
    """
    df = df.copy()
    shifted = df[target].shift(1)
    for w in windows:
        df[f"{target}_roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"{target}_roll_std_{w}"] = shifted.rolling(w, min_periods=1).std()
    return df


# --------------------------------------------------------------------------- #
# Lagged features
# --------------------------------------------------------------------------- #
def add_lag_features(
    df: pd.DataFrame, target: str = TARGET, lags: tuple[int, ...] = (1, 2, 3, 6)
) -> pd.DataFrame:
    """Add lagged copies of the target (e.g. value 10/20/30/60 minutes ago)."""
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


def autocorrelation_lags(series: pd.Series, n_lags: int = 24, threshold: float = 0.2) -> list[int]:
    """Return lag indices whose autocorrelation exceeds ``threshold``.

    Used to justify the choice of lag features. Returns at most ``n_lags`` lags.
    """
    from statsmodels.tsa.stattools import acf

    values = acf(series.dropna(), nlags=n_lags, fft=True)
    return [lag for lag in range(1, n_lags + 1) if abs(values[lag]) >= threshold]


# --------------------------------------------------------------------------- #
# Interaction features
# --------------------------------------------------------------------------- #
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature x humidity interaction terms.

    Indoor comfort (and therefore appliance use) depends jointly on temperature and
    humidity, so the product of paired sensors is a physically meaningful feature.
    Also adds the indoor/outdoor temperature gradient.
    """
    df = df.copy()
    temp_hum_pairs = [
        ("T1", "RH_1"), ("T2", "RH_2"), ("T3", "RH_3"),
        ("T_out", "RH_out"),
    ]
    for t, rh in temp_hum_pairs:
        if t in df.columns and rh in df.columns:
            df[f"{t}_x_{rh}"] = df[t] * df[rh]

    if "T1" in df.columns and "T_out" in df.columns:
        df["temp_gradient"] = df["T1"] - df["T_out"]
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Run the full feature pipeline and drop rows with NaNs introduced by lags."""
    df = add_time_features(df)
    df = add_interaction_features(df)
    df = add_rolling_features(df, target=target)
    df = add_lag_features(df, target=target)
    df = df.dropna()
    return df


def get_feature_columns(df: pd.DataFrame, target: str = TARGET) -> list[str]:
    """Return the numeric model-input columns (everything except the target and the
    raw categorical helper ``WeekStatus``)."""
    drop = {target, "WeekStatus"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in drop]


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
def rank_feature_importance(
    df: pd.DataFrame, feature_cols: list[str], target: str = TARGET, n_estimators: int = 200
) -> pd.DataFrame:
    """Rank features by Random-Forest impurity importance.

    A tree ensemble captures non-linear / interaction effects that a plain
    correlation analysis would miss, giving a more reliable importance ranking for
    this multivariate problem.
    """
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    rf.fit(df[feature_cols], df[target])
    imp = (
        pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp


def select_top_features(importance: pd.DataFrame, top_k: int = 25) -> list[str]:
    """Return the ``top_k`` most important feature names."""
    return importance.head(top_k)["feature"].tolist()


if __name__ == "__main__":
    from pathlib import Path
    from data_preprocessing import (
        load_data, drop_noise_columns, handle_missing_values, treat_outliers,
    )

    here = Path(__file__).resolve().parents[1]
    df = load_data(here / "data" / "raw" / "energy_data_set.csv")
    df = handle_missing_values(drop_noise_columns(df))
    df, _ = treat_outliers(df)
    feats = build_features(df)
    cols = get_feature_columns(feats)
    print("Engineered frame:", feats.shape, "| #features:", len(cols))
    print("Suggested ACF lags:", autocorrelation_lags(df[TARGET]))
