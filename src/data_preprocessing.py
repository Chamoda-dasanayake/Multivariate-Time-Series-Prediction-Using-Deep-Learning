"""
data_preprocessing.py
=====================

Data loading, cleaning, outlier treatment, scaling and temporal splitting for the
Appliances Energy Prediction dataset.

The module is intentionally framework agnostic (pure pandas / numpy / scikit-learn)
so it can be reused by the baseline models, the deep-learning pipeline and the
exploratory notebook.

Dataset notes
-------------
* Records are sampled at a fixed 10-minute cadence.
* ``Appliances`` (Wh) is the regression target.
* ``rv1`` and ``rv2`` are *random* variables that the dataset authors injected as
  noise. They carry no signal and are dropped during cleaning.
* The raw file ships a ``date`` timestamp instead of the ``NSM`` / ``WeekStatus`` /
  ``Day_of_week`` columns described in the brief; those are reconstructed in
  :mod:`feature_engineering`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Columns the dataset authors added purely as random noise (no predictive value).
NOISE_COLUMNS: tuple[str, ...] = ("rv1", "rv2")

# The regression target.
TARGET: str = "Appliances"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV, parse the timestamp and return a time-sorted frame.

    Parameters
    ----------
    path:
        Path to ``energy_data_set.csv``.

    Returns
    -------
    pandas.DataFrame
        Frame indexed by a monotonically increasing ``date`` column. The index is
        a proper :class:`~pandas.DatetimeIndex` so that resampling / rolling
        operations downstream behave correctly.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    return df


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact per-column summary (dtype, missing count, basic stats)."""
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(3),
            "n_unique": df.nunique(),
            "min": df.min(numeric_only=True),
            "max": df.max(numeric_only=True),
            "mean": df.mean(numeric_only=True),
        }
    )
    return summary


def drop_noise_columns(df: pd.DataFrame, columns: Iterable[str] = NOISE_COLUMNS) -> pd.DataFrame:
    """Drop the random-noise columns if present (safe no-op otherwise)."""
    present = [c for c in columns if c in df.columns]
    return df.drop(columns=present)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using time-aware interpolation.

    Strategy
    --------
    Because the data is an evenly-spaced time series, linear interpolation along
    the time axis is preferred over column-wise mean imputation: it preserves the
    local trend of each sensor and avoids flattening short gaps to a global mean.
    Any residual gaps at the very start/end (which interpolation cannot fill) are
    back/forward filled.
    """
    df = df.sort_index()
    df = df.interpolate(method="time", limit_direction="both")
    df = df.bfill().ffill()
    return df


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return a boolean mask of IQR outliers for a single column.

    A point is an outlier if it falls outside ``[Q1 - k*IQR, Q3 + k*IQR]``.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def treat_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    k: float = 1.5,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap (winsorise) outliers in the environmental sensor columns.

    Rationale
    ---------
    The target ``Appliances`` is genuinely heavy-tailed (real spikes in energy use
    carry signal), so we deliberately **do not** clip the target — removing those
    rows would discard the very events we want to predict. Instead we winsorise the
    *sensor* columns to the IQR fence, which damps obviously erroneous readings
    while keeping every timestamp (important for temporal continuity).

    Returns
    -------
    (cleaned_df, report)
        ``report`` lists the outlier count per treated column.
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in df.select_dtypes(include=np.number).columns if c != target]

    rows = []
    for col in columns:
        mask = detect_outliers_iqr(df[col], k=k)
        n = int(mask.sum())
        if n:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            df[col] = df[col].clip(lower, upper)
        rows.append({"column": col, "n_outliers": n, "pct": round(100 * n / len(df), 2)})

    report = pd.DataFrame(rows).sort_values("n_outliers", ascending=False).reset_index(drop=True)
    return df, report


def temporal_train_test_split(
    df: pd.DataFrame, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically to avoid look-ahead leakage.

    The first ``train_frac`` of the (time-ordered) rows become the training set and
    the remainder the test set. No shuffling is performed.
    """
    n_train = int(len(df) * train_frac)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def scale_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler]:
    """Standardise features and target using statistics fit on the training set only.

    Standardisation (zero mean, unit variance) is chosen over Min-Max scaling
    because several sensor distributions are skewed/heavy-tailed; standardisation
    is less sensitive to the extreme values than Min-Max and pairs well with the
    gradient-based optimisation used by the neural networks. The target is scaled
    with its own scaler so predictions can be inverse-transformed back to Wh.

    Returns
    -------
    (train_scaled, test_scaled, feature_scaler, target_scaler)
    """
    feature_scaler = StandardScaler().fit(train[feature_cols])
    target_scaler = StandardScaler().fit(train[[target]])

    train_scaled = train.copy()
    test_scaled = test.copy()

    train_scaled[feature_cols] = feature_scaler.transform(train[feature_cols])
    test_scaled[feature_cols] = feature_scaler.transform(test[feature_cols])

    train_scaled[target] = target_scaler.transform(train[[target]]).ravel()
    test_scaled[target] = target_scaler.transform(test[[target]]).ravel()

    return train_scaled, test_scaled, feature_scaler, target_scaler


if __name__ == "__main__":
    # Smoke test when run directly.
    here = Path(__file__).resolve().parents[1]
    raw = load_data(here / "data" / "raw" / "energy_data_set.csv")
    print("Loaded:", raw.shape)
    print(basic_summary(raw).head(10))
    clean = handle_missing_values(drop_noise_columns(raw))
    clean, rep = treat_outliers(clean)
    print(rep.head())
    tr, te = temporal_train_test_split(clean)
    print("Train/Test:", tr.shape, te.shape)
