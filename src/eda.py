"""
eda.py
======

Generate the exploratory-data-analysis figures referenced in the report:
target trend, daily/weekly seasonality, correlation heatmap and target distribution.

Run as a script::

    python src/eda.py

Figures are written to ``reports/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import drop_noise_columns, handle_missing_values, load_data

TARGET = "Appliances"


def run_eda(raw_path: str | Path, fig_dir: str | Path) -> None:
    """Produce and save the core EDA visualisations."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = handle_missing_values(drop_noise_columns(load_data(raw_path)))

    # 1. Target trend over the full period (resampled to hourly for legibility).
    plt.figure(figsize=(14, 4))
    df[TARGET].resample("1h").mean().plot(linewidth=0.7)
    plt.title("Appliance energy consumption over time (hourly mean)")
    plt.ylabel("Appliances (Wh)")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_target_trend.png", dpi=120)
    plt.close()

    # 2. Distribution of the target (note the right skew / spikes).
    plt.figure(figsize=(8, 4))
    sns.histplot(df[TARGET], bins=80, kde=True)
    plt.title("Distribution of Appliances energy use")
    plt.xlabel("Appliances (Wh)")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_target_distribution.png", dpi=120)
    plt.close()

    # 3. Average consumption by hour of day (daily seasonality).
    by_hour = df.groupby(df.index.hour)[TARGET].mean()
    plt.figure(figsize=(9, 4))
    by_hour.plot(marker="o")
    plt.title("Average consumption by hour of day")
    plt.xlabel("Hour")
    plt.ylabel("Mean Appliances (Wh)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_hourly_profile.png", dpi=120)
    plt.close()

    # 4. Consumption by day of week (weekday vs weekend).
    by_dow = df.groupby(df.index.dayofweek)[TARGET].mean()
    plt.figure(figsize=(9, 4))
    by_dow.plot(kind="bar")
    plt.title("Average consumption by day of week (0=Mon .. 6=Sun)")
    plt.xlabel("Day of week")
    plt.ylabel("Mean Appliances (Wh)")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_dayofweek_profile.png", dpi=120)
    plt.close()

    # 5. Correlation heatmap of the main sensors with the target.
    cols = [TARGET, "lights", "T1", "RH_1", "T2", "RH_2", "T3", "T_out",
            "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]
    cols = [c for c in cols if c in df.columns]
    plt.figure(figsize=(11, 9))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (selected features)")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_correlation_heatmap.png", dpi=120)
    plt.close()

    # 6. Boxplot of the target by hour to show spread / outliers.
    plt.figure(figsize=(13, 4))
    sns.boxplot(x=df.index.hour, y=df[TARGET])
    plt.title("Appliances distribution by hour (boxplot)")
    plt.xlabel("Hour")
    plt.ylabel("Appliances (Wh)")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_boxplot_by_hour.png", dpi=120)
    plt.close()

    print(f"EDA figures saved to {fig_dir}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    run_eda(here / "data" / "raw" / "energy_data_set.csv", here / "reports" / "figures")
