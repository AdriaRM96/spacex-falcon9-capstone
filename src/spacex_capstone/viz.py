"""Plotting helpers for the exploratory data analysis notebooks.

Backs notebook 05's repeated `catplot` pattern (FlightNumber/PayloadMass vs.
LaunchSite/Orbit, colored by landing outcome) plus the orbit success-rate bar
chart and yearly success-rate trend line.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_categorical_relationship(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = "Class",
    aspect: float = 5,
) -> sns.axisgrid.FacetGrid:
    """Plot a `catplot` of `x` vs. `y`, colored by `hue`.

    This is the pattern repeated across notebook 05's Tasks 1, 2, 4, and 5:
    FlightNumber/PayloadMass against LaunchSite/Orbit, colored by landing
    outcome (`Class`).

    Args:
        df: DataFrame containing `x`, `y`, and `hue` columns.
        x: Column to plot on the x-axis (typically numeric).
        y: Column to plot on the y-axis (typically categorical).
        hue: Column used to color points (defaults to the landing label).
        aspect: Width-to-height ratio passed to `catplot`.

    Returns:
        The seaborn `FacetGrid` produced by `catplot`, for further styling.
    """
    grid = sns.catplot(y=y, x=x, hue=hue, data=df, aspect=aspect)
    plt.xlabel(x, fontsize=20)
    plt.ylabel(y, fontsize=20)
    return grid


def plot_success_rate_by_group(
    df: pd.DataFrame,
    group_col: str,
    class_col: str = "Class",
    kind: str = "bar",
) -> None:
    """Plot the mean landing-success rate for each value of `group_col`.

    Since `class_col` is binary (0/1), its group-wise mean is exactly the
    success rate for that group. Used for both the per-orbit bar chart
    (Task 3) and the per-year trend line (Task 6) in notebook 05.

    Args:
        df: DataFrame containing `group_col` and `class_col`.
        group_col: Column to group by (e.g. "Orbit" or "Year").
        class_col: Binary landing-success column.
        kind: "bar" for a bar chart (categorical groups like Orbit) or
            "line" for a trend (ordered groups like Year).
    """
    success_rate = df.groupby(group_col)[class_col].mean()

    plt.figure(figsize=(10, 6))
    if kind == "bar":
        sns.barplot(x=success_rate.index, y=success_rate.values)
    elif kind == "line":
        plt.plot(success_rate.index, success_rate.values, marker="o")
    else:
        raise ValueError(f"Unsupported kind: {kind!r} (expected 'bar' or 'line')")

    plt.xlabel(group_col, fontsize=20)
    plt.ylabel("Success Rate", fontsize=20)
