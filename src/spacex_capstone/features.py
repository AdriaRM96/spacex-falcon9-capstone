"""Feature and label engineering for the landing-success prediction task.

Backs notebook 03 (the `Class` label) and notebook 05 (one-hot encoding for
modeling).
"""

from __future__ import annotations

import pandas as pd

# Outcome strings that mean the first stage did NOT land successfully.
# Derived from notebook 03's `df['Outcome'].value_counts()` on the
# dataset_part_1 data: any outcome other than these (e.g. "True ASDS",
# "True RTLS", "True Ocean") counts as a successful landing.
BAD_LANDING_OUTCOMES = frozenset(
    {
        "None None",
        "False ASDS",
        "False Ocean",
        "None ASDS",
        "False RTLS",
    }
)


def build_landing_class(outcome: pd.Series, bad_outcomes: frozenset[str] = BAD_LANDING_OUTCOMES) -> list[int]:
    """Convert raw `Outcome` strings into a binary landing-success label.

    Args:
        outcome: A pandas Series of `Outcome` strings (e.g. "True ASDS",
            "False Ocean", "None None").
        bad_outcomes: The set of outcome strings that count as a failed
            landing. Defaults to `BAD_LANDING_OUTCOMES`.

    Returns:
        A list of 0/1 ints, one per row: 0 if the outcome is in
        `bad_outcomes` (landing failed or was never attempted), 1 otherwise
        (landing succeeded).
    """
    return [0 if value in bad_outcomes else 1 for value in outcome]


def one_hot_encode_features(
    features: pd.DataFrame,
    categorical_columns: list[str] = ("Orbit", "LaunchSite", "LandingPad", "Serial"),
) -> pd.DataFrame:
    """One-hot encode categorical columns and cast everything to float64.

    Mirrors notebook 05's feature engineering step: `get_dummies()` expands
    each categorical column into 0/1 indicator columns, and the whole
    resulting frame is cast to a single numeric dtype that scikit-learn
    estimators expect.

    Args:
        features: DataFrame with a mix of numeric and categorical columns.
        categorical_columns: Columns to one-hot encode.

    Returns:
        A new DataFrame, fully numeric (float64), with `categorical_columns`
        replaced by their one-hot indicator columns.
    """
    encoded = pd.get_dummies(features, columns=list(categorical_columns))
    return encoded.astype("float64")
