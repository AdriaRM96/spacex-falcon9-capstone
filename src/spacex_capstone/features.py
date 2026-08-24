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


def build_landing_class_from_text(booster_landing: pd.Series) -> list[int]:
    """Convert free-text booster-landing descriptions into a binary label.

    Used for the Wikipedia-scraped dataset (notebook 08), where the
    `Booster landing` cell's text gets truncated to just its first text node
    by the HTML parser (e.g. "Success (" instead of "Success (drone ship)")
    -- see `spacex_capstone.scraping.scrape_launch_page` for why. That's
    still enough to tell success from failure: any value starting with
    "Success" (case-insensitive) is a successful landing; everything else
    ("Failure (...", "No attempt", "Partial failure (...") counts as not
    landed. This is a coarser rule than `build_landing_class`'s exact-match
    lookup against known outcome strings, because the source text itself is
    coarser here.

    Args:
        booster_landing: A pandas Series of (possibly truncated) landing
            description strings.

    Returns:
        A list of 0/1 ints, one per row.
    """
    return [1 if str(value).strip().lower().startswith("success") else 0 for value in booster_landing]


# Wikipedia's launch-site links now use the base facility name (e.g. "Cape
# Canaveral") rather than the specific pad code the SpaceX-API-derived
# dataset uses (e.g. "CCAFS SLC-40" vs "CCAFS LC-40"). To combine both
# sources into one feature space, the older, more granular site names are
# coarsened down to match Wikipedia's facility-level granularity -- this
# loses the pad-level distinction for the original 90 launches, which is a
# real trade-off documented in notebook 08.
LAUNCH_SITE_TO_FACILITY = {
    "CCAFS LC-40": "Cape Canaveral",
    "CCAFS SLC-40": "Cape Canaveral",
    "CCAFS SLC 40": "Cape Canaveral",
    "KSC LC-39A": "Kennedy",
    "KSC LC 39A": "Kennedy",
    "VAFB SLC-4E": "Vandenberg",
    "VAFB SLC 4E": "Vandenberg",
}


def normalize_launch_site(launch_site: pd.Series) -> pd.Series:
    """Coarsen pad-level launch site names down to facility-level names.

    See `LAUNCH_SITE_TO_FACILITY` for why: it lets the original,
    API-sourced launch sites be combined with Wikipedia-sourced ones, which
    only distinguish at the facility level.
    """
    return launch_site.map(lambda site: LAUNCH_SITE_TO_FACILITY.get(site, site))


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
