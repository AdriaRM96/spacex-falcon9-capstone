"""Data loading and cleaning for the SpaceX launch dataset.

These functions back notebook 01 (API collection) and notebook 03 (wrangling):
fetching launch records with a resilience fallback for the SpaceX API's
recurring outages, and cleaning the raw response into a one-row-per-launch
table ready for feature engineering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests


def fetch_json_with_fallback(url: str, fallback_path: str | Path, timeout: int = 15) -> dict | list:
    """Fetch JSON from `url`, falling back to a local snapshot on failure.

    The SpaceX API has had recurring outages, so any pipeline that depends on
    it needs a way to keep running. This tries the live request first and
    only falls back to `fallback_path` (a frozen JSON snapshot) if the
    request fails for any reason.

    Args:
        url: The live endpoint to request.
        fallback_path: Path to a local JSON file with a prior successful
            response, used only if the live request fails.
        timeout: Request timeout in seconds.

    Returns:
        The parsed JSON payload, from either the live request or the
        fallback file.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        with open(fallback_path) as f:
            return json.load(f)


def clean_launch_records(
    data: pd.DataFrame,
    cutoff_date: str = "2020-11-13",
) -> pd.DataFrame:
    """Reduce raw launch records to one clean row per single-core, single-payload launch.

    Mirrors the cleaning step in notebook 01: launches with more than one
    payload or more than one core (Falcon Heavy-style multi-booster flights)
    are dropped, the nested single-element lists are unwrapped, `date_utc` is
    parsed into a plain date, and only launches up to `cutoff_date` are kept
    so results stay consistent across reruns.

    Args:
        data: Raw launch records with `rocket`, `payloads`, `launchpad`,
            `cores`, `flight_number`, and `date_utc` columns.
        cutoff_date: ISO date string; launches after this date are excluded.

    Returns:
        A DataFrame with `cores` and `payloads` unwrapped to single values
        and a `date` column, filtered to launches on or before `cutoff_date`.
    """
    df = data[["rocket", "payloads", "launchpad", "cores", "flight_number", "date_utc"]].copy()
    df = df[df["cores"].map(len) == 1]
    df = df[df["payloads"].map(len) == 1]
    df["cores"] = df["cores"].map(lambda x: x[0])
    df["payloads"] = df["payloads"].map(lambda x: x[0])
    df["date"] = pd.to_datetime(df["date_utc"]).dt.date
    return df[df["date"] <= pd.to_datetime(cutoff_date).date()]


def load_reference_dataset(path: str | Path) -> pd.DataFrame:
    """Load a frozen, already-resolved launch dataset (e.g. dataset_part_1.csv).

    Used as a second-level fallback when per-record API enrichment
    (rocket/launchpad/payload/core lookups) is unavailable.
    """
    return pd.read_csv(path)
