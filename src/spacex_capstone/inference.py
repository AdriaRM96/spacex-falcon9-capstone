"""Prediction logic shared by notebook 07's business-impact section, the
model-export script, and the live /predict endpoint.

Extracted here (rather than left inline in the notebook) so there is one
implementation of "turn a payload mass / orbit / launch site into a landing
probability and an expected cost" instead of two.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

COST_WITH_REUSE = 62_000_000
COST_WITHOUT_REUSE = 165_000_000

# The one-hot orbit/launch-site categories the deployed model actually
# supports -- fixed at training time (Orbit_*/LaunchSite_* columns in
# dataset_part_3.csv), not something a caller can extend at request time.
VALID_ORBITS = ["ES-L1", "GEO", "GTO", "HEO", "ISS", "LEO", "MEO", "PO", "SO", "SSO", "VLEO"]
VALID_LAUNCH_SITES = ["CCAFS SLC 40", "KSC LC 39A", "VAFB SLC 4E"]


def expected_launch_cost(
    p_success: float,
    cost_with_reuse: float = COST_WITH_REUSE,
    cost_without_reuse: float = COST_WITHOUT_REUSE,
) -> float:
    """Blend the reused/expendable launch costs by predicted landing probability.

    See notebook 07's "Business Impact" section for the full reasoning and
    its caveats (these are SpaceX's public marketing figures, not real
    accounting data).
    """
    return p_success * cost_with_reuse + (1 - p_success) * cost_without_reuse


def build_baseline(reference: pd.DataFrame) -> pd.Series:
    """One representative "typical launch" value per column.

    One-hot indicator columns (only 0/1 values) use their mode -- the most
    common category; genuinely numeric columns use their median. Used to
    fill in every feature `build_scenario_row` doesn't explicitly set.
    """
    baseline = {}
    for col in reference.columns:
        values = reference[col]
        if set(values.unique()) <= {0, 1, 0.0, 1.0}:
            baseline[col] = values.mode()[0]
        else:
            baseline[col] = values.median()
    return pd.Series(baseline)


def build_scenario_row(
    payload_mass: float,
    orbit: str,
    launch_site: str,
    feature_names: list[str],
    baseline: pd.Series,
) -> dict:
    """Build one feature row for a hypothetical payload/orbit/site combination.

    Every column not explicitly chosen by the caller (booster serial,
    landing pad, flight number, etc.) is held at `baseline`'s value for that
    column -- the same "typical launch" baseline notebook 07 uses for its
    scenario comparisons. This is also why a real prediction request only
    needs payload mass, orbit, and launch site: those are the only
    variables a mission planner actually gets to choose, and the only ones
    that could plausibly be known ahead of a real future launch (nobody can
    supply a not-yet-assigned booster's serial number).

    Args:
        payload_mass: Payload mass in kg.
        orbit: One of `VALID_ORBITS`.
        launch_site: One of `VALID_LAUNCH_SITES`.
        feature_names: Full ordered column list the model was trained on.
        baseline: A representative row (e.g. medians/modes) to fill in every
            other column.

    Returns:
        A dict mapping every column in `feature_names` to a value, ready to
        be wrapped in a single-row DataFrame.
    """
    orbit_col = f"Orbit_{orbit}"
    site_col = f"LaunchSite_{launch_site}"
    if orbit_col not in feature_names:
        raise ValueError(f"Unknown orbit {orbit!r}; expected one of {VALID_ORBITS}")
    if site_col not in feature_names:
        raise ValueError(f"Unknown launch site {launch_site!r}; expected one of {VALID_LAUNCH_SITES}")

    orbit_cols = [c for c in feature_names if c.startswith("Orbit_")]
    site_cols = [c for c in feature_names if c.startswith("LaunchSite_")]

    row = {col: baseline[col] for col in feature_names}
    row["PayloadMass"] = payload_mass
    for col in orbit_cols:
        row[col] = 1.0 if col == orbit_col else 0.0
    for col in site_cols:
        row[col] = 1.0 if col == site_col else 0.0
    return row


@dataclass
class PredictionResult:
    p_success: float
    expected_cost_usd: float


def predict_landing(
    payload_mass: float,
    orbit: str,
    launch_site: str,
    model,
    scaler,
    feature_names: list[str],
    baseline: pd.Series,
) -> PredictionResult:
    """Predict landing probability and expected cost for one scenario.

    `model` must support `predict_proba` (wrap with
    `sklearn.calibration.CalibratedClassifierCV` first if it doesn't --
    see `train_and_export.py`).
    """
    row = build_scenario_row(payload_mass, orbit, launch_site, feature_names, baseline)
    row_df = pd.DataFrame([row], columns=feature_names)
    row_scaled = scaler.transform(row_df)
    p_success = float(model.predict_proba(row_scaled)[:, 1][0])
    return PredictionResult(p_success=p_success, expected_cost_usd=expected_launch_cost(p_success))
