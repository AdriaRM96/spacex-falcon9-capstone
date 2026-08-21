import pandas as pd
import pytest

from spacex_capstone.features import (
    BAD_LANDING_OUTCOMES,
    build_landing_class,
    one_hot_encode_features,
)


class TestBuildLandingClass:
    def test_successful_outcomes_map_to_one(self):
        outcomes = pd.Series(["True ASDS", "True RTLS", "True Ocean"])
        assert build_landing_class(outcomes) == [1, 1, 1]

    def test_failed_outcomes_map_to_zero(self):
        outcomes = pd.Series(["False ASDS", "False Ocean", "False RTLS", "None None", "None ASDS"])
        assert build_landing_class(outcomes) == [0, 0, 0, 0, 0]

    def test_mixed_outcomes(self):
        outcomes = pd.Series(["True ASDS", "None None", "False RTLS", "True Ocean"])
        assert build_landing_class(outcomes) == [1, 0, 0, 1]

    def test_default_bad_outcomes_matches_module_constant(self):
        outcomes = pd.Series(list(BAD_LANDING_OUTCOMES))
        assert build_landing_class(outcomes) == [0] * len(BAD_LANDING_OUTCOMES)

    def test_custom_bad_outcomes_override_default(self):
        # With a custom (smaller) bad-outcome set, an outcome that's normally
        # "bad" should now count as successful.
        outcomes = pd.Series(["False Ocean", "True ASDS"])
        result = build_landing_class(outcomes, bad_outcomes=frozenset({"True ASDS"}))
        assert result == [1, 0]

    def test_empty_series_returns_empty_list(self):
        assert build_landing_class(pd.Series([], dtype=str)) == []


class TestOneHotEncodeFeatures:
    def test_categorical_columns_become_indicator_columns(self):
        df = pd.DataFrame(
            {
                "PayloadMass": [500, 1000],
                "Orbit": ["LEO", "GTO"],
                "LaunchSite": ["A", "B"],
                "LandingPad": ["P1", "P2"],
                "Serial": ["S1", "S2"],
            }
        )
        encoded = one_hot_encode_features(df)

        assert "Orbit" not in encoded.columns
        assert "Orbit_LEO" in encoded.columns
        assert "Orbit_GTO" in encoded.columns
        assert "PayloadMass" in encoded.columns

    def test_result_is_entirely_float64(self):
        df = pd.DataFrame(
            {
                "PayloadMass": [500, 1000],
                "Orbit": ["LEO", "GTO"],
                "LaunchSite": ["A", "B"],
                "LandingPad": ["P1", "P2"],
                "Serial": ["S1", "S2"],
            }
        )
        encoded = one_hot_encode_features(df)
        assert (encoded.dtypes == "float64").all()

    def test_custom_categorical_columns(self):
        df = pd.DataFrame({"a": [1, 2], "color": ["red", "blue"]})
        encoded = one_hot_encode_features(df, categorical_columns=["color"])
        assert "color_red" in encoded.columns
        assert "color_blue" in encoded.columns
        assert "a" in encoded.columns
