import pandas as pd
import pytest

from spacex_capstone.features import (
    BAD_LANDING_OUTCOMES,
    LAUNCH_SITE_TO_FACILITY,
    build_landing_class,
    build_landing_class_from_text,
    normalize_launch_site,
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


class TestBuildLandingClassFromText:
    def test_success_prefix_maps_to_one(self):
        values = pd.Series(["Success (", "Success", "SUCCESS (drone ship)", "success!"])
        assert build_landing_class_from_text(values) == [1, 1, 1, 1]

    def test_non_success_prefix_maps_to_zero(self):
        values = pd.Series(["Failure (", "No attempt", "Partial failure (", ""])
        assert build_landing_class_from_text(values) == [0, 0, 0, 0]

    def test_mixed_values(self):
        values = pd.Series(["Success (", "No attempt", "Failure ("])
        assert build_landing_class_from_text(values) == [1, 0, 0]

    def test_handles_non_string_values(self):
        # None/NaN can show up from an empty scraped cell; str(value) must
        # not raise, and should count as a non-success.
        values = pd.Series(["Success (", None])
        assert build_landing_class_from_text(values) == [1, 0]


class TestNormalizeLaunchSite:
    def test_known_pad_codes_map_to_facility(self):
        sites = pd.Series(["CCAFS LC-40", "CCAFS SLC-40", "KSC LC-39A", "VAFB SLC-4E"])
        result = normalize_launch_site(sites)
        assert result.tolist() == ["Cape Canaveral", "Cape Canaveral", "Kennedy", "Vandenberg"]

    def test_already_facility_level_names_pass_through(self):
        sites = pd.Series(["Cape Canaveral", "Kennedy", "Vandenberg"])
        result = normalize_launch_site(sites)
        assert result.tolist() == ["Cape Canaveral", "Kennedy", "Vandenberg"]

    def test_unknown_site_passes_through_unchanged(self):
        sites = pd.Series(["Some Future Site"])
        assert normalize_launch_site(sites).tolist() == ["Some Future Site"]

    def test_mapping_covers_all_original_pad_codes(self):
        # Guards against silently losing a mapping if the original dataset's
        # site-code spelling ever changes.
        assert set(LAUNCH_SITE_TO_FACILITY.values()) == {"Cape Canaveral", "Kennedy", "Vandenberg"}


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
