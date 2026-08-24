import pandas as pd
import pytest

from spacex_capstone.inference import (
    VALID_LAUNCH_SITES,
    VALID_ORBITS,
    build_baseline,
    build_scenario_row,
    expected_launch_cost,
)


class TestExpectedLaunchCost:
    def test_certain_success_equals_reuse_cost(self):
        assert expected_launch_cost(1.0) == 62_000_000

    def test_certain_failure_equals_expendable_cost(self):
        assert expected_launch_cost(0.0) == 165_000_000

    def test_halfway_is_the_midpoint(self):
        assert expected_launch_cost(0.5) == (62_000_000 + 165_000_000) / 2

    def test_custom_cost_figures(self):
        assert expected_launch_cost(1.0, cost_with_reuse=10, cost_without_reuse=100) == 10


class TestBuildBaseline:
    def test_binary_column_uses_mode(self):
        df = pd.DataFrame({"Orbit_LEO": [1.0, 1.0, 0.0]})
        assert build_baseline(df)["Orbit_LEO"] == 1.0

    def test_numeric_column_uses_median(self):
        df = pd.DataFrame({"PayloadMass": [1000.0, 2000.0, 9000.0]})
        assert build_baseline(df)["PayloadMass"] == 2000.0

    def test_mixed_columns(self):
        df = pd.DataFrame({"Orbit_LEO": [0.0, 1.0, 1.0], "PayloadMass": [500.0, 1500.0, 2500.0]})
        baseline = build_baseline(df)
        assert baseline["Orbit_LEO"] == 1.0
        assert baseline["PayloadMass"] == 1500.0


class TestBuildScenarioRow:
    FEATURE_NAMES = ["FlightNumber", "PayloadMass", "Orbit_LEO", "Orbit_GTO", "LaunchSite_KSC LC 39A", "LaunchSite_CCAFS SLC 40"]

    def _baseline(self):
        return pd.Series(
            {
                "FlightNumber": 45.0,
                "PayloadMass": 3000.0,
                "Orbit_LEO": 1.0,
                "Orbit_GTO": 0.0,
                "LaunchSite_KSC LC 39A": 0.0,
                "LaunchSite_CCAFS SLC 40": 1.0,
            }
        )

    def test_sets_payload_mass(self):
        row = build_scenario_row(5000, "LEO", "KSC LC 39A", self.FEATURE_NAMES, self._baseline())
        assert row["PayloadMass"] == 5000

    def test_one_hot_encodes_chosen_orbit_and_zeroes_others(self):
        row = build_scenario_row(5000, "GTO", "KSC LC 39A", self.FEATURE_NAMES, self._baseline())
        assert row["Orbit_GTO"] == 1.0
        assert row["Orbit_LEO"] == 0.0

    def test_one_hot_encodes_chosen_site_and_zeroes_others(self):
        row = build_scenario_row(5000, "LEO", "KSC LC 39A", self.FEATURE_NAMES, self._baseline())
        assert row["LaunchSite_KSC LC 39A"] == 1.0
        assert row["LaunchSite_CCAFS SLC 40"] == 0.0

    def test_other_columns_come_from_baseline(self):
        row = build_scenario_row(5000, "LEO", "KSC LC 39A", self.FEATURE_NAMES, self._baseline())
        assert row["FlightNumber"] == 45.0

    def test_unknown_orbit_raises(self):
        with pytest.raises(ValueError, match="orbit"):
            build_scenario_row(5000, "MARS", "KSC LC 39A", self.FEATURE_NAMES, self._baseline())

    def test_unknown_launch_site_raises(self):
        with pytest.raises(ValueError, match="launch site"):
            build_scenario_row(5000, "LEO", "Mars Base One", self.FEATURE_NAMES, self._baseline())

    def test_valid_orbits_and_sites_are_all_accepted(self):
        # Sanity check that the module-level enums actually match a real
        # feature set's Orbit_*/LaunchSite_* columns, using the full set.
        feature_names = (
            ["PayloadMass"]
            + [f"Orbit_{o}" for o in VALID_ORBITS]
            + [f"LaunchSite_{s}" for s in VALID_LAUNCH_SITES]
        )
        baseline = pd.Series({name: 0.0 for name in feature_names})
        for orbit in VALID_ORBITS:
            for site in VALID_LAUNCH_SITES:
                build_scenario_row(1000, orbit, site, feature_names, baseline)  # should not raise
