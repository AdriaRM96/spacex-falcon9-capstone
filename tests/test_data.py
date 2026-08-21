import json

import pandas as pd
import pytest
import requests

from spacex_capstone.data import (
    clean_launch_records,
    fetch_json_with_fallback,
    load_reference_dataset,
)


class TestFetchJsonWithFallback:
    def test_returns_live_response_on_success(self, monkeypatch, tmp_path):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"source": "live"}

        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())

        fallback_path = tmp_path / "fallback.json"
        fallback_path.write_text(json.dumps({"source": "fallback"}))

        result = fetch_json_with_fallback("https://example.invalid", fallback_path)
        assert result == {"source": "live"}

    def test_falls_back_on_request_exception(self, monkeypatch, tmp_path):
        def raise_connection_error(*args, **kwargs):
            raise requests.exceptions.ConnectionError("simulated outage")

        monkeypatch.setattr(requests, "get", raise_connection_error)

        fallback_path = tmp_path / "fallback.json"
        fallback_path.write_text(json.dumps({"source": "fallback"}))

        result = fetch_json_with_fallback("https://example.invalid", fallback_path)
        assert result == {"source": "fallback"}

    def test_falls_back_on_http_error_status(self, monkeypatch, tmp_path):
        class FailingResponse:
            def raise_for_status(self):
                raise requests.exceptions.HTTPError("525 simulated")

        monkeypatch.setattr(requests, "get", lambda *a, **k: FailingResponse())

        fallback_path = tmp_path / "fallback.json"
        fallback_path.write_text(json.dumps({"source": "fallback"}))

        result = fetch_json_with_fallback("https://example.invalid", fallback_path)
        assert result == {"source": "fallback"}


class TestCleanLaunchRecords:
    def _sample_data(self):
        return pd.DataFrame(
            {
                "rocket": ["r1", "r2", "r3"],
                "payloads": [["p1"], ["p2", "p3"], ["p4"]],
                "launchpad": ["l1", "l2", "l3"],
                "cores": [["c1"], ["c2"], ["c3", "c4"]],
                "flight_number": [1, 2, 3],
                "date_utc": ["2019-01-01T00:00:00Z", "2019-06-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            }
        )

    def test_drops_multi_payload_and_multi_core_rows(self):
        df = self._sample_data()
        cleaned = clean_launch_records(df, cutoff_date="2025-01-01")
        # row 2 has 2 payloads, row 3 has 2 cores -- only row 1 survives
        assert cleaned["flight_number"].tolist() == [1]

    def test_unwraps_single_element_lists(self):
        df = self._sample_data()
        cleaned = clean_launch_records(df, cutoff_date="2025-01-01")
        assert cleaned["payloads"].iloc[0] == "p1"
        assert cleaned["cores"].iloc[0] == "c1"

    def test_filters_by_cutoff_date(self):
        df = pd.DataFrame(
            {
                "rocket": ["r1", "r2"],
                "payloads": [["p1"], ["p2"]],
                "launchpad": ["l1", "l2"],
                "cores": [["c1"], ["c2"]],
                "flight_number": [1, 2],
                "date_utc": ["2019-01-01T00:00:00Z", "2021-06-01T00:00:00Z"],
            }
        )
        cleaned = clean_launch_records(df, cutoff_date="2020-01-01")
        assert cleaned["flight_number"].tolist() == [1]


class TestLoadReferenceDataset:
    def test_loads_csv_as_dataframe(self, tmp_path):
        csv_path = tmp_path / "reference.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

        loaded = load_reference_dataset(csv_path)
        assert list(loaded.columns) == ["a", "b"]
        assert loaded["a"].tolist() == [1, 2]
