"""Offline tests for downloads, quality checks and option B analysis."""

import json
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
import numpy as np

import run as app
import download
import quality
import analysis


# Fixtures and fake API


def kline(timestamp, price="100.00000000", volume="1.00000000", trades=1):
    """Build one fake API row with the usual 12 fields."""
    quote_volume = "100.00000000"
    if volume == "0":
        quote_volume = "0"
    return [timestamp, price, price, price, price, volume, timestamp + 999,
            quote_volume, trades, volume, quote_volume, "0"]


@pytest.fixture
def config(tmp_path):
    """Create a five-second config in the test folder."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-01-01T00:00:05Z",
        "data_dir": "data",
        "report_dir": "reports",
    }), encoding="utf-8")
    return app.load_config(path)


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the network calls with a small fake API."""
    # Keep file writes and data conversion real; only replace network calls.
    def setup(rows, page_size=1000, fail_on=None, server_time=2000000000000):
        """Set up fake pages, server time and an optional interrupted request."""
        # Share the request log between the fake functions.
        state = {"calls": []}
        session = Mock()

        def open_connection():
            """Return the fake session for this test."""
            return {"session": session}

        def get_time(config, client):
            """Return the chosen server time."""
            return server_time

        def get_page(config, client, start_ms, end_ms):
            """Return the next fake page and record the requested range."""
            state["calls"].append((start_ms, end_ms))
            if fail_on is not None and len(state["calls"]) == fail_on:
                raise RuntimeError("simulated connection interruption")
            selected = []
            for row in rows:
                if start_ms <= row[0] < end_ms:
                    selected.append(row)
            return selected[:page_size], {"request_url": "https://data-api.binance.vision/api/v3/klines"}


        monkeypatch.setattr(download, "open_client", open_connection)
        monkeypatch.setattr(download, "get_server_time", get_time)
        monkeypatch.setattr(download, "get_klines", get_page)
        return state
    return setup


# Config and dates

def test_naive_datetime_is_rejected():
    """Reject timestamps that leave out the timezone."""
    with pytest.raises(ValueError, match="timezone"):
        download.parse_utc("2023-01-01T00:00:00")


def test_offsets_are_converted_to_utc():
    """Convert an explicit offset to the matching UTC time."""
    assert download.parse_utc("2023-01-01T01:00:00+01:00") == download.parse_utc("2023-01-01")


def test_subseconds_are_rejected():
    """Reject fractional-second window bounds."""
    with pytest.raises(ValueError, match="whole seconds"):
        download.parse_utc("2023-01-01T00:00:00.1Z")


def test_half_open_day_windows_cover_partial_boundaries(config):
    """Split the range into two short boundary days and one full day."""
    config["start"] = download.parse_utc("2023-01-01T23:59:58Z")
    config["end"] = download.parse_utc("2023-01-03T00:00:02Z")
    windows = download.day_windows(config)
    assert [w["seconds"] for w in windows] == [2, 86400, 2]
    assert windows[0]["end"] == windows[1]["start"]
    assert windows[1]["end"] == windows[2]["start"]
    assert download.make_plan(config)["expected_seconds"] == 86404


def test_reversed_window_is_rejected(config):
    """Reject an empty or reversed time window."""
    config["end"] = config["start"]
    with pytest.raises(ValueError, match="precede"):
        app.validate_config(config)


def test_unapproved_api_source_is_rejected(config):
    """Reject an API source outside the allowlist."""
    config["base_url"] = "https://example.com"
    with pytest.raises(ValueError, match="approved Binance"):
        app.validate_config(config)


def test_endpoints_with_credentials_are_rejected(config):
    """Reject URLs containing credentials."""
    config["base_url"] = "https://user:password@data-api.binance.vision"
    with pytest.raises(ValueError):
        app.validate_config(config)


def test_scope_is_explicit(config):
    """Keep the requested interval fixed at one second."""
    config["interval"] = "1m"
    with pytest.raises(ValueError, match="restricted"):
        app.validate_config(config)


# Requests and retry limits

def response(code=200, payload=None, headers=None):
    """Build a fake requests response with a status, payload and headers."""
    result = Mock()
    result.status_code = code
    result.headers = headers or {}
    result.url = "https://data-api.binance.vision/api/v3/klines"
    if payload is None:
        payload = []
    result.json.return_value = payload
    result.text = "simulated response"
    return result


@pytest.fixture
def http(monkeypatch):
    """Set up a fake connection and record waits during request tests."""
    client = download.open_client()
    client["session"].close()

    client["session"] = Mock()
    sleep = Mock()
    monkeypatch.setattr(download.time, "sleep", sleep)
    return client, sleep


def test_inclusive_api_end_matches_exclusive_project_end(config, http):
    """Subtract one millisecond when passing the exclusive end to Binance."""
    client, sleep = http
    client["session"].get.return_value = response()
    download.get_klines(config, client, 1000, 5000)
    params = client["session"].get.call_args.kwargs["params"]
    assert params == {"symbol": "BTCUSDT", "interval": "1s", "startTime": 1000, "endTime": 4999, "limit": 1000}


def test_429_honors_retry_after_and_reuses_request(config, http):
    """Wait for Retry-After and retry the same request after a 429."""
    client, sleep = http
    # Return a 429 first, then a successful response.
    client["session"].get.side_effect = [response(429, headers={"Retry-After": "17"}), response()]
    download.get_klines(config, client, 1000, 5000)
    waits = []
    for call in sleep.call_args_list:
        waits.append(call.args[0])
    assert 17 in waits
    assert client["session"].get.call_args_list[0] == client["session"].get.call_args_list[1]


def test_ip_ban_stops_without_retry_or_endpoint_rotation(config, http):
    """Stop immediately when the exchange returns 418."""
    client, sleep = http
    client["session"].get.return_value = response(418)

    with pytest.raises(RuntimeError, match="IP ban"):
        download.get_klines(config, client, 1000, 5000)
    assert client["session"].get.call_count == 1


def test_400_is_not_retried(config, http):
    """Let a bad request fail after the first attempt."""
    client, sleep = http
    client["session"].get.return_value = response(400)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        download.get_klines(config, client, 1000, 5000)
    assert client["session"].get.call_count == 1


def test_network_retries_are_bounded(config, http):
    """Stop after the configured number of network timeouts."""
    client, sleep = http
    config["max_attempts"] = 3
    client["session"].get.side_effect = requests.Timeout("test timeout")
    with pytest.raises(RuntimeError, match="exhausted"):
        download.get_klines(config, client, 1000, 5000)
    assert client["session"].get.call_count == 3


def test_server_weight_header_causes_backoff(config, http, monkeypatch):
    """Wait when the server reports that the weight budget is nearly used up."""
    client, sleep = http
    monkeypatch.setattr(download.time, "time", Mock(return_value=1))
    client["session"].get.side_effect = [response(headers={"X-MBX-USED-WEIGHT-1M": "999"}), response()]
    download.get_klines(config, client, 1000, 2000)
    download.get_klines(config, client, 2000, 3000)
    waits = []
    for call in sleep.call_args_list:
        waits.append(call.args[0])
    assert max(waits) >= 59


def test_error_payload_is_not_misread_as_empty_data(config, http):
    """Catch an API error carried in an HTTP 200 response."""
    client, sleep = http
    client["session"].get.return_value = response(payload={"code": -1121, "msg": "Invalid symbol"})
    with pytest.raises(RuntimeError, match="error payload"):
        download.get_klines(config, client, 1000, 5000)


# Pagination and resume

def rows_for(config):
    """Build a few consecutive rows from the configured start time."""
    start = download.epoch_ms(config["start"])
    rows = []
    for i in range(5):
        rows.append(kline(start + i * 1000))
    return rows


def test_short_pages_do_not_skip_remaining_seconds(config, fake_api):
    """Keep fetching after a short page if the window still has data."""
    state = fake_api(rows_for(config), page_size=2)
    download.fetch(config)
    window = download.day_windows(config)[0]
    assert len(download.read_partition(download.partition_path(config, window))) == 5
    assert [call[0] for call in state["calls"]] == [window["start_ms"], window["start_ms"] + 2000, window["start_ms"] + 4000]
    assert download.read_json(download.manifest_path(config, window))["status"] == "materialized"


def test_interrupted_download_replays_durable_pages(config, fake_api):
    """Reuse saved pages when restarting an interrupted download."""
    rows = rows_for(config)
    fake_api(rows, page_size=2, fail_on=2)
    with pytest.raises(RuntimeError, match="interruption"):
        download.fetch(config)
    # Restart with the pages from the interrupted run still on disk.
    resumed = fake_api(rows, page_size=2)
    download.fetch(config)
    window = download.day_windows(config)[0]
    assert resumed["calls"][0][0] == window["start_ms"] + 2000
    assert len(download.read_partition(download.partition_path(config, window))) == 5


def test_completed_partition_is_reused_without_kline_calls(config, fake_api):
    """Reuse a verified daily file without asking for its candles again."""
    fake_api(rows_for(config))
    download.fetch(config)
    state = fake_api([])
    result = download.fetch(config)
    assert state["calls"] == []
    assert result[0]["reused"]


def test_cache_corruption_is_not_silently_accepted(config, fake_api):
    """Reject a cached payload that no longer matches its hash."""
    fake_api(rows_for(config), page_size=2, fail_on=2)
    with pytest.raises(RuntimeError):
        download.fetch(config)
    cache = list((config["data_dir"] / "raw").rglob("*.json.gz"))[0]
    page = download.read_gzip_json(cache)
    # Change the payload but leave its saved hash alone.
    page["data"][0][1] = "999"
    download.write_gzip_json(cache, page)
    fake_api(rows_for(config))
    with pytest.raises(ValueError, match="payload hash"):
        download.fetch(config)


def test_modified_parquet_is_not_overwritten_or_trusted(config, fake_api):
    """Stop when a completed Parquet file has been changed."""
    fake_api(rows_for(config))
    download.fetch(config)
    output = download.partition_path(config, download.day_windows(config)[0])
    output.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        download.fetch(config)


def test_empty_response_still_materializes_for_explicit_gap_reporting(config, fake_api):
    """Save an empty response so the audit can report the missing window."""
    fake_api([])
    download.fetch(config)
    output = download.partition_path(config, download.day_windows(config)[0])
    assert len(download.read_partition(output)) == 0


def test_future_or_unfinished_candle_is_rejected(config, fake_api):
    """Reject a range extending past the last closed second."""
    state = fake_api([], server_time=download.epoch_ms(config["end"]) - 1)
    with pytest.raises(ValueError, match="unfinished"):
        download.fetch(config)
    assert state["calls"] == []


def test_2025_rest_response_still_uses_milliseconds(config, fake_api):
    """Read REST timestamps as milliseconds for 2025 data too."""
    config["start"] = download.parse_utc("2025-01-01")
    config["end"] = download.parse_utc("2025-01-01T00:00:05Z")
    fake_api(rows_for(config))
    download.fetch(config)
    frame = download.read_partition(download.partition_path(config, download.day_windows(config)[0]))
    assert frame.iloc[0]["open_time"].year == 2025


# Coverage and row checks

def make_rows(start, seconds):
    """Build the requested number of consecutive second candles."""
    rows = []
    for i in range(seconds):
        rows.append(kline(start + i * 1000))
    return rows


def test_exact_leading_middle_and_trailing_gap_intervals(config):
    """Report exact bounds for gaps at the start, middle and end."""
    window = download.day_windows(config)[0]
    frame = download.rows_to_frame([kline(window["start_ms"] + 1000), kline(window["start_ms"] + 3000)])
    result, gaps = quality.audit_frame(frame, window)
    assert result["missing_seconds"] == 3
    assert [g["missing_seconds"] for g in gaps] == [1, 1, 1]
    assert gaps[0]["start_utc"] == "2023-01-01T00:00:00Z"
    assert gaps[-1]["end_exclusive_utc"] == "2023-01-01T00:00:05Z"


def test_duplicate_does_not_hide_a_missing_second(config):
    """Flag a duplicate even when total rows match the expected count."""
    window = download.day_windows(config)[0]
    rows = []
    # Second 1 is duplicated and second 2 is missing.
    for i in [0, 1, 1, 3, 4]:
        rows.append(kline(window["start_ms"] + i * 1000))
    result, gaps = quality.audit_frame(download.rows_to_frame(rows), window)
    assert result["rows"] == result["expected_seconds"]
    assert result["duplicate_seconds"] == 1
    assert result["missing_seconds"] == 1
    assert result["status"] == "FAIL"


def test_zero_volume_is_preserved_and_is_not_a_gap(config):
    """Count zero-volume candles as observed seconds."""
    window = download.day_windows(config)[0]
    rows = []
    for i in range(5):
        rows.append(kline(window["start_ms"] + i * 1000, volume="0", trades=0))
    result, gaps = quality.audit_frame(download.rows_to_frame(rows), window)
    assert result["status"] == "PASS"
    assert result["zero_volume_seconds"] == 5
    assert gaps == []


def test_ohlc_nonfinite_negative_and_taker_violations_are_detected(config):
    """Catch bad OHLC values, infinity, negative volume and excess taker volume."""
    window = download.day_windows(config)[0]
    rows = make_rows(window["start_ms"], 5)
    rows[0][2] = "99"  # High is below open.
    rows[1][3] = "-1"  # Negative low.
    rows[2][5] = "-1"  # Negative BTC volume.
    rows[3][4] = "Infinity"  # Infinite close.
    rows[4][9] = "2"  # Taker BTC volume exceeds total BTC volume.
    result, gaps = quality.audit_frame(download.rows_to_frame(rows), window)
    for rule in ["ohlc_violation_rows", "non_finite_rows", "non_positive_price_rows",
                 "negative_volume_rows", "taker_volume_violation_rows"]:
        assert result[rule] > 0
    assert result["status"] == "FAIL"


def test_missing_partition_cannot_pass(config):
    """Fail the audit when a daily file is missing."""
    summary, daily, gaps = quality.audit(config)
    assert not summary["ready_for_analysis"]
    assert summary["missing_seconds"] == 5
    assert summary["trusted_partitions"] == 0
    assert len(summary["artifact_issues"]) == 1


def test_audit_rechecks_parquet_hash(config, fake_api):
    """Catch a Parquet file changed after download."""
    fake_api(make_rows(download.epoch_ms(config["start"]), 5))
    download.fetch(config)
    download.partition_path(config, download.day_windows(config)[0]).write_bytes(b"corrupt")
    summary, daily, gaps = quality.audit(config)
    assert summary["quality_status"] == "FAIL"
    assert "SHA-256" in summary["artifact_issues"][0]["error"]


def test_incomplete_receipt_cannot_count_a_partition_twice(config, fake_api):
    """Count an invalid receipt once and leave its window uncovered."""
    fake_api(make_rows(download.epoch_ms(config["start"]), 5))
    download.fetch(config)
    path = download.manifest_path(config, download.day_windows(config)[0])
    receipt = download.read_json(path)
    del receipt["raw_pages"]
    download.write_json(path, receipt)
    summary, daily, gaps = quality.audit(config)
    assert summary["expected_seconds"] == 5
    assert summary["observed_rows"] == 0
    assert summary["trusted_partitions"] == 0
    assert len(daily) == 1


def test_clean_data_passes_with_explicit_utc_schema(config, fake_api):
    """Pass clean data with explicit UTC timestamps."""
    fake_api(make_rows(download.epoch_ms(config["start"]), 5))
    download.fetch(config)
    summary, daily, gaps = quality.audit(config)
    assert summary["ready_for_analysis"]
    assert gaps.empty
    assert daily.iloc[0]["expected_seconds"] == 5
    assert daily.iloc[0]["coverage_fraction"] == 1
    frame = download.read_partition(download.partition_path(config, download.day_windows(config)[0]))
    assert str(frame["open_time"].dt.tz) == "UTC"


def test_partial_days_across_midnight_have_correct_coverage(config, fake_api):
    """Use each partial day's requested seconds as its coverage denominator."""
    config["start"] = download.parse_utc("2023-01-01T23:59:58Z")
    config["end"] = download.parse_utc("2023-01-02T00:00:02Z")
    fake_api(make_rows(download.epoch_ms(config["start"]), 4))
    download.fetch(config)
    summary, daily, gaps = quality.audit(config)
    assert summary["ready_for_analysis"]
    assert daily["expected_seconds"].tolist() == [2, 2]
    assert daily["coverage_fraction"].tolist() == [1, 1]


def test_schema_rejects_wrong_timestamp_unit(config):
    """Reject microsecond values passed as REST milliseconds."""
    window = download.day_windows(config)[0]
    with pytest.raises(ValueError, match="timestamp units"):
        download.page_open_times([kline(window["start_ms"] * 1000)], window["start_ms"], window["end_ms"])


def test_schema_rejects_noninteger_trade_counts(config):
    """Reject fractional trade counts."""
    window = download.day_windows(config)[0]
    with pytest.raises(ValueError, match="JSON integers"):
        download.page_open_times([kline(window["start_ms"], trades=1.5)], window["start_ms"], window["end_ms"])


def test_lock_prevents_concurrent_writers_and_releases_after_failure(config):
    """Reject a second writer and release the first writer's lock on exit."""
    lock = download.create_lock(config["data_dir"])
    try:
        with pytest.raises(RuntimeError, match="lock exists"):
            download.create_lock(config["data_dir"])
    finally:
        download.remove_lock(lock)
    assert not (config["data_dir"] / ".writer.lock").exists()


# Check outputs and offline CLI

def test_check_saves_only_minimal_results(config, fake_api):
    """Save exactly the quality summary and two detail tables."""
    start = download.epoch_ms(config["start"])
    rows = []
    for i in range(5):
        rows.append(kline(start + i * 1000))
    fake_api(rows)
    download.fetch(config)
    summary = quality.check_data(config)
    assert summary["ready_for_analysis"]

    output = config["report_dir"]
    names = set()
    for path in output.iterdir():
        names.add(path.name)
    assert names == {"quality.json", "daily_quality.csv", "gaps.csv"}
    saved = json.loads((output / "quality.json").read_text(encoding="utf-8"))
    assert saved["observed_rows"] == 5
    daily = pd.read_csv(output / "daily_quality.csv")
    assert daily.iloc[0]["coverage_fraction"] == 1
    assert pd.read_csv(output / "gaps.csv").empty


def test_missing_data_saves_failure_details(config):
    """Save useful failure details when local data is missing."""
    summary = quality.check_data(config)
    assert summary["quality_status"] == "FAIL"
    assert summary["missing_seconds"] == 5
    output = config["report_dir"]
    assert (output / "quality.json").exists()
    gaps = pd.read_csv(output / "gaps.csv")
    assert gaps["missing_seconds"].sum() == 5


def test_cli_check_returns_nonzero_for_missing_data(tmp_path):
    """Return exit code 2 when the requested data is missing."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "start": "2023-01-01",
        "end": "2023-01-02",
        "data_dir": "data",
        "report_dir": "reports",
    }), encoding="utf-8")
    assert app.main(["--config", str(path), "check"]) == 2


def test_cli_check_is_offline(config, fake_api, monkeypatch):
    """Complete a local check with all network calls disabled."""
    start = download.epoch_ms(config["start"])
    rows = []
    for i in range(5):
        rows.append(kline(start + i * 1000))
    fake_api(rows)
    download.fetch(config)
    download_call = Mock(side_effect=AssertionError("check must not download"))
    network = Mock(side_effect=AssertionError("check must stay offline"))
    monkeypatch.setattr("run.fetch", download_call)
    monkeypatch.setattr("requests.sessions.Session.request", network)
    path = config["data_dir"].parent / "config.json"
    assert app.main(["--config", str(path), "check"]) == 0
    download_call.assert_not_called()
    network.assert_not_called()


# File writes and retries

def test_atomic_replace_retries_transient_windows_sharing_violation(tmp_path, monkeypatch):
    """Retry two temporary permission errors before replacing the file."""
    source, target = tmp_path / "new", tmp_path / "old"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    # Keep the real replace function for the successful third attempt.
    actual_replace = download.os.replace
    calls = []

    def transient_replace(left, right):
        """Fail twice, then call the real file replacement."""
        calls.append((left, right))
        if len(calls) < 3:
            raise PermissionError("temporary Windows file sharing conflict")
        return actual_replace(left, right)

    monkeypatch.setattr(download.os, "replace", transient_replace)
    # Record waits without slowing down the test.
    monkeypatch.setattr(download.time, "sleep", Mock())
    download.atomic_replace(source, target)
    assert len(calls) == 3
    assert target.read_bytes() == b"new"


def test_failed_atomic_write_preserves_existing_artifact(tmp_path, monkeypatch):
    """Keep the old file and remove the temporary file if replacement keeps failing."""
    target = tmp_path / "existing.json"
    target.write_bytes(b"original")
    monkeypatch.setattr(download.os, "replace", Mock(side_effect=PermissionError("permanent")))
    monkeypatch.setattr(download.time, "sleep", Mock())
    with pytest.raises(PermissionError):
        download.atomic_bytes(target, b"replacement")
    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob("*.tmp")) == []


# Known halt exception


def known_gap_rows(config):
    """Keep two observed seconds on each side of the documented gap."""
    config["start"] = download.parse_utc("2023-03-24T12:39:40Z")
    config["end"] = download.parse_utc("2023-03-24T14:00:02Z")
    window = download.day_windows(config)[0]
    times = [window["start_ms"], window["start_ms"] + 1000,
             window["end_ms"] - 2000, window["end_ms"] - 1000]
    rows = []
    for timestamp in times:
        rows.append(kline(timestamp))
    return window, rows


def test_documented_gap_is_warning_without_imputation(config):
    """Warn for the exact known gap and preserve the original rows."""
    window, rows = known_gap_rows(config)
    frame = download.rows_to_frame(rows)
    before = frame.copy(deep=True)
    result, gaps = quality.audit_frame(frame, window)
    assert result["status"] == "WARNING"
    assert result["missing_seconds"] == 4818
    assert result["known_missing_seconds"] == 4818
    assert result["unexpected_missing_seconds"] == 0
    assert result["rows"] == 4
    assert result["coverage_fraction"] < 1
    assert len(gaps) == 1
    assert gaps[0]["status"] == "WARNING"
    assert gaps[0]["start_utc"] == "2023-03-24T12:39:42Z"
    assert gaps[0]["end_exclusive_utc"] == "2023-03-24T14:00:00Z"
    pd.testing.assert_frame_equal(frame, before)


def test_one_extra_missing_second_at_either_boundary_fails(config):
    """Fail if either side of the known gap is missing one extra second."""
    window, rows = known_gap_rows(config)
    for missing_row in [1, 2]:
        candidate = rows.copy()
        del candidate[missing_row]
        result, gaps = quality.audit_frame(download.rows_to_frame(candidate), window)
        assert result["status"] == "FAIL"
        assert result["missing_seconds"] == 4819
        assert result["known_missing_seconds"] == 4818
        assert result["unexpected_missing_seconds"] == 1
        assert len(gaps) == 2
        assert set(gap["status"] for gap in gaps) == {"WARNING", "FAIL"}


def test_same_length_gap_on_another_day_is_not_exempt(config):
    """Reject a gap of the same length on a different date."""
    window, rows = known_gap_rows(config)
    shift = 86400 * 1000
    for row in rows:
        row[0] += shift
        row[6] += shift
    window = download.make_window(download.parse_utc("2023-03-25T12:39:40Z"),
                                  download.parse_utc("2023-03-25T14:00:02Z"))
    result, gaps = quality.audit_frame(download.rows_to_frame(rows), window)
    assert result["status"] == "FAIL"
    assert result["known_missing_seconds"] == 0
    assert result["unexpected_missing_seconds"] == 4818
    assert gaps[0]["status"] == "FAIL"


def test_partial_request_inside_documented_gap_warns(config):
    """Limit the warning to the requested part of the known gap."""
    window = download.make_window(download.parse_utc("2023-03-24T12:40:00Z"),
                                  download.parse_utc("2023-03-24T12:40:05Z"))
    result, gaps = quality.audit_frame(download.rows_to_frame([]), window)
    assert result["status"] == "WARNING"
    assert result["known_missing_seconds"] == 5
    assert result["unexpected_missing_seconds"] == 0
    assert gaps[0]["missing_seconds"] == 5


def test_bad_values_inside_documented_interval_still_fail(config, fake_api):
    """Fail on a bad value even inside the documented halt interval."""
    config["start"] = download.parse_utc("2023-03-24T12:40:00Z")
    config["end"] = download.parse_utc("2023-03-24T12:40:05Z")
    fake_api([kline(download.epoch_ms(config["start"]), price="-1")])
    download.fetch(config)
    summary, daily, gaps = quality.audit(config)
    assert summary["quality_status"] == "FAIL"
    assert not summary["ready_for_analysis"]
    assert summary["known_missing_seconds"] == 4
    assert summary["unexpected_missing_seconds"] == 0
    assert summary["violations"]["non_positive_price_rows"] == 1


def test_duplicates_with_known_gap_still_fail(config):
    """Fail on duplicate rows alongside the known gap."""
    window, rows = known_gap_rows(config)
    rows.insert(1, rows[0].copy())
    result, gaps = quality.audit_frame(download.rows_to_frame(rows), window)
    assert result["status"] == "FAIL"
    assert result["known_missing_seconds"] == 4818
    assert result["duplicate_seconds"] == 1


def test_missing_file_inside_documented_interval_still_fails(config):
    """Fail on a missing file even when its range lies inside the halt."""
    config["start"] = download.parse_utc("2023-03-24T12:40:00Z")
    config["end"] = download.parse_utc("2023-03-24T12:40:05Z")
    summary, daily, gaps = quality.audit(config)
    assert summary["quality_status"] == "FAIL"
    assert not summary["ready_for_analysis"]
    assert summary["known_missing_seconds"] == 0
    assert summary["unexpected_missing_seconds"] == 5
    assert len(summary["artifact_issues"]) == 1
    assert gaps.iloc[0]["status"] == "FAIL"


def test_cli_warning_succeeds_and_original_files_are_unchanged(config, fake_api):
    """Return success for the known warning and preserve all source file hashes."""
    window, rows = known_gap_rows(config)
    fake_api(rows)
    download.fetch(config)
    path = config["data_dir"].parent / "config.json"
    settings = download.read_json(path)
    settings["start"] = download.iso(config["start"])
    settings["end"] = download.iso(config["end"])
    download.write_json(path, settings)

    data_path = download.partition_path(config, window)
    paths = [data_path]
    paths.extend((config["data_dir"] / "raw").rglob("*.json.gz"))
    hashes = {}
    for data_file in paths:
        hashes[data_file] = download.sha256_file(data_file)

    assert app.main(["--config", str(path), "check"]) == 0
    summary = download.read_json(config["report_dir"] / "quality.json")
    assert summary["quality_status"] == "WARNING"
    assert summary["ready_for_analysis"]
    assert summary["known_missing_seconds"] == 4818
    assert summary["unexpected_missing_seconds"] == 0
    assert summary["missing_seconds"] == 4818
    assert summary["observed_rows"] == 4
    assert len(summary["warnings"]) == 1
    gap_table = pd.read_csv(config["report_dir"] / "gaps.csv")
    assert gap_table.iloc[0]["status"] == "WARNING"
    for data_file in paths:
        assert download.sha256_file(data_file) == hashes[data_file]
    assert len(download.read_partition(data_path)) == 4


def test_known_and_unexpected_gaps_make_global_result_fail(config, fake_api):
    """Keep the overall result at FAIL when an unexpected gap is also present."""
    window, rows = known_gap_rows(config)
    del rows[1]
    fake_api(rows)
    download.fetch(config)
    summary, daily, gaps = quality.audit(config)
    assert summary["quality_status"] == "FAIL"
    assert not summary["ready_for_analysis"]
    assert summary["known_missing_seconds"] == 4818
    assert summary["unexpected_missing_seconds"] == 1
    assert len(summary["warnings"]) == 1


# Analysis gaps, time boundaries and denominators

def analysis_sample(seconds=240):
    """Build consecutive candles with a slowly rising price for analysis tests."""
    start = download.epoch_ms(download.parse_utc("2023-01-01"))
    rows = []
    for i in range(seconds):
        rows.append(kline(start + i * 1000, price=str(100 + i / 100)))
    return download.rows_to_frame(rows)


def test_analysis_second_return_crosses_midnight_but_not_gap():
    """Carry returns across a continuous midnight and reset them at a gap."""
    frame = analysis_sample(4).drop(index=2).reset_index(drop=True)
    before = frame.copy(deep=True)
    previous = pd.Timestamp("2022-12-31T23:59:59Z")
    result = analysis.second_returns(frame, previous, 50)
    assert result.iloc[0] == pytest.approx(np.log(2))
    assert result.iloc[1] == pytest.approx(np.log(100.01 / 100))
    assert pd.isna(result.iloc[2])
    pd.testing.assert_frame_equal(frame, before)


def test_analysis_partial_minute_blocks_current_and_future_returns():
    """Break current and forward returns at an incomplete minute."""
    frame = analysis_sample().drop(index=80).reset_index(drop=True)
    start = download.parse_utc("2023-01-01")
    window = download.make_window(start, start + pd.Timedelta(minutes=4))
    minute, hours, daily = analysis.aggregate_day(frame, window, analysis.second_returns(frame))
    features = analysis.minute_features(minute)
    assert features["n"].tolist() == [60, 59, 60, 60]
    assert features["return"].iloc[:3].isna().all()
    assert np.isfinite(features["return"].iloc[3])
    assert pd.isna(analysis.future_returns(features, 3).iloc[0])
    assert np.isfinite(analysis.future_returns(features, 1).iloc[2])
    assert daily["rows"] == 239
    assert not daily["complete_day"]


def test_analysis_zero_volume_minute_is_observed_without_imbalance():
    """Keep a zero-volume minute while leaving its imbalance undefined."""
    frame = analysis_sample(120)
    for column in ["open", "high", "low", "close"]:
        frame[column] = 100.0
    for column in download.VOLUME_COLUMNS:
        frame.loc[60:, column] = 0.0
    frame.loc[60:, "trade_count"] = 0
    start = download.parse_utc("2023-01-01")
    window = download.make_window(start, start + pd.Timedelta(minutes=2))
    minute, hours, daily = analysis.aggregate_day(frame, window, analysis.second_returns(frame))
    result = analysis.minute_features(minute)
    assert result.iloc[1]["n"] == 60
    assert result.iloc[1]["return"] == 0
    assert pd.isna(result.iloc[1]["imbalance"])
    assert daily["zero_volume_seconds"] == 60


def test_analysis_incremental_correlation_matches_direct_calculation():
    """Match direct correlation with chunked sums and leave constant-series results undefined."""
    totals = {}
    key = ("All", "1s", "return", 1)
    analysis.add_correlation(totals, key, [1, 2, np.nan], [3, 1, 9])
    analysis.add_correlation(totals, key, [4, 8], [5, 2])
    result = analysis.correlation_table(totals).iloc[0]
    assert result["correlation"] == pytest.approx(np.corrcoef([1, 2, 4, 8], [3, 1, 5, 2])[0, 1])
    assert result["pairs"] == 4
    totals = {}
    analysis.add_correlation(totals, key, [1, 1, 1], [1, 2, 3])
    assert pd.isna(analysis.correlation_table(totals).iloc[0]["correlation"])


def test_analysis_weekly_comparison_requires_all_seven_complete_days():
    """Compare daily averages only when all seven days in the week are complete."""
    rows = []
    for date in pd.date_range("2023-12-25", periods=17, tz="UTC"):
        volume = 10
        rv = 2
        if date.dayofweek >= 5:
            volume = 4
            rv = 1
        rows.append({"date": date, "weekday": date.dayofweek,
                     "complete_day": date != pd.Timestamp("2024-01-03", tz="UTC"),
                     "volume_btc": volume, "rv_pct": rv})
    result = analysis.weekly_table(pd.DataFrame(rows))
    assert len(result) == 1
    assert result.iloc[0]["volume_ratio"] == pytest.approx(0.4)
    assert result.iloc[0]["rv_ratio"] == pytest.approx(0.5)


def test_analysis_hourly_shares_weight_days_equally():
    """Give both complete days equal weight despite different volume totals."""
    dates = pd.date_range("2023-01-01", periods=3, tz="UTC")
    daily = pd.DataFrame({"date": dates, "complete_day": [True, True, False]})
    hours = []
    for i, date in enumerate(dates):
        for hour in range(24):
            volume = [30 / 23, 800 / 23, 0][i]
            if hour == 14:
                volume = [70, 200, 1000000][i]
            hours.append({"date": date, "hour": hour, "volume": volume, "rv2": 1, "n": 3600})
    result = analysis.hourly_table(daily, pd.DataFrame(hours))
    row = result[(result["year"] == "All") & (result["hour_utc"] == 14)].iloc[0]
    assert row["volume_share_pct"] == pytest.approx(45)
    assert row["volume_days"] == 2


def test_analysis_refuses_failed_quality(config, fake_api, tmp_path):
    """Stop analysis before writing outputs if the quality check fails."""
    start = download.epoch_ms(config["start"])
    fake_api([kline(start)])
    download.fetch(config)
    assert analysis.main(["--config", str(tmp_path / "config.json")]) == 2
    assert not (config["report_dir"] / "analysis").exists()
    assert not (config["data_dir"] / ".writer.lock").exists()


def test_analysis_small_range_writes_outputs_without_changing_data(config, fake_api, tmp_path):
    """Handle a short range, save all charts and preserve the source file."""
    start = download.epoch_ms(config["start"])
    rows = []
    for i in range(5):
        rows.append(kline(start + i * 1000))
    fake_api(rows)
    download.fetch(config)
    window = download.day_windows(config)[0]
    source = download.partition_path(config, window)
    before = download.sha256_file(source)
    assert analysis.main(["--config", str(tmp_path / "config.json")]) == 0
    output = config["report_dir"] / "analysis"
    summary = download.read_json(output / "summary.json")
    assert summary["quality"]["observed_rows"] == 5
    assert summary["complete_minutes"] == 0
    assert summary["complete_weeks_for_comparison"] == 0
    assert pd.read_csv(output / "tables" / "weekly.csv").empty
    assert pd.read_csv(output / "tables" / "session_daily.csv").empty
    assert len(list((output / "figures").glob("*.png"))) == 14
    assert pd.read_csv(output / "tables" / "forecast_scores.csv").empty
    inference = pd.read_csv(output / "tables" / "inference.csv")
    assert inference["p_value"].isna().all()
    assert not inference["reject_null_5pct"].any()
    assert download.sha256_file(source) == before


def session_sample(dates):
    """Build full minute days with constant rates for session boundary checks."""
    parts = []
    for date in dates:
        times = pd.date_range(date, periods=1440, freq="min")
        parts.append(pd.DataFrame({"volume": 60.0, "quote_volume": 6000.0, "trade_count": 120,
                                   "n": 60, "rv2": 0.000001, "zero_seconds": 6, "valid_returns": 60}, index=times))
    minute = pd.concat(parts)
    daily = pd.DataFrame({"date": dates, "complete_day": True})
    return minute, daily


@pytest.mark.parametrize("day,london_start,new_york_start", [
    ("2024-03-09", 9, 14), ("2024-03-10", 9, 13),
    ("2024-03-30", 9, 13), ("2024-03-31", 8, 13),
    ("2024-10-26", 8, 13), ("2024-10-27", 9, 13),
    ("2024-11-02", 9, 13), ("2024-11-03", 9, 14),
])
def test_session_calendar_follows_separate_dst_changes(day, london_start, new_york_start):
    """Check both DST transitions and the weeks when the US and UK offsets differ."""
    dates = pd.DatetimeIndex([pd.Timestamp(day, tz="UTC")])
    minute, daily = session_sample(dates)
    calendar = analysis.session_calendar(daily).set_index("session")
    assert calendar.loc["Hong Kong", "start_utc"].hour == 1
    assert calendar.loc["London", "start_utc"].hour == london_start
    assert calendar.loc["New York", "start_utc"].hour == new_york_start
    labels = analysis.session_labels(minute.index)
    for name, zone in analysis.SESSIONS:
        assert labels[name].sum() == 480
        assert calendar.loc[name, "end_utc"] - calendar.loc[name, "start_utc"] == pd.Timedelta(hours=8)
        assert calendar.loc[name, "is_dst"] == (calendar.loc[name, "start_utc"].tz_convert(zone).dst().total_seconds() != 0)


def test_session_labels_keep_half_open_boundaries_and_overlaps():
    """Keep the open minute, exclude the close minute and preserve simultaneous windows."""
    times = pd.to_datetime(["2024-07-01T00:59Z", "2024-07-01T01:00Z", "2024-07-01T07:59Z",
                            "2024-07-01T08:00Z", "2024-07-01T09:00Z", "2024-07-01T13:00Z",
                            "2024-07-01T16:00Z", "2024-07-01T21:00Z"])
    result = analysis.session_labels(times)
    assert result["session"].tolist() == ["Outside", "Hong Kong", "Hong Kong", "Hong Kong + London",
                                           "London", "London + New York", "New York", "Outside"]
    assert result.iloc[3]["Hong Kong"]
    assert result.iloc[3]["London"]
    assert not result.iloc[4]["Hong Kong"]


def test_session_buckets_preserve_totals_and_adjust_for_duration():
    """Partition every complete day once and keep rate estimates stable across overlap lengths."""
    dates = pd.to_datetime(["2024-01-15", "2024-03-18", "2024-07-15"], utc=True)
    minute, daily = session_sample(dates)
    before = minute.copy()
    calendar = analysis.session_calendar(daily)
    result = analysis.session_daily_table(minute, daily, calendar)
    exclusive = result[result["scheme"] == "exclusive"]
    grouped = exclusive.groupby("date")
    assert (grouped["minutes"].sum() == 1440).all()
    assert (grouped["n"].sum() == 86400).all()
    for column in ["volume", "quote_volume", "trade_count", "rv2", "zero_seconds", "valid_returns"]:
        expected = minute.groupby(minute.index.normalize())[column].sum()
        np.testing.assert_allclose(grouped[column].sum(), expected)
    regional = result[result["scheme"] == "regional"]
    assert len(regional) == 9
    assert (regional["minutes"] == 480).all()
    overlap = exclusive[exclusive["session"] == "London + New York"].set_index("date")
    assert overlap.loc[dates[0], "minutes"] == 180
    assert overlap.loc[dates[1], "minutes"] == 240
    assert overlap.loc[dates[2], "minutes"] == 180
    assert len(exclusive[exclusive["session"] == "Hong Kong + London"]) == 1
    np.testing.assert_allclose(result["turnover_intensity"], 1)
    np.testing.assert_allclose(result["variance_intensity"], 1)
    np.testing.assert_allclose(result["zero_volume_pct"], 10)
    summary = analysis.session_summary_table(result)
    np.testing.assert_allclose(summary["turnover_usdt_per_minute"], 6000)
    np.testing.assert_allclose(summary["variance_bps2_per_minute"], 100)
    pd.testing.assert_frame_equal(minute, before)


def test_sessions_exclude_incomplete_days_and_keep_zero_rates_undefined():
    """Exclude the halt day and leave relative intensity undefined when the whole day is zero."""
    dates = pd.date_range("2023-03-23", periods=3, tz="UTC")
    minute, daily = session_sample(dates)
    daily.loc[1, "complete_day"] = False
    minute.loc[:, ["volume", "quote_volume", "trade_count", "rv2"]] = 0
    minute["zero_seconds"] = 60
    calendar = analysis.session_calendar(daily)
    result = analysis.session_daily_table(minute, daily, calendar)
    assert not result["date"].isin([dates[1]]).any()
    assert result["turnover_intensity"].isna().all()
    assert result["variance_intensity"].isna().all()
    assert (result["zero_volume_pct"] == 100).all()
    summary = analysis.session_summary_table(result)
    assert (summary["turnover_days"] == 0).all()
    assert (summary["variance_days"] == 0).all()
    assert (summary["turnover_usdt_per_minute"] == 0).all()


def test_session_intensity_averages_days_equally():
    """Give quiet and busy days equal weight in the normalized session comparison."""
    dates = pd.date_range("2024-01-15", periods=2, tz="UTC")
    minute, daily = session_sample(dates)
    second_day = minute.index.normalize() == dates[1]
    minute.loc[second_day, "quote_volume"] *= 100
    hk = analysis.session_labels(minute.index)["Hong Kong"]
    minute.loc[hk & ~second_day, "quote_volume"] *= 2
    result = analysis.session_daily_table(minute, daily, analysis.session_calendar(daily))
    summary = analysis.session_summary_table(result)
    row = summary[(summary["year"] == "All") & (summary["day_type"] == "All") &
                  (summary["scheme"] == "regional") & (summary["session"] == "Hong Kong")].iloc[0]
    assert row["mean_turnover_intensity"] == pytest.approx((1.5 + 1) / 2)
    assert row["turnover_usdt_per_minute"] == pytest.approx((12000 + 600000) / 2)


def test_session_weekends_pair_only_complete_weeks():
    """Match within weeks despite changing activity levels and exclude incomplete weeks."""
    dates = pd.date_range("2024-01-01", periods=21, tz="UTC")
    minute, daily = session_sample(dates)
    daily.loc[16, "complete_day"] = False
    weekend = minute.index.dayofweek >= 5
    later = minute.index >= dates[7]
    minute.loc[later, "quote_volume"] *= 100
    minute.loc[weekend, "quote_volume"] *= 0.4
    minute.loc[weekend, "rv2"] *= 0.25
    rows = analysis.session_daily_table(minute, daily, analysis.session_calendar(daily))
    weekly, summary = analysis.session_weekend_tables(rows)
    assert (weekly.groupby(["scheme", "session"]).size() == 2).all()
    np.testing.assert_allclose(weekly["turnover_ratio"], 0.4)
    np.testing.assert_allclose(weekly["variance_ratio"], 0.25)
    assert (summary["weeks"] == 2).all()
    assert (summary["fraction_below_one"] == 1).all()
    assert summary["ci95_low"].isna().all()
    assert summary["ci95_high"].isna().all()
    assert (summary["status"] == "insufficient_blocks").all()


def test_session_hourly_profiles_sum_to_one_hundred():
    """Keep day types and DST regimes separate while normalizing each full day."""
    dates = pd.to_datetime(["2024-01-15", "2024-03-18", "2024-07-15", "2024-07-20"], utc=True)
    minute, daily = session_sample(dates)
    hourly = analysis.session_hourly_table(minute, daily, analysis.session_calendar(daily))
    pooled = hourly[hourly["year"] == "All"]
    assert pooled["dst_regime"].nunique() == 3
    assert set(pooled["day_type"]) == {"Weekday", "Weekend"}
    totals = pooled.groupby(["dst_regime", "day_type"])[["turnover_share_pct", "variance_share_pct"]].sum()
    np.testing.assert_allclose(totals, 100)
    np.testing.assert_allclose(pooled["turnover_share_pct"], 100 / 24)


def validation_sample():
    """Build consecutive minutes around both yearly cutoffs with a gap between them."""
    first = pd.date_range("2024-12-31T22:00:00Z", periods=240, freq="min")
    second = pd.date_range("2025-12-31T22:00:00Z", periods=240, freq="min")
    times = first.append(second)
    rng = np.random.default_rng(18)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(times))))
    raw = pd.DataFrame({"n": 60, "close": close, "quote_volume": 100.0,
                        "taker_buy_quote_volume": rng.uniform(1, 99, len(times))}, index=times)
    return raw


def test_forecast_features_do_not_change_when_future_data_changes():
    """Keep earlier finished-minute features unchanged after editing future prices and volume."""
    raw = validation_sample()
    cutoff = pd.Timestamp("2025-01-01T00:00:00Z")
    before = analysis.minute_features(raw)
    changed = raw.copy()
    changed.loc[changed.index >= cutoff, "close"] *= 2
    changed.loc[changed.index >= cutoff, "quote_volume"] *= 5
    after = analysis.minute_features(changed)
    earlier = raw.index < cutoff
    pd.testing.assert_frame_equal(before.loc[earlier, ["return", "imbalance"]], after.loc[earlier, ["return", "imbalance"]])
    samples = analysis.forecast_samples(before, 5)
    assert (samples["available_at"] == samples.index + pd.Timedelta(minutes=1)).all()
    assert (samples["target_end"] - samples["available_at"] == pd.Timedelta(minutes=5)).all()
    assert set(samples.columns) == {"return", "imbalance", "available_at", "target_end", "target"}


@pytest.mark.parametrize("horizon", [1, 5, 15])
def test_forecast_targets_are_purged_at_year_boundaries(horizon):
    """Keep targets within their stage and retain past context for the next stage."""
    minute = analysis.minute_features(validation_sample())
    samples = analysis.forecast_samples(minute, horizon)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    cutoff = pd.Timestamp("2025-01-01", tz="UTC")
    train, removed = analysis.period_samples(samples, start, cutoff)
    assert removed == horizon - 1
    assert train["target_end"].max() == cutoff
    assert (train["available_at"] < cutoff).all()
    valid, removed = analysis.period_samples(samples, cutoff, pd.Timestamp("2026-01-01", tz="UTC"))
    assert valid["available_at"].min() == cutoff
    assert not set(train.index).intersection(valid.index)
    assert (valid["target_end"] <= pd.Timestamp("2026-01-01", tz="UTC")).all()


def test_future_targets_never_cross_a_missing_minute():
    """Exclude forward targets that jump across a missing minute."""
    raw = validation_sample().iloc[:120].copy()
    raw = raw.drop(raw.index[60])
    samples = analysis.forecast_samples(analysis.minute_features(raw), 15)
    gap = pd.Timestamp("2024-12-31T23:00:00Z")
    crossing = (samples["available_at"] <= gap) & (samples["target_end"] > gap)
    assert not crossing.any()


def test_frozen_models_and_selection_ignore_test_data_changes():
    """Changing 2026 data must leave training parameters and the validation choice unchanged."""
    raw = validation_sample()
    config = {"start": pd.Timestamp("2023-01-01", tz="UTC"), "end": pd.Timestamp("2026-01-02", tz="UTC")}
    periods = analysis.validation_periods(config)
    audit, before, scores, days = analysis.forecast_tables(analysis.minute_features(raw), periods)
    changed = raw.copy()
    test = changed.index >= pd.Timestamp("2026-01-01", tz="UTC")
    changed.loc[test, "close"] *= np.linspace(1, 1.5, test.sum())
    changed.loc[test, "taker_buy_quote_volume"] = 99
    audit_after, after, new_scores, new_days = analysis.forecast_tables(analysis.minute_features(changed), periods)
    pd.testing.assert_frame_equal(before, after)
    old_validation = scores[scores["stage"] == "validation"].reset_index(drop=True)
    new_validation = new_scores[new_scores["stage"] == "validation"].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_validation, new_validation)
    assert audit["target_within_period"].all()
    assert before.groupby("horizon_minutes")["selected_on_validation"].sum().eq(1).all()
    for horizon, group in old_validation.groupby("horizon_minutes"):
        chosen = group[group["selected_on_validation"]].iloc[0]
        assert chosen["mse"] == group["mse"].min()
    assert (before["fit_last_target_end"] <= pd.Timestamp("2025-01-01", tz="UTC")).all()


def test_linear_forecast_uses_training_scaler_and_matches_known_relation():
    """Recover a simple linear relation and reuse its training scale on later values."""
    x = np.linspace(-1, 1, 100)
    train = pd.DataFrame({"imbalance": x, "target": 0.002 + 0.01 * x})
    model = analysis.fit_forecast(train, "imbalance")
    later = pd.DataFrame({"imbalance": [2.0, 3.0]})
    np.testing.assert_allclose(analysis.predict_forecast(later, model), [0.022, 0.032], atol=1e-14)
    assert model["center"] == pytest.approx(x.mean())
    assert model["scale"] == pytest.approx(x.std())


def test_no_validation_period_means_no_model_selection():
    """Do not select a forecast from training scores when validation data is absent."""
    raw = validation_sample().iloc[:90]
    config = {"start": pd.Timestamp("2023-01-01", tz="UTC"), "end": pd.Timestamp("2025-01-01", tz="UTC")}
    audit, models, scores, days = analysis.forecast_tables(analysis.minute_features(raw), analysis.validation_periods(config))
    assert not models.empty
    assert not models["selected_on_validation"].any()
    assert days.empty


def test_bootstrap_indices_preserve_adjacent_blocks():
    """Resample consecutive blocks reproducibly, including calendar positions with missing data."""
    indices = analysis.block_indices(35, 7, draws=20)
    assert indices.shape == (20, 35)
    for sample in indices:
        for start in range(0, 35, 7):
            assert (np.diff(sample[start:start + 7]) % 35 == 1).all()
    np.testing.assert_array_equal(indices, analysis.block_indices(35, 7, draws=20))


def test_daily_pair_sums_match_direct_correlation():
    """Recover direct Pearson correlation from daily pair sums without losing paired alignment."""
    rng = np.random.default_rng(31)
    x = rng.normal(size=1400)
    y = 0.4 * x + rng.normal(size=1400)
    x[12] = np.nan
    dates = pd.date_range("2024-01-01", periods=140, tz="UTC").repeat(10)
    selected = np.ones(len(x), dtype=bool)
    daily = analysis.daily_pair_sums(x, y, dates, selected)
    values = daily[analysis.MOMENTS].to_numpy()
    check = analysis.block_test(values, "correlation", 7, indices=analysis.block_indices(140, 7, draws=399))
    valid = np.isfinite(x) & np.isfinite(y)
    assert check["estimate"] == pytest.approx(np.corrcoef(x[valid], y[valid])[0, 1])
    assert check["status"] == "ok"
    assert 0 < check["p_value"] <= 1
    assert check["bootstrap_draws"] == 399


def test_block_intervals_widen_for_serially_correlated_data():
    """Show that blocks retain dependence missed by independent resampling."""
    rng = np.random.default_rng(11)
    values = np.zeros(500)
    for i in range(1, len(values)):
        values[i] = 0.95 * values[i - 1] + rng.normal()
    iid = analysis.block_test(values, "mean", 1, indices=analysis.block_indices(500, 1, draws=499))
    blocks = analysis.block_test(values, "mean", 14, indices=analysis.block_indices(500, 14, draws=499))
    assert blocks["ci95_high"] - blocks["ci95_low"] > 1.5 * (iid["ci95_high"] - iid["ci95_low"])
    small = analysis.block_test(values[:20], "mean", 7)
    assert small["status"] == "insufficient_blocks"
    assert pd.isna(small["p_value"])


def test_holm_adjustment_uses_all_planned_tests_and_keeps_families_separate():
    """Match a known Holm calculation and preserve undefined tests in the family size."""
    table = pd.DataFrame({"stage": ["train"] * 4 + ["validation"], "block_setting": ["primary"] * 5,
                          "test_id": ["a", "b", "c", "d", "a"], "p_value": [0.03, 0.01, 0.04, np.nan, 0.01],
                          "estimate": 0.2, "null": 0})
    result = analysis.holm_adjust(table)
    np.testing.assert_allclose(result.loc[[0, 1, 2, 4], "p_holm"], [0.09, 0.04, 0.09, 0.01])
    assert pd.isna(result.loc[3, "p_holm"])
    assert result["reject_null_5pct"].tolist() == [False, True, False, False, True]
    assert not result["supported_at_both_block_lengths"].any()


def test_weekly_bootstrap_keeps_missing_calendar_weeks(monkeypatch):
    """Keep missing weeks in place when drawing calendar blocks."""
    dates = pd.date_range("2024-01-01", periods=40, freq="7D", tz="UTC")
    weekly = pd.DataFrame({"week_start": dates, "volume_ratio": np.linspace(0.3, 0.7, 40), "rv_ratio": 0.5})
    weekly = weekly.drop(index=[7, 8])
    calls = []
    original = analysis.block_test
    def record(values, kind, size, null=0, indices=None):
        """Record the calendar series passed to the real bootstrap."""
        calls.append(values.copy())
        return original(values, kind, size, null, indices)
    monkeypatch.setattr(analysis, "block_test", record)
    result = analysis.weekly_summary(weekly)
    assert len(calls[0]) == 40
    assert np.isnan(calls[0][7:9]).all()
    assert (result["weeks"] == 38).all()
    assert (result["block_weeks"] == 4).all()
