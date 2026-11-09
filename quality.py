"""Check local Parquet files and record coverage, bad rows and gaps."""

from datetime import datetime, timezone
import logging

import numpy as np
import pandas as pd

from download import (
    SCHEMA_VERSION, __version__, PRICE_COLUMNS, VALUE_COLUMNS, VOLUME_COLUMNS,
    atomic_bytes, check_column_types, config_identity, day_windows, epoch_ms, iso,
    manifest_path, now_utc, partition_path, read_json, read_partition,
    parse_utc, sha256_file, validate_manifest, write_json,
)


# Gap labels and row checks


GAP_COLUMNS = ["partition_date", "start_utc", "end_exclusive_utc", "missing_seconds", "status", "reason"]
# Only this exact source gap gets a warning.
# The exchange halt was longer than the gap in the historical candles.
KNOWN_GAP = {
    "start_utc": "2023-03-24T12:39:42Z",
    "end_exclusive_utc": "2023-03-24T14:00:00Z",
    "reason": "Documented Binance spot trading halt on 2023-03-24; original records remain missing.",
}
RULES = [
    "duplicate_seconds", "out_of_window_rows", "off_grid_rows", "out_of_order_rows",
    "null_rows", "non_finite_rows", "non_positive_price_rows", "negative_volume_rows",
    "negative_trade_count_rows", "ohlc_violation_rows", "close_time_violation_rows",
    "taker_volume_violation_rows", "trade_volume_inconsistency_rows",
]


def make_gap(window, start_ms, end_ms):
    """Describe a missing interval as UTC bounds and a second count."""
    start = datetime.fromtimestamp(start_ms // 1000, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ms // 1000, tz=timezone.utc)
    return {
        "partition_date": window["date"],
        "start_utc": iso(start),
        "end_exclusive_utc": iso(end),
        "missing_seconds": (end_ms - start_ms) // 1000,
    }


def gaps_for_times(open_ms, window):
    """Find missing seconds at the start, middle and end of the window."""
    valid_times = []
    for value in open_ms:
        value = int(value)
        if window["start_ms"] <= value < window["end_ms"] and value % 1000 == 0:
            valid_times.append(value)
    # Use unique times to count coverage. Duplicate rows stay in the input frame.

    valid_times = sorted(set(valid_times))
    gaps = []
    cursor = window["start_ms"]
    for timestamp in valid_times:
        if timestamp > cursor:
            gaps.append(make_gap(window, cursor, timestamp))
        cursor = timestamp + 1000
    # Include the trailing gap, or the whole window if there are no rows.
    if cursor < window["end_ms"]:
        gaps.append(make_gap(window, cursor, window["end_ms"]))
    return gaps


def classify_gaps(gaps, window):
    """Split gaps at the known halt boundaries and label each piece."""
    known_start = epoch_ms(parse_utc(KNOWN_GAP["start_utc"]))
    known_end = epoch_ms(parse_utc(KNOWN_GAP["end_exclusive_utc"]))
    results = []
    for gap in gaps:
        start = epoch_ms(parse_utc(gap["start_utc"]))
        end = epoch_ms(parse_utc(gap["end_exclusive_utc"]))
        overlap_start = max(start, known_start)
        overlap_end = min(end, known_end)

        if overlap_start >= overlap_end:
            item = gap.copy()
            item["status"] = "FAIL"
            item["reason"] = "Unexpected missing seconds."
            results.append(item)
            continue

        # Even one missing second outside the known interval still fails.
        if start < overlap_start:
            item = make_gap(window, start, overlap_start)
            item["status"] = "FAIL"
            item["reason"] = "Unexpected missing seconds."
            results.append(item)

        item = make_gap(window, overlap_start, overlap_end)
        item["status"] = "WARNING"
        item["reason"] = KNOWN_GAP["reason"]
        results.append(item)

        if overlap_end < end:
            item = make_gap(window, overlap_end, end)
            item["status"] = "FAIL"
            item["reason"] = "Unexpected missing seconds."
            results.append(item)
    return results


def audit_frame(frame, window):
    """Return row checks and gap details for one window."""
    # Convert to nanoseconds before doing integer timestamp checks.

    opens_ns = frame["open_time"].dt.as_unit("ns").astype("int64")
    closes_ns = frame["close_time"].dt.as_unit("ns").astype("int64")
    opens_ms = opens_ns // 1000000

    # Compare OHLC values within each row.
    high_too_low = frame["high"] < frame[["open", "close", "low"]].max(axis=1)
    low_too_high = frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    close_before_open = closes_ns < opens_ns
    close_after_second = closes_ns >= opens_ns + 1000000000

    # Allow a small rounding error when comparing taker volume with total volume.
    base_tolerance = np.maximum(1e-8, frame["volume"].abs() * 1e-12)
    quote_tolerance = np.maximum(1e-8, frame["quote_volume"].abs() * 1e-12)
    too_much_taker_base = frame["taker_buy_base_volume"] > frame["volume"] + base_tolerance
    too_much_taker_quote = frame["taker_buy_quote_volume"] > frame["quote_volume"] + quote_tolerance
    no_trades = frame["trade_count"] == 0
    no_base_volume = frame["volume"] == 0
    no_quote_volume = frame["quote_volume"] == 0
    # Zero trades and zero volume should agree.

    inconsistent_volume = (no_trades != no_base_volume) | (no_base_volume != no_quote_volume)
    outside_window = (opens_ms < window["start_ms"]) | (opens_ms >= window["end_ms"])
    # Catch NaN and infinity across all value columns.
    finite_values = np.isfinite(frame[VALUE_COLUMNS]).all(axis=1)

    # Each count records how many rows break that rule.

    counts = {}
    counts["duplicate_seconds"] = int(frame["open_time"].duplicated().sum())
    counts["out_of_window_rows"] = int(outside_window.sum())

    counts["off_grid_rows"] = int((opens_ns % 1000000000 != 0).sum())

    counts["out_of_order_rows"] = int((opens_ns.diff() < 0).sum())
    counts["null_rows"] = int(frame.isna().any(axis=1).sum())
    counts["non_finite_rows"] = int((~finite_values).sum())
    counts["non_positive_price_rows"] = int((frame[PRICE_COLUMNS] <= 0).any(axis=1).sum())
    counts["negative_volume_rows"] = int((frame[VOLUME_COLUMNS] < 0).any(axis=1).sum())
    counts["negative_trade_count_rows"] = int((frame["trade_count"] < 0).sum())
    counts["ohlc_violation_rows"] = int((high_too_low | low_too_high).sum())
    counts["close_time_violation_rows"] = int((close_before_open | close_after_second).sum())
    counts["taker_volume_violation_rows"] = int((too_much_taker_base | too_much_taker_quote).sum())
    counts["trade_volume_inconsistency_rows"] = int(inconsistent_volume.sum())

    gaps = classify_gaps(gaps_for_times(opens_ms, window), window)
    missing = 0
    known_missing = 0
    unexpected_missing = 0
    for gap in gaps:
        missing += gap["missing_seconds"]
        if gap["status"] == "WARNING":
            known_missing += gap["missing_seconds"]
        else:
            unexpected_missing += gap["missing_seconds"]
    status = "FAIL"
    if unexpected_missing == 0 and sum(counts.values()) == 0:
        status = "PASS"
        if known_missing > 0:
            status = "WARNING"
    result = {
        "date": window["date"],
        "start_utc": iso(window["start"]),
        "end_exclusive_utc": iso(window["end"]),
        "expected_seconds": window["seconds"],
        "rows": len(frame),
        "observed_unique_seconds": window["seconds"] - missing,
        "missing_seconds": missing,
        "known_missing_seconds": known_missing,
        "unexpected_missing_seconds": unexpected_missing,
        "coverage_fraction": (window["seconds"] - missing) / window["seconds"],
        "zero_volume_seconds": int(no_base_volume.sum()),
        "status": status,
    }

    result.update(counts)
    return result, gaps


def audit(config):
    """Check daily files offline and return the summary, daily table and gaps."""
    partitions = []
    gaps = []
    issues = []
    counts = {}
    for rule in RULES:
        counts[rule] = 0
    checked_windows = 0

    for window in day_windows(config):
        path = partition_path(config, window)
        receipt = manifest_path(config, window)
        try:
            if not path.exists() or not receipt.exists():
                raise ValueError("Missing Parquet partition or receipt.")
            manifest = read_json(receipt)
            validate_manifest(config, window, manifest)
            if sha256_file(path) != manifest.get("parquet_sha256"):
                raise ValueError("Parquet SHA-256 mismatch.")

            # Check the raw-page details before counting this file as trusted.
            if not isinstance(manifest.get("raw_pages"), list):
                raise ValueError("Receipt must list its raw API pages.")
            for page in manifest["raw_pages"]:
                if not isinstance(page, dict) or not isinstance(page.get("retrieved_at"), str):
                    raise ValueError("Invalid raw-page details in receipt.")

            frame = read_partition(path)
            check_column_types(frame)
            if len(frame) != manifest.get("row_count"):
                raise ValueError("Receipt row count differs from Parquet.")
            result, partition_gaps = audit_frame(frame, window)
            gaps.extend(partition_gaps)
            for rule in RULES:
                counts[rule] += result[rule]
            partitions.append(result)
        except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
            # An untrusted file leaves its entire window uncovered.
            logging.error("Could not trust %s: %s", window["date"], error)
            issues.append({"date": window["date"], "error": str(error)})
            # The halt exception only applies to gaps in a trusted file.
            gap = make_gap(window, window["start_ms"], window["end_ms"])
            gap["status"] = "FAIL"
            gap["reason"] = "Missing or untrusted partition."
            gaps.append(gap)
            partitions.append({
                "date": window["date"],
                "start_utc": iso(window["start"]),
                "end_exclusive_utc": iso(window["end"]),
                "expected_seconds": window["seconds"],
                "rows": 0,
                "observed_unique_seconds": 0,
                "missing_seconds": window["seconds"],
                "known_missing_seconds": 0,
                "unexpected_missing_seconds": window["seconds"],
                "coverage_fraction": 0,
                "status": "UNTRUSTED",
            })
        checked_windows += 1
        if checked_windows % 100 == 0:
            logging.info("Checked %s daily windows", checked_windows)

    expected = 0
    missing = 0
    known_missing = 0
    unexpected_missing = 0
    observed = 0
    zero_volume = 0
    for partition in partitions:
        expected += partition["expected_seconds"]
        missing += partition["missing_seconds"]
        known_missing += partition["known_missing_seconds"]
        unexpected_missing += partition["unexpected_missing_seconds"]
        observed += partition["rows"]
        zero_volume += partition.get("zero_volume_seconds", 0)
    # Allow the known gap while keeping actual coverage and missing counts.
    passed = len(issues) == 0 and unexpected_missing == 0 and sum(counts.values()) == 0
    status = "FAIL"
    if passed:
        status = "PASS"
        if known_missing > 0:
            status = "WARNING"
    warnings = []
    for gap in gaps:
        if gap["status"] == "WARNING":
            warnings.append(gap.copy())
    summary = {
        "generated_at": now_utc(),
        "software_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "request": config_identity(config),
        "quality_status": status,
        "ready_for_analysis": passed,
        "expected_partitions": len(partitions),
        "trusted_partitions": len(partitions) - len(issues),
        "expected_seconds": expected,
        "observed_rows": observed,
        "observed_unique_seconds": expected - missing,
        "missing_seconds": missing,
        "known_missing_seconds": known_missing,
        "unexpected_missing_seconds": unexpected_missing,
        "coverage_fraction": (expected - missing) / expected,
        "zero_volume_seconds": zero_volume,
        "violations": counts,
        "artifact_issues": issues,
        "warnings": warnings,
    }
    return summary, pd.DataFrame(partitions), pd.DataFrame(gaps, columns=GAP_COLUMNS)


def check_data(config):
    """Run the audit and save quality.json, daily_quality.csv and gaps.csv."""
    summary, daily, gaps = audit(config)
    output = config["report_dir"]
    # Save the detail tables first and the summary last.
    atomic_bytes(output / "daily_quality.csv", daily.to_csv(index=False).encode("utf-8"))
    atomic_bytes(output / "gaps.csv", gaps.to_csv(index=False).encode("utf-8"))
    write_json(output / "quality.json", summary)
    return summary
