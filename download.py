"""Download BTCUSDT seconds and keep daily files plus raw page caches."""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
import logging
import os
import time

import pandas as pd
import requests

# Keep the schema version separate from the code version.
__version__ = "1.5.0"
SCHEMA_VERSION = "1"


# Dates and windows

def parse_utc(value):
    """Parse a date or timestamp as a whole-second UTC datetime."""
    if not isinstance(value, str):
        raise ValueError("Dates must be strings.")
    # A plain date starts at midnight UTC.
    if len(value) == 10:
        value = value + "T00:00:00Z"
    result = pd.Timestamp(value)
    # Require a timezone so local machine settings cannot shift the range.
    if result.tzinfo is None:
        raise ValueError("Datetime strings need a timezone, for example 2023-01-01T00:00:00Z.")
    if result.microsecond != 0 or result.nanosecond != 0:
        raise ValueError("Window boundaries must fall on whole seconds.")
    return result.tz_convert("UTC").to_pydatetime()


def iso(value):
    """Format a timestamp in UTC with a trailing Z."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def epoch_ms(value):
    """Convert a whole-second datetime to Unix milliseconds."""
    return int(value.timestamp()) * 1000


def make_window(start, end):
    """Build a window for [start, end), including its file key and expected row count."""
    window = {}
    window["start"] = start
    window["end"] = end
    window["start_ms"] = epoch_ms(start)
    window["end_ms"] = epoch_ms(end)
    # Include both bounds so partial days get different file names.
    window["key"] = str(window["start_ms"]) + "-" + str(window["end_ms"])
    window["date"] = start.date().isoformat()
    window["seconds"] = int((end - start).total_seconds())
    return window


def day_windows(config):
    """Split the requested range at each UTC midnight."""
    windows = []
    start = config["start"]
    while start < config["end"]:
        midnight = start.replace(hour=0, minute=0, second=0)
        # Stop at the next midnight or the requested end, whichever comes first.
        end = min(midnight + timedelta(days=1), config["end"])
        windows.append(make_window(start, end))

        start = end
    return windows


def config_identity(config):
    """Keep the settings that identify the data source and requested range."""
    return {
        "market": "binance_spot",
        "symbol": config["symbol"],
        "interval": config["interval"],
        "start": iso(config["start"]),
        "end_exclusive": iso(config["end"]),
        "base_url": config["base_url"],
    }


def make_plan(config):
    """Estimate rows, API calls and the minimum time spent on request spacing."""
    seconds = 0
    requests = 0
    windows = day_windows(config)
    for window in windows:
        seconds += window["seconds"]
        # Round up: 2,001 seconds need three pages.
        requests += (window["seconds"] + 999) // 1000
    plan = config_identity(config)
    plan["expected_seconds"] = seconds
    plan["partitions"] = len(windows)
    plan["minimum_kline_requests_if_complete"] = requests
    # Network and file work will add to this lower bound.
    minutes = requests * config["min_request_interval_seconds"] / 60
    plan["pacing_only_lower_bound_minutes"] = round(minutes, 2)
    plan["note"] = "Network latency, retries and local processing add to this lower bound."
    return plan


# Files and cache

def now_utc():
    """Return the current UTC time for logs and receipts."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_bytes(value):
    """Use the same JSON encoding each time so payload hashes stay consistent."""
    # Stable key order and spacing make the payload hash repeatable.
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return text.encode("utf-8")


def atomic_replace(source, target):
    """Replace the target file, retrying brief Windows permission errors."""
    # Windows may briefly hold the file open. Give it a few chances to clear.
    for attempt in range(7):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.25)


def atomic_bytes(path, data):
    """Write to a temporary file, flush it, then replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as file:
            file.write(data)
            # Flush Python buffers, then ask the OS to flush the file.
            file.flush()
            os.fsync(file.fileno())
        atomic_replace(temporary, path)
    finally:
        # Clean up any temporary file left after a failed write.
        if temporary.exists():
            temporary.unlink()


def write_json(path, value):
    """Save a readable JSON file using the temporary-file helper."""
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    atomic_bytes(path, (text + "\n").encode("utf-8"))


def read_json(path):
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_gzip_json(path, value):
    """Compress a JSON payload and save it to the cache."""
    data = gzip.compress(json_bytes(value), mtime=0)
    atomic_bytes(path, data)


def read_gzip_json(path):
    """Read a compressed JSON cache file."""
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path):
    """Hash the file in chunks to keep memory use small."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        # Read 1 MiB at a time.
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def create_lock(data_dir):
    """Create a lock so only one process can use this data folder at a time."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / ".writer.lock"
    try:
        # Mode x fails if another process already owns the lock.
        with path.open("x", encoding="utf-8") as file:
            json.dump({"pid": os.getpid(), "started_at": now_utc()}, file)
    except FileExistsError:
        raise RuntimeError("A writer lock exists at " + str(path) + ". See README for recovery.")
    return path


def remove_lock(path):
    """Release the lock if this run created one."""
    if path is not None and path.exists():
        path.unlink()


# API requests

def open_client():
    """Start a reusable HTTP session and its rate-limit counters."""
    # Keep the connection and rate-limit state together.
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "btc-market-data/1.1"})
    return {
        "session": session,
        "last_request": None,
        "minute": None,
        "local_weight": 0,
        "server_weight": 0,
    }


def wait_before_request(config, client, weight):
    """Wait for the request spacing and minute weight budget."""
    # Use a monotonic clock for elapsed time between requests.
    if client["last_request"] is not None:
        elapsed = time.monotonic() - client["last_request"]
        wait = config["min_request_interval_seconds"] - elapsed
        if wait > 0:
            time.sleep(wait)

    # Reset the local counter when the minute changes.
    # The server header also accounts for other requests from this IP.
    minute = int(time.time() // 60)
    if minute != client["minute"]:
        client["minute"] = minute
        client["local_weight"] = 0
        client["server_weight"] = 0
    # Use the larger count to stay within the budget.
    used_weight = max(client["local_weight"], client["server_weight"])
    if used_weight + weight > config["request_weight_budget_per_minute"]:
        # Wait for the next minute, with a small buffer.
        wait = 60 - time.time() % 60 + 0.25
        logging.info("Waiting %.2f seconds for the local weight budget", wait)
        time.sleep(wait)
        client["minute"] = int(time.time() // 60)
        client["local_weight"] = 0
        client["server_weight"] = 0
    client["last_request"] = time.monotonic()
    client["local_weight"] += weight


def retry_after_seconds(value):
    """Read Retry-After as either seconds or an HTTP date."""
    if value is None:
        return 0
    try:
        return max(0, float(value))
    except ValueError:
        # Retry-After can also be a date.
        try:
            retry_time = parsedate_to_datetime(value).timestamp()
            return max(0, retry_time - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0


def get_json(config, client, endpoint, params=None, weight=2):
    """Send a GET request and retry temporary failures up to the configured limit."""
    url = config["base_url"] + endpoint

    for attempt in range(config["max_attempts"]):
        wait_before_request(config, client, weight)
        try:
            response = client["session"].get(url, params=params, timeout=config["timeout_seconds"])
        except (requests.ConnectionError, requests.Timeout):
            if attempt + 1 == config["max_attempts"]:
                raise RuntimeError("Network retries exhausted for " + endpoint)
            # Wait 2, 4, 6, ... seconds between retries.
            wait = 2 * (attempt + 1)
            logging.warning("Network error; retrying in %s seconds", wait)
            time.sleep(wait)
            continue

        # This header covers the total weight used by this IP.
        used_weight = response.headers.get("X-MBX-USED-WEIGHT-1M")
        if used_weight is not None:
            try:
                client["server_weight"] = max(client["server_weight"], int(used_weight))
            except ValueError:
                logging.warning("Invalid weight header: %r", used_weight)
        if response.status_code == 418:
            raise RuntimeError("Binance reports an IP ban (418). Stop and check Retry-After.")
        # Retry rate limits and temporary server errors.
        if response.status_code == 429 or response.status_code >= 500:
            if attempt + 1 == config["max_attempts"]:
                raise RuntimeError("HTTP " + str(response.status_code) + ": retries exhausted.")
            wait = max(2 * (attempt + 1), retry_after_seconds(response.headers.get("Retry-After")))
            logging.warning("HTTP %s; retrying in %.2f seconds", response.status_code, wait)
            time.sleep(wait)
            continue
        if response.status_code != 200:
            raise RuntimeError("HTTP " + str(response.status_code) + ": " + response.text[:300])

        # Check for an API error even if HTTP returned 200.
        data = response.json()
        if isinstance(data, dict) and "code" in data:
            raise RuntimeError("Binance error payload: " + str(data))
        metadata = {
            "request_url": response.url,
            "server_date": response.headers.get("Date"),
            "used_weight_1m": used_weight,
            "http_status": response.status_code,
        }
        return data, metadata


def get_server_time(config, client):
    """Get the exchange server time in milliseconds."""

    data, metadata = get_json(config, client, "/api/v3/time", weight=1)
    if not isinstance(data, dict) or type(data.get("serverTime")) is not int:
        raise ValueError("Invalid serverTime response.")
    return data["serverTime"]


def get_klines(config, client, start_ms, end_ms):
    """Fetch up to 1,000 BTCUSDT candles from [start_ms, end_ms)."""
    params = {
        "symbol": config["symbol"],
        "interval": "1s",
        "startTime": start_ms,
        "endTime": end_ms - 1,  # Binance includes endTime, so subtract 1 ms for our exclusive end.
        "limit": 1000,
    }
    data, metadata = get_json(config, client, "/api/v3/klines", params)
    if not isinstance(data, list):
        raise ValueError("Expected a list of klines.")
    metadata["params"] = params
    return data, metadata


# Raw rows and column types

# The API returns these 12 fields in this order.
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]
# Drop the unused ignore field from the saved table.
OUTPUT_COLUMNS = COLUMNS[:-1]
PRICE_COLUMNS = ["open", "high", "low", "close"]
VOLUME_COLUMNS = ["volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
VALUE_COLUMNS = PRICE_COLUMNS + VOLUME_COLUMNS


def page_open_times(rows, start_ms, end_ms):
    """Check the raw page and return its candle start times for pagination."""
    if len(rows) > 1000:
        raise ValueError("API returned more than 1000 rows.")
    times = []
    previous = None
    for row in rows:
        if not isinstance(row, list) or len(row) != 12:
            raise ValueError("Each kline must contain exactly 12 fields.")
        # Start time, close time and trade count must be integers.
        # Check the exact type here because bool is also a subclass of int.
        for position in [0, 6, 8]:
            if type(row[position]) is not int:
                raise ValueError("Timestamps and trade counts must be JSON integers.")
        timestamp = row[0]
        if timestamp < start_ms or timestamp >= end_ms:
            raise ValueError("A row is outside the requested window; check timestamp units.")
        if timestamp % 1000 != 0:
            raise ValueError("A 1s candle must start on a whole second.")
        # Keep duplicates for the audit, but stop if time goes backwards.
        if previous is not None and timestamp < previous:
            raise ValueError("API rows must be in chronological order.")
        times.append(timestamp)
        previous = timestamp
    return times


def rows_to_frame(rows):
    """Convert API rows to the saved column types and UTC timestamps."""
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.drop(columns=["ignore"])
    # Parse the API timestamps as milliseconds, then store them as UTC microseconds.
    for name in ["open_time", "close_time"]:
        frame[name] = pd.to_datetime(frame[name], unit="ms", utc=True)
        frame[name] = frame[name].dt.as_unit("us")
    # Bad numeric text should raise an error.
    for name in VALUE_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        frame[name] = frame[name].astype("float64")
    frame["trade_count"] = frame["trade_count"].astype("int64")
    return frame


def check_column_types(frame):
    """Check the saved column order, timestamp types and numeric types."""
    # Check column order too, since several fields have the same type.
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError("Parquet columns differ from the expected columns.")
    for name in ["open_time", "close_time"]:
        if not isinstance(frame[name].dtype, pd.DatetimeTZDtype):
            raise ValueError(name + " must contain timezone-aware timestamps.")
        if str(frame[name].dt.tz) != "UTC":
            raise ValueError(name + " must use UTC.")
        if frame[name].dt.unit != "us":
            raise ValueError(name + " must use microsecond storage resolution.")
    for name in VALUE_COLUMNS:
        if str(frame[name].dtype) != "float64":
            raise ValueError(name + " must use float64.")
    if str(frame["trade_count"].dtype) != "int64":
        raise ValueError("trade_count must use int64.")


# Daily downloads

def partition_path(config, window):
    """Get the Parquet path for this daily window."""
    folder = config["data_dir"] / "processed" / config["symbol"] / "1s"
    folder = folder / ("date=" + window["date"])
    return folder / (window["key"] + ".parquet")


def manifest_path(config, window):
    """Get the receipt path for this daily window."""
    return config["data_dir"] / "manifests" / config["symbol"] / (window["key"] + ".json")


def identity(config, window):
    """Build the source, window and schema details stored in each receipt."""
    return {
        "market": "binance_spot",
        "base_url": config["base_url"],
        "symbol": config["symbol"],
        "interval": "1s",
        "start": iso(window["start"]),
        "end_exclusive": iso(window["end"]),
        "schema_version": SCHEMA_VERSION,
    }


def validate_manifest(config, window, manifest):
    """Check that a receipt belongs to the requested window and file."""
    if manifest.get("identity") != identity(config, window):
        raise ValueError("Receipt does not match the source, window or schema: " + window["key"])
    if manifest.get("status") != "materialized":
        raise ValueError("Partition was not fully materialized: " + window["key"])
    # Relative paths let the data folder move without breaking the receipt.
    expected_path = partition_path(config, window).relative_to(config["data_dir"]).as_posix()
    if manifest.get("parquet_path") != expected_path:
        raise ValueError("Unexpected partition path in receipt: " + window["key"])


def write_parquet(path, frame):
    """Save the daily frame through a temporary Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:

        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        # Flush the finished Parquet file before replacing the target.
        with temporary.open("r+b") as file:
            os.fsync(file.fileno())
        atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_partition(path):
    """Read one daily Parquet file."""
    # Open the file directly so date=... stays a folder name, with no extra column.
    with path.open("rb") as file:
        return pd.read_parquet(file, engine="pyarrow")


def load_page(config, client, window, cursor):
    """Read a verified cache page, or fetch and save a new page."""
    params = {
        "symbol": config["symbol"],
        "interval": "1s",
        "startTime": cursor,
        "endTime": window["end_ms"] - 1,
        "limit": 1000,
    }
    folder = config["data_dir"] / "raw" / "rest" / config["symbol"] / "1s" / window["date"]
    cache = folder / (str(cursor) + "-" + str(window["end_ms"]) + ".json.gz")
    # A cached page must still match the current request.
    if cache.exists():
        page = read_gzip_json(cache)
        if page.get("params") != params or page.get("base_url") != config["base_url"]:
            raise ValueError("Cached request metadata mismatch: " + str(cache))
        if page.get("time_unit") != "ms":
            raise ValueError("Cached timestamp unit is not milliseconds: " + str(cache))
        # Recompute the hash to catch changes to the cached payload.
        payload_hash = hashlib.sha256(json_bytes(page["data"])).hexdigest()
        if payload_hash != page.get("payload_sha256"):
            raise ValueError("Cached payload hash mismatch: " + str(cache))
    else:
        data, metadata = get_klines(config, client, cursor, window["end_ms"])
        page = {
            "base_url": config["base_url"],
            "params": params,
            "time_unit": "ms",
            "retrieved_at": now_utc(),
            "response_metadata": metadata,
            "payload_sha256": hashlib.sha256(json_bytes(data)).hexdigest(),
            "data": data,
        }
        # Save the page before advancing the cursor so an interrupted run can replay it.
        write_gzip_json(cache, page)
    return page, cache


def materialize_window(config, window, client):
    """Finish one daily window, reusing verified files and cached pages where possible."""
    output = partition_path(config, window)
    receipt = manifest_path(config, window)
    # Reuse a completed day only after checking both its receipt and file hash.
    if receipt.exists() and output.exists():
        manifest = read_json(receipt)
        validate_manifest(config, window, manifest)
        if sha256_file(output) != manifest.get("parquet_sha256"):
            raise ValueError("Parquet hash mismatch: " + str(output))
        logging.info("Already saved %s (%s rows)", window["date"], manifest["row_count"])
        return {"date": window["date"], "rows": manifest["row_count"], "reused": True}

    cursor = window["start_ms"]
    # Keep at most one day of raw rows in memory.
    rows = []
    raw_pages = []
    checkpoint = config["data_dir"] / "checkpoints" / config["symbol"] / (window["key"] + ".json")
    while cursor < window["end_ms"]:
        page, cache = load_page(config, client, window, cursor)
        data = page["data"]
        times = page_open_times(data, cursor, window["end_ms"])

        rows.extend(data)
        raw_pages.append({
            "path": cache.relative_to(config["data_dir"]).as_posix(),
            "payload_sha256": page["payload_sha256"],
            "row_count": len(data),
            "retrieved_at": page["retrieved_at"],
        })

        # A short page can still have more data after it.
        if len(times) == 0:
            cursor = window["end_ms"]
        else:

            cursor = times[-1] + 1000
        # Checkpoints show progress; cached pages are what we use to resume.
        write_json(checkpoint, {
            "status": "in_progress",
            "identity": identity(config, window),
            "next_start_ms": cursor,
            "rows_cached": len(rows),
            "updated_at": now_utc(),
        })
        if len(raw_pages) % 10 == 0:
            logging.info("%s: cached %s / %s expected seconds", window["date"], len(rows), window["seconds"])

    # Save the observed rows and let the quality check report any gaps.
    frame = rows_to_frame(rows)
    write_parquet(output, frame)
    write_json(receipt, {
        "status": "materialized",
        "identity": identity(config, window),
        "software_version": __version__,
        "materialized_at": now_utc(),
        "row_count": len(frame),
        "expected_seconds": window["seconds"],
        "parquet_path": output.relative_to(config["data_dir"]).as_posix(),
        "parquet_sha256": sha256_file(output),
        "raw_pages": raw_pages,
        "note": "Materialized does not mean quality-approved. Run check to validate.",
    })
    write_json(checkpoint, {
        "status": "materialized",
        "identity": identity(config, window),
        "next_start_ms": window["end_ms"],
        "rows_cached": len(rows),
        "updated_at": now_utc(),
    })
    logging.info("Saved %s: %s rows", window["date"], len(frame))
    return {"date": window["date"], "rows": len(frame), "reused": False}


def fetch(config):
    """Download each daily window after checking the exchange server time."""
    client = open_client()
    try:
        server_time = get_server_time(config, client)
        # Only request candles that have finished on the exchange clock.
        last_closed_boundary = server_time // 1000 * 1000
        if epoch_ms(config["end"]) > last_closed_boundary:
            raise ValueError("The requested end includes an unfinished or future candle.")
        results = []
        for window in day_windows(config):
            logging.info("Fetching [%s, %s)", iso(window["start"]), iso(window["end"]))
            result = materialize_window(config, window, client)
            results.append(result)
        return results
    finally:
        # Close the session even if the download fails or gets interrupted.
        client["session"].close()
