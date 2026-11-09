"""Load the config and run the download or quality check."""

import argparse
from datetime import timedelta
import json
import logging
from pathlib import Path

from download import create_lock, fetch, make_plan, parse_utc, remove_lock
from quality import check_data


# Config

def validate_config(config):
    """Check the symbol, dates, API source and local paths."""
    if config["symbol"] != "BTCUSDT" or config["interval"] != "1s":
        raise ValueError("This project is restricted to BTCUSDT spot 1s klines.")
    # Both bounds must be whole seconds in UTC.
    for name in ["start", "end"]:
        value = config[name]
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("Use timezone-aware UTC boundaries.")
        if value.microsecond != 0:
            raise ValueError("Use whole-second boundaries.")
    if config["start"] >= config["end"]:
        raise ValueError("start must precede end; end is exclusive.")
    if config["start"] < parse_utc("2023-01-01"):
        raise ValueError("The project range starts on 2023-01-01 UTC.")

    allowed_urls = ["https://data-api.binance.vision", "https://api.binance.com"]
    if config["base_url"] not in allowed_urls:
        raise ValueError("Use an approved Binance HTTPS spot endpoint.")
    if not 0.1 <= config["min_request_interval_seconds"] <= 60:
        raise ValueError("The request interval must be between 0.1 and 60 seconds.")
    if not 2 <= config["request_weight_budget_per_minute"] <= 1000:
        raise ValueError("The local weight budget must be between 2 and 1000.")
    if config["timeout_seconds"] <= 0 or not 1 <= config["max_attempts"] <= 10:
        raise ValueError("Use a positive timeout and 1-10 attempts.")
    if config["data_dir"].resolve() == config["report_dir"].resolve():
        raise ValueError("data_dir and report_dir must differ.")


def load_config(path, start=None, end=None, data_dir=None, report_dir=None):
    """Read the JSON config, apply any CLI overrides and check the result."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a JSON object.")
    # Config paths are relative to the project root when the file lives in configs/.
    root = path.parent
    if root.name == "configs":
        root = root.parent

    # Fill in settings that were left out of the JSON file.
    config.setdefault("symbol", "BTCUSDT")
    config.setdefault("interval", "1s")
    config.setdefault("base_url", "https://data-api.binance.vision")
    config.setdefault("min_request_interval_seconds", 0.25)
    config.setdefault("request_weight_budget_per_minute", 1000)
    config.setdefault("timeout_seconds", 30)
    config.setdefault("max_attempts", 5)
    config.setdefault("data_dir", "data/full")
    config.setdefault("report_dir", "reports/full")
    # CLI arguments take priority over the config file.
    if start is not None:
        config["start"] = start
    if end is not None:
        config["end"] = end
    if data_dir is not None:
        config["data_dir"] = data_dir
    if report_dir is not None:
        config["report_dir"] = report_dir
    # Convert dates and paths once, before the rest of the code uses them.
    config["start"] = parse_utc(config["start"])
    config["end"] = parse_utc(config["end"])
    config["data_dir"] = root / config["data_dir"]
    config["report_dir"] = root / config["report_dir"]
    validate_config(config)
    return config


# CLI

def main(argv=None):
    """Run the chosen command and return its exit code."""

    parser = argparse.ArgumentParser(description="BTCUSDT spot 1s data preparation")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.json"))
    parser.add_argument("--start", help="Inclusive UTC start")
    parser.add_argument("--end", help="Exclusive UTC end")
    parser.add_argument("--data-dir", help="Data folder, relative to the project root")
    parser.add_argument("--report-dir", help="Quality-check output folder, relative to the project root")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("command", choices=["plan", "fetch", "check", "run"])
    args = parser.parse_args(argv)


    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


    lock = None
    try:
        config = load_config(args.config, args.start, args.end, args.data_dir, args.report_dir)
        # A plan only reads the config, so it can return before taking the lock.
        if args.command == "plan":
            print(json.dumps(make_plan(config), indent=2))
            return 0

        # Keep concurrent runs from writing to the same data folder.
        lock = create_lock(config["data_dir"])
        if args.command == "fetch" or args.command == "run":
            fetch(config)
        if args.command == "check" or args.command == "run":

            summary = check_data(config)
            result = {
                "quality_status": summary["quality_status"],
                "ready_for_analysis": summary["ready_for_analysis"],
                "requested_window": summary["request"],
                "expected_seconds": summary["expected_seconds"],
                "observed_rows": summary["observed_rows"],
                "missing_seconds": summary["missing_seconds"],
                "known_missing_seconds": summary["known_missing_seconds"],
                "unexpected_missing_seconds": summary["unexpected_missing_seconds"],
                "warnings": summary["warnings"],
                "quality_file": str(config["report_dir"] / "quality.json"),
            }
            print(json.dumps(result, indent=2))
            if summary["known_missing_seconds"] > 0:
                logging.warning("%s seconds are missing within the documented Binance halt; original data is not filled.", summary["known_missing_seconds"])
            # Use the quality result to decide the exit code.
            if not summary["ready_for_analysis"]:
                return 2
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted. Run the same command again to resume from cached pages.")
        return 130
    except Exception as error:
        logging.error("%s: %s", type(error).__name__, error, exc_info=args.verbose)
        return 1
    finally:
        # Release the lock on success, errors or Ctrl+C.
        remove_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())
