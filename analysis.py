"""Option B: check local data, calculate trading patterns and save tables and charts."""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images without opening a window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download import (
    create_lock, day_windows, iso, now_utc, partition_path, read_partition,
    remove_lock, write_json,
)
from quality import audit
from run import load_config


SESSIONS = [("Hong Kong", "Asia/Hong_Kong"),
            ("London", "Europe/London"),
            ("New York", "America/New_York")]
MOMENTS = ["pairs", "sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy"]
BOOTSTRAP_DRAWS = 1999


# Returns and correlations

def second_returns(frame, previous_time=None, previous_close=None):
    """Calculate log returns between consecutive seconds, carrying the last close across midnight."""
    result = np.log(frame["close"] / frame["close"].shift(1))
    continuous = frame["open_time"].diff() == pd.Timedelta(seconds=1)
    result = result.where(continuous)
    if len(frame) > 0 and previous_time is not None:
        if frame["open_time"].iloc[0] - previous_time == pd.Timedelta(seconds=1):
            result.iloc[0] = np.log(frame["close"].iloc[0] / previous_close)
    return result


def add_correlation(totals, key, x, y):
    """Add the count, sums, squares and cross-products needed for Pearson correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    values = [len(x), x.sum(), y.sum(), np.dot(x, x), np.dot(y, y), np.dot(x, y)]
    if key not in totals:
        totals[key] = np.zeros(6)
    totals[key] += np.array(values)
    return values


def correlation_table(totals):
    """Turn the accumulated sums into correlations and valid pair counts."""
    rows = []
    for key, values in totals.items():
        n, sx, sy, sxx, syy, sxy = values
        corr = np.nan
        if n >= 2:
            vx = sxx - sx * sx / n
            vy = syy - sy * sy / n
            if vx > 0 and vy > 0:
                corr = (sxy - sx * sy / n) / np.sqrt(vx * vy)
                corr = float(np.clip(corr, -1, 1))
        rows.append({"year": key[0], "frequency": key[1], "metric": key[2],
                     "lag": key[3], "pairs": int(n), "correlation": corr})
    return pd.DataFrame(rows)


# Daily aggregation

def aggregate_day(frame, window, returns):
    """Return minute and hourly tables plus a daily summary from the observed seconds."""
    work = frame.copy()
    work["minute"] = work["open_time"].dt.floor("min")
    work["n"] = 1
    work["rv2"] = returns * returns
    work["zero_seconds"] = (frame["volume"] == 0).astype(int)
    work["valid_returns"] = returns.notna().astype(int)
    rules = {"open": "first", "high": "max", "low": "min", "close": "last",
             "volume": "sum", "quote_volume": "sum", "trade_count": "sum",
             "taker_buy_quote_volume": "sum", "n": "sum", "rv2": "sum",
             "zero_seconds": "sum", "valid_returns": "sum"}
    minute = work.groupby("minute").agg(rules)
    # rv2 sums the valid squared second returns.
    hourly = minute.groupby(minute.index.hour)[["volume", "rv2", "n"]].sum()
    hourly.index.name = "hour"
    hourly = hourly.reset_index()
    hourly["date"] = pd.Timestamp(window["date"], tz="UTC")

    daily = {
        "date": pd.Timestamp(window["date"], tz="UTC"),
        "expected_seconds": window["seconds"], "rows": len(frame),
        "complete_day": window["seconds"] == 86400 and len(frame) == 86400,
        "open": frame["open"].iloc[0], "close": frame["close"].iloc[-1],
        "high": frame["high"].max(), "low": frame["low"].min(),
        "volume_btc": frame["volume"].sum(),
        "turnover_usdt": frame["quote_volume"].sum(),
        "trades": frame["trade_count"].sum(),
        "zero_volume_seconds": int((frame["volume"] == 0).sum()),
        "valid_second_returns": int(returns.notna().sum()),
        "rv_pct": np.sqrt((returns * returns).sum()) * 100,
    }
    return minute, hourly, daily


def minute_features(minute):
    """Mark continuous blocks of full minutes and add returns and trade imbalance."""
    result = minute.copy()
    full = result["n"] == 60
    adjacent = result.index.to_series().diff() == pd.Timedelta(minutes=1)
    connected = full & full.shift(1, fill_value=False) & adjacent
    # Start a new block whenever full minutes stop lining up.
    result["block"] = (~connected).cumsum()
    result["return"] = np.log(result["close"] / result["close"].shift(1)).where(connected)
    result["abs_return"] = result["return"].abs()
    # +1 means all taker buys; -1 means all taker sells.
    denominator = result["quote_volume"].where(result["quote_volume"] > 0)
    result["imbalance"] = (2 * result["taker_buy_quote_volume"] / denominator - 1).where(full)
    return result


def future_returns(minute, horizon):
    """Calculate forward close-to-close returns within the same continuous block."""
    same_block = minute["block"] == minute["block"].shift(-horizon)
    times = minute.index.to_series()
    right_time = times.shift(-horizon) - times == pd.Timedelta(minutes=horizon)
    full = (minute["n"] == 60) & (minute["n"].shift(-horizon) == 60)
    result = np.log(minute["close"].shift(-horizon) / minute["close"])
    return result.where(same_block & right_time & full)


def load_analysis_data(config):
    """Read audited files one day at a time and keep the smaller aggregates."""
    minute_parts = []
    hour_parts = []
    daily_rows = []
    totals = {}
    second_daily = []
    previous_time = None
    previous_close = None
    windows = day_windows(config)
    for i, window in enumerate(windows):
        frame = read_partition(partition_path(config, window))
        if frame.empty:
            previous_time = None
            previous_close = None
            continue
        returns = second_returns(frame, previous_time, previous_close)
        previous_time = frame["open_time"].iloc[-1]
        previous_close = frame["close"].iloc[-1]
        year = str(frame["open_time"].iloc[0].year)
        times = frame["open_time"]
        # Accumulate these pairs a day at a time to keep memory use small.
        for lag in [1, 5, 60, 300]:
            adjacent = times - times.shift(lag) == pd.Timedelta(seconds=lag)
            previous = returns.shift(lag).where(adjacent)
            for metric, x, y in [("return", previous, returns), ("absolute_return", previous.abs(), returns.abs())]:
                for label in ["All", year]:
                    values = add_correlation(totals, (label, "1s", metric, lag), x, y)
                row = {"date": pd.Timestamp(window["date"], tz="UTC"), "frequency": "1s", "metric": metric, "lag": lag}
                for name, value in zip(MOMENTS, values):
                    row[name] = value
                second_daily.append(row)
        minute, hourly, daily = aggregate_day(frame, window, returns)
        minute_parts.append(minute)
        hour_parts.append(hourly)
        daily_rows.append(daily)
        if (i + 1) % 100 == 0 or i + 1 == len(windows):
            logging.info("Aggregated %s / %s days", i + 1, len(windows))
    if not minute_parts:
        raise ValueError("No observed records are available for analysis.")
    minute = minute_features(pd.concat(minute_parts))
    daily = pd.DataFrame(daily_rows)
    daily["year"] = daily["date"].dt.year.astype(str)
    daily["month"] = daily["date"].dt.strftime("%Y-%m")
    daily["weekday"] = daily["date"].dt.dayofweek
    daily["zero_volume_pct"] = 100 * daily["zero_volume_seconds"] / daily["rows"]
    return minute, pd.concat(hour_parts, ignore_index=True), daily, totals, pd.DataFrame(second_daily)


# Result tables

def add_minute_correlations(minute, totals):
    """Add pooled and yearly minute correlations, including current and future returns."""
    periods = [("All", minute)]
    for year, group in minute.groupby(minute.index.year):
        periods.append((str(year), group))
    for year, group in periods:
        for lag in [1, 5, 30, 60]:
            same_block = group["block"] == group["block"].shift(lag)
            times = group.index.to_series()
            adjacent = times - times.shift(lag) == pd.Timedelta(minutes=lag)
            previous = group["return"].shift(lag).where(same_block & adjacent)
            add_correlation(totals, (year, "1min", "return", lag), previous, group["return"])
            add_correlation(totals, (year, "1min", "absolute_return", lag), previous.abs(), group["abs_return"])
        add_correlation(totals, (year, "1min", "imbalance_vs_return", 0), group["imbalance"], group["return"])
        for horizon in [1, 5, 15]:
            future = future_returns(group, horizon)
            add_correlation(totals, (year, "1min", "imbalance_vs_return", horizon), group["imbalance"], future)


def activity_table(daily, period):
    """Summarize activity by year or month using the actual observed seconds."""
    rows = []
    for name, group in daily.groupby(period):
        seconds = int(group["rows"].sum())
        trades = int(group["trades"].sum())
        average_trade = np.nan
        if trades > 0:
            average_trade = group["turnover_usdt"].sum() / trades
        rows.append({period: name, "start_date": group["date"].min(),
                     "end_date": group["date"].max(), "days": len(group),
                     "complete_days": int(group["complete_day"].sum()), "observed_seconds": seconds,
                     "zero_volume_pct": 100 * group["zero_volume_seconds"].sum() / seconds,
                     "trades_per_second": trades / seconds, "mean_trade_usdt": average_trade,
                     "mean_daily_volume_btc": group["volume_btc"].mean(),
                     "mean_daily_turnover_usdt": group["turnover_usdt"].mean(),
                     "median_daily_rv_pct": group["rv_pct"].median()})
    return pd.DataFrame(rows)


def hourly_table(daily, hourly):
    """Calculate hourly shares for each complete day, then give each day equal weight."""
    dates = daily.loc[daily["complete_day"], "date"]
    selected = hourly[hourly["date"].isin(dates)].copy()
    for source, target in [("volume", "volume_share_pct"), ("rv2", "variance_share_pct")]:
        total = selected.groupby("date")[source].transform("sum")
        selected[target] = 100 * selected[source] / total.where(total > 0)
    periods = [("All", selected)]
    for year, group in selected.groupby(selected["date"].dt.year):
        periods.append((str(year), group))
    rows = []
    for year, group in periods:
        for hour, part in group.groupby("hour"):
            rows.append({"year": year, "hour_utc": hour,
                         "volume_days": int(part["volume_share_pct"].count()),
                         "variance_days": int(part["variance_share_pct"].count()),
                         "volume_share_pct": part["volume_share_pct"].mean(),
                         "variance_share_pct": part["variance_share_pct"].mean()})
    columns = ["year", "hour_utc", "volume_days", "variance_days", "volume_share_pct", "variance_share_pct"]
    return pd.DataFrame(rows, columns=columns)


def weekly_table(daily):
    """Compare weekend and weekday daily averages within complete Monday-Sunday weeks."""
    work = daily[daily["complete_day"]].copy()
    work["week"] = work["date"] - pd.to_timedelta(work["weekday"], unit="D")
    rows = []
    for week, group in work.groupby("week"):
        if len(group) != 7:
            continue
        weekdays = group[group["weekday"] < 5]
        weekends = group[group["weekday"] >= 5]
        row = {"week_start": week}
        for source, target in [("volume_btc", "volume_ratio"), ("rv_pct", "rv_ratio")]:
            denominator = weekdays[source].mean()
            row[target] = np.nan
            if denominator > 0:
                row[target] = weekends[source].mean() / denominator
        rows.append(row)
    return pd.DataFrame(rows, columns=["week_start", "volume_ratio", "rv_ratio"])


def weekly_summary(weekly, metrics=("volume_ratio", "rv_ratio")):
    """Summarize weekly ratios using blocks of four adjacent calendar weeks."""
    rows = []
    for metric in metrics:
        values = weekly[metric].dropna().to_numpy()
        median = np.nan
        lower_fraction = np.nan
        if len(values) > 0:
            median = float(np.median(values))
            lower_fraction = float((values < 1).mean())
        calendar_values = np.array([])
        if not weekly.empty:
            dates = pd.date_range(weekly["week_start"].min(), weekly["week_start"].max(), freq="7D")
            calendar_values = weekly.set_index("week_start")[metric].reindex(dates).to_numpy()
        check = block_test(calendar_values, "median", 4, null=1)
        rows.append({"metric": metric, "weeks": len(values), "median_ratio": median,
                     "fraction_below_one": lower_fraction, "ci95_low": check["ci95_low"],
                     "ci95_high": check["ci95_high"], "block_weeks": 4,
                     "bootstrap_draws": check["bootstrap_draws"], "status": check["status"]})
    return pd.DataFrame(rows)


def robustness_table(daily, hourly):
    """Repeat the seasonal comparisons after removing the top 1% of daily volatility."""
    complete = daily[daily["complete_day"]].copy()
    threshold = complete["rv_pct"].quantile(0.99)
    filtered = complete[complete["rv_pct"] <= threshold]
    rows = []
    for name, group in [("complete_days", complete), ("without_top_1pct_rv", filtered)]:
        hours = hourly_table(group, hourly)
        peak = hours[(hours["year"] == "All") & hours["hour_utc"].isin([14, 15])]
        peak_share = np.nan
        if len(peak) == 2:
            peak_share = peak["volume_share_pct"].sum(min_count=2)
        weeks = weekly_table(group)
        rows.append({"sample": name, "days": len(group), "weeks": len(weeks),
                     "rv_exclusion_threshold_pct": threshold,
                     "volume_share_14_16_pct": peak_share,
                     "median_weekend_volume_ratio": weeks["volume_ratio"].median(),
                     "median_weekend_rv_ratio": weeks["rv_ratio"].median()})
    return pd.DataFrame(rows)


# Validation

def validation_periods(config):
    """Use fixed calendar boundaries for training, validation and retrospective testing."""
    start = pd.Timestamp(config["start"])
    end = pd.Timestamp(config["end"])
    cuts = [("train", start, pd.Timestamp("2025-01-01", tz="UTC")),
            ("validation", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
            ("retrospective_test", pd.Timestamp("2026-01-01", tz="UTC"), end)]
    rows = []
    for stage, left, right in cuts:
        left = max(left, start)
        right = min(right, end)
        if left < right:
            rows.append({"stage": stage, "start": left, "end_exclusive": right})
    return pd.DataFrame(rows, columns=["stage", "start", "end_exclusive"])


def forecast_samples(minute, horizon):
    """Use finished-minute features and record when the future target ends."""
    result = minute[["return", "imbalance"]].copy()
    result["available_at"] = minute.index + pd.Timedelta(minutes=1)
    result["target_end"] = result["available_at"] + pd.Timedelta(minutes=horizon)
    result["target"] = future_returns(minute, horizon)
    return result.dropna(subset=["return", "imbalance", "target"])


def period_samples(samples, start, end):
    """Drop predictions whose target would reach beyond the period boundary."""
    inside = (samples["available_at"] >= start) & (samples["available_at"] < end)
    usable = inside & (samples["target_end"] <= end)
    return samples.loc[usable].copy(), int((inside & ~usable).sum())


def daily_pair_sums(x, y, dates, selected):
    """Keep daily sufficient statistics for paired observations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.asarray(selected) & np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    work = pd.DataFrame({"date": dates[valid], "pairs": 1, "sum_x": x, "sum_y": y,
                         "sum_xx": x * x, "sum_yy": y * y, "sum_xy": x * y})
    return work.groupby("date")[MOMENTS].sum().reset_index()


def correlation_days(minute, second_daily, periods):
    """Save daily pair sums for each frequency, lag and chronological stage."""
    parts = []
    ready = minute.index + pd.Timedelta(minutes=1)
    dates = ready.normalize()
    for period in periods.itertuples(index=False):
        second = second_daily[(second_daily["date"] >= period.start) &
                              (second_daily["date"] + pd.Timedelta(days=1) <= period.end_exclusive)].copy()
        second["stage"] = period.stage
        parts.append(second)
        pairs = []
        for lag in [1, 5, 30, 60]:
            same = minute["block"] == minute["block"].shift(-lag)
            times = minute.index.to_series()
            adjacent = times.shift(-lag) - times == pd.Timedelta(minutes=lag)
            for metric, source in [("return", "return"), ("absolute_return", "abs_return")]:
                pairs.append((metric, lag, minute[source], minute[source].shift(-lag).where(same & adjacent)))
        pairs.append(("imbalance_vs_return", 0, minute["imbalance"], minute["return"]))
        for horizon in [1, 5, 15]:
            pairs.append(("imbalance_vs_return", horizon, minute["imbalance"], future_returns(minute, horizon)))
        for metric, lag, x, y in pairs:
            selected = (ready >= period.start) & (ready < period.end_exclusive)
            selected = selected & (ready + pd.Timedelta(minutes=lag) <= period.end_exclusive)
            part = daily_pair_sums(x, y, dates, selected)
            part["frequency"] = "1min"
            part["metric"] = metric
            part["lag"] = lag
            part["stage"] = period.stage
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fit_forecast(samples, feature):
    """Fit a constant or a one-feature least-squares forecast using training rows."""
    if len(samples) < 30:
        return None
    model = {"feature": feature, "train_pairs": len(samples), "center": 0.0, "scale": 1.0,
             "intercept": float(samples["target"].mean()), "slope": 0.0}
    if feature != "mean":
        x = samples[feature].to_numpy()
        model["center"] = float(x.mean())
        model["scale"] = float(x.std())
        if model["scale"] <= 0:
            return None
        z = (x - model["center"]) / model["scale"]
        y = samples["target"].to_numpy() - model["intercept"]
        model["slope"] = float(np.dot(z, y) / np.dot(z, z))
    return model


def predict_forecast(samples, model):
    """Apply frozen training parameters to the finished-minute features."""
    result = np.full(len(samples), model["intercept"])
    if model["feature"] != "mean":
        x = samples[model["feature"]].to_numpy()
        result += model["slope"] * (x - model["center"]) / model["scale"]
    return result


def forecast_tables(minute, periods):
    """Fit on training data, select on validation MSE and keep the choice fixed for testing."""
    audits = []
    models = []
    scores = []
    daily_parts = []
    for horizon in [1, 5, 15]:
        samples = forecast_samples(minute, horizon)
        groups = {}
        for period in periods.itertuples(index=False):
            group, purged = period_samples(samples, period.start, period.end_exclusive)
            groups[period.stage] = group
            audits.append({"stage": period.stage, "horizon_minutes": horizon, "pairs": len(group),
                           "boundary_pairs_removed": purged, "first_available_at": group["available_at"].min(),
                           "last_available_at": group["available_at"].max(), "last_target_end": group["target_end"].max(),
                           "end_exclusive": period.end_exclusive, "features": "return,imbalance",
                           "target_within_period": bool((group["target_end"] <= period.end_exclusive).all())})
        train = groups.get("train")
        if train is None or len(train) < 30:
            continue
        fitted = {}
        for feature in ["mean", "return", "imbalance"]:
            model = fit_forecast(train, feature)
            if model is not None:
                model["horizon_minutes"] = horizon
                model["fit_start"] = train["available_at"].min()
                model["fit_last_target_end"] = train["target_end"].max()
                fitted[feature] = model
        chosen = ""
        validation = groups.get("validation")
        if validation is not None and not validation.empty:
            best_loss = np.inf
            for feature, model in fitted.items():
                error = validation["target"].to_numpy() - predict_forecast(validation, model)
                loss = float(np.mean(error * error))
                if loss < best_loss:
                    chosen = feature
                    best_loss = loss
        for feature, model in fitted.items():
            model["selected_on_validation"] = feature == chosen
            models.append(model)
        for stage, group in groups.items():
            if group.empty:
                continue
            actual = group["target"].to_numpy()
            base_error = actual - fitted["mean"]["intercept"]
            base_mse = float(np.mean(base_error * base_error))
            for feature, model in fitted.items():
                error = actual - predict_forecast(group, model)
                mse = float(np.mean(error * error))
                gain = np.nan
                if base_mse > 0:
                    gain = 1 - mse / base_mse
                scores.append({"stage": stage, "horizon_minutes": horizon, "model": feature, "pairs": len(group),
                               "mse": mse, "baseline_mse": base_mse, "r2_vs_train_mean": gain,
                               "selected_on_validation": feature == chosen})
                if feature == chosen:
                    work = pd.DataFrame({"date": group["available_at"].dt.normalize(), "pairs": 1,
                                         "baseline_sse": base_error * base_error, "model_sse": error * error})
                    work = work.groupby("date")[["pairs", "baseline_sse", "model_sse"]].sum().reset_index()
                    work["stage"] = stage
                    work["horizon_minutes"] = horizon
                    work["model"] = feature
                    daily_parts.append(work)
    audit_columns = ["stage", "horizon_minutes", "pairs", "boundary_pairs_removed", "first_available_at",
                     "last_available_at", "last_target_end", "end_exclusive", "features", "target_within_period"]
    model_columns = ["feature", "train_pairs", "center", "scale", "intercept", "slope", "horizon_minutes",
                     "fit_start", "fit_last_target_end", "selected_on_validation"]
    score_columns = ["stage", "horizon_minutes", "model", "pairs", "mse", "baseline_mse",
                     "r2_vs_train_mean", "selected_on_validation"]
    daily = pd.DataFrame(columns=["date", "pairs", "baseline_sse", "model_sse", "stage", "horizon_minutes", "model"])
    if daily_parts:
        daily = pd.concat(daily_parts, ignore_index=True)
    return pd.DataFrame(audits, columns=audit_columns), pd.DataFrame(models, columns=model_columns), pd.DataFrame(scores, columns=score_columns), daily


def block_indices(length, block_size, draws=BOOTSTRAP_DRAWS):
    """Sample adjacent calendar blocks with replacement and circular wrapping."""
    rng = np.random.default_rng(42)
    result = np.empty((draws, length), dtype=np.int32)
    blocks = (length + block_size - 1) // block_size
    for i in range(draws):
        starts = rng.integers(0, length, size=blocks)
        selected = (starts[:, None] + np.arange(block_size)) % length
        result[i] = selected.ravel()[:length]
    return result


def correlations_from_sums(values):
    """Calculate correlations from one or more rows of sufficient statistics."""
    values = np.asarray(values, dtype=float).reshape(-1, 6)
    n, sx, sy, sxx, syy, sxy = values.T
    denominator = np.where(n > 0, n, np.nan)
    vx = sxx - sx * sx / denominator
    vy = syy - sy * sy / denominator
    result = np.full(len(values), np.nan)
    valid = (n >= 2) & (vx > 0) & (vy > 0)
    result[valid] = (sxy[valid] - sx[valid] * sy[valid] / n[valid]) / np.sqrt(vx[valid] * vy[valid])
    return np.clip(result, -1, 1)


def block_test(values, kind, block_size, null=0.0, indices=None):
    """Return a basic bootstrap interval and a centered two-sided bootstrap p-value."""
    values = np.asarray(values, dtype=float)
    if kind == "correlation":
        estimate = correlations_from_sums(values.sum(axis=0))[0]
        usable = int((values[:, 0] > 0).sum())
    elif kind == "forecast_gain":
        total = values.sum(axis=0)
        estimate = np.nan
        if total[1] > 0:
            estimate = 1 - total[2] / total[1]
        usable = int((values[:, 0] > 0).sum())
    else:
        finite = values[np.isfinite(values)]
        usable = len(finite)
        estimate = np.nan
        if usable:
            estimate = float(np.mean(finite))
            if kind == "median":
                estimate = float(np.median(finite))
    row = {"estimate": estimate, "null": null, "calendar_units": len(values), "usable_units": usable,
           "block_size": block_size, "ci95_low": np.nan, "ci95_high": np.nan,
           "p_value": np.nan, "bootstrap_draws": 0, "status": "insufficient_blocks"}
    if not np.isfinite(estimate) or usable < 4 * block_size:
        return row
    if indices is None:
        indices = block_indices(len(values), block_size)
    if kind in ["correlation", "forecast_gain"]:
        totals = values[indices].sum(axis=1)
        if kind == "correlation":
            samples = correlations_from_sums(totals)
        else:
            denominator = np.where(totals[:, 1] > 0, totals[:, 1], np.nan)
            samples = 1 - totals[:, 2] / denominator
    else:
        selected = values[indices]
        selected = selected[np.isfinite(selected).any(axis=1)]
        samples = np.nanmean(selected, axis=1)
        if kind == "median":
            samples = np.nanmedian(selected, axis=1)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 0.95 * len(indices):
        row["status"] = "too_few_valid_resamples"
        return row
    low, high = 2 * estimate - np.quantile(samples, [0.975, 0.025])
    if kind == "correlation":
        low, high = np.clip([low, high], -1, 1)
    row["ci95_low"] = float(low)
    row["ci95_high"] = float(high)
    row["p_value"] = float((1 + (np.abs(samples - estimate) >= abs(estimate - null)).sum()) / (len(samples) + 1))
    row["bootstrap_draws"] = len(samples)
    row["status"] = "ok"
    return row


def holm_adjust(table):
    """Adjust all planned tests within each stage and block-length setting."""
    result = table.copy()
    result["p_holm"] = np.nan
    result["family_tests"] = 0
    for key, group in result.groupby(["stage", "block_setting"]):
        ordered = group["p_value"].dropna().sort_values()
        previous = 0.0
        result.loc[group.index, "family_tests"] = len(group)
        for rank, (index, value) in enumerate(ordered.items()):
            adjusted = min(1.0, max(previous, (len(group) - rank) * value))
            result.loc[index, "p_holm"] = adjusted
            previous = adjusted
    result["reject_null_5pct"] = result["p_holm"] < 0.05
    result["supported_at_both_block_lengths"] = False
    for key, group in result.groupby(["stage", "test_id"]):
        effects = group["estimate"] - group["null"]
        same_sign = (effects > 0).all() or (effects < 0).all()
        if len(group) == 2 and group["reject_null_5pct"].all() and same_sign:
            result.loc[group.index, "supported_at_both_block_lengths"] = True
    return result


# Regional time windows

def session_calendar(daily):
    """Convert each local 09:00-17:00 window to UTC using the timezone's DST rules."""
    rows = []
    for date, complete in zip(daily["date"], daily["complete_day"]):
        day_rows = []
        dst_names = []
        for name, zone in SESSIONS:
            local_start = (date.tz_localize(None) + pd.Timedelta(hours=9)).tz_localize(zone)
            local_end = (date.tz_localize(None) + pd.Timedelta(hours=17)).tz_localize(zone)
            is_dst = local_start.dst().total_seconds() != 0
            day_type = "Weekday"
            if local_start.dayofweek >= 5:
                day_type = "Weekend"
            if name != "Hong Kong":
                state = "standard"
                if is_dst:
                    state = "DST"
                dst_names.append(name + " " + state)
            day_rows.append({"date": date, "session": name, "timezone": zone,
                             "local_start": "09:00", "local_end_exclusive": "17:00",
                             "start_utc": local_start.tz_convert("UTC"),
                             "end_utc": local_end.tz_convert("UTC"),
                             "utc_offset_hours": local_start.utcoffset().total_seconds() / 3600,
                             "is_dst": is_dst, "day_type": day_type, "complete_day": bool(complete)})
        for row in day_rows:
            row["dst_regime"] = " / ".join(dst_names)
            rows.append(row)
    return pd.DataFrame(rows)


def session_labels(times):
    """Keep each active window flag and an exclusive label for their combination."""
    labels = pd.DataFrame(index=times)
    labels["session"] = "Outside"
    for name, zone in SESSIONS:
        local = times.tz_convert(zone)
        active = (local.hour >= 9) & (local.hour < 17)
        labels[name] = active
        first = active & (labels["session"] == "Outside")
        overlap = active & ~first
        labels.loc[first, "session"] = name
        labels.loc[overlap, "session"] = labels.loc[overlap, "session"] + " + " + name
    return labels


def session_daily_table(minute, daily, calendar):
    """Sum complete days into inclusive city windows and exclusive overlap buckets."""
    dates = daily.loc[daily["complete_day"], "date"]
    columns = ["volume", "quote_volume", "trade_count", "n", "rv2", "zero_seconds", "valid_returns"]
    work = minute.loc[minute.index.normalize().isin(dates), columns].copy()
    work["date"] = work.index.normalize()
    work["minutes"] = 1
    labels = session_labels(work.index)
    parts = []
    for name, zone in SESSIONS:
        part = work.loc[labels[name]].groupby("date")[columns + ["minutes"]].sum().reset_index()
        part["scheme"] = "regional"
        part["session"] = name
        parts.append(part)
    work["session"] = labels["session"]
    part = work.groupby(["date", "session"])[columns + ["minutes"]].sum().reset_index()
    part["scheme"] = "exclusive"
    parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result["year"] = result["date"].dt.year.astype(str)
    result["day_type"] = "Weekday"
    result.loc[result["date"].dt.dayofweek >= 5, "day_type"] = "Weekend"
    regimes = calendar.drop_duplicates("date").set_index("date")["dst_regime"]
    result["dst_regime"] = result["date"].map(regimes)
    result["turnover_usdt_per_minute"] = result["quote_volume"] / result["minutes"]
    result["variance_bps2_per_minute"] = 1e8 * result["rv2"] / result["minutes"]
    result["rv_pct"] = 100 * np.sqrt(result["rv2"])
    result["trades_per_second"] = result["trade_count"] / result["n"]
    result["zero_volume_pct"] = 100 * result["zero_seconds"] / result["n"]
    # Compare each bucket with the average minute of its own UTC day.
    totals = work.groupby("date")[["quote_volume", "rv2"]].sum()
    for source, target in [("quote_volume", "turnover_intensity"), ("rv2", "variance_intensity")]:
        baseline = result["date"].map(totals[source]) / 1440
        result[target] = result[source] / result["minutes"] / baseline.where(baseline > 0)
    return result


def session_summary_table(daily_sessions):
    """Compare activity per unit of time and average the within-day intensities equally."""
    periods = [("All", daily_sessions)]
    for year, part in daily_sessions.groupby("year"):
        periods.append((year, part))
    rows = []
    for year, period in periods:
        for day_type in ["All", "Weekday", "Weekend"]:
            selected = period
            if day_type != "All":
                selected = period[period["day_type"] == day_type]
            for key, group in selected.groupby(["scheme", "session"]):
                minutes = int(group["minutes"].sum())
                seconds = int(group["n"].sum())
                rows.append({"year": year, "day_type": day_type, "scheme": key[0], "session": key[1],
                             "days": len(group), "start_date": group["date"].min(), "end_date": group["date"].max(),
                             "minutes": minutes, "mean_minutes_per_day": group["minutes"].mean(),
                             "turnover_usdt_per_minute": group["quote_volume"].sum() / minutes,
                             "volume_btc_per_minute": group["volume"].sum() / minutes,
                             "trades_per_second": group["trade_count"].sum() / seconds,
                             "variance_bps2_per_minute": 1e8 * group["rv2"].sum() / minutes,
                             "zero_volume_pct": 100 * group["zero_seconds"].sum() / seconds,
                             "valid_second_returns": int(group["valid_returns"].sum()),
                             "turnover_days": int(group["turnover_intensity"].count()),
                             "variance_days": int(group["variance_intensity"].count()),
                             "mean_turnover_intensity": group["turnover_intensity"].mean(),
                             "mean_variance_intensity": group["variance_intensity"].mean()})
    columns = ["year", "day_type", "scheme", "session", "days", "start_date", "end_date", "minutes",
               "mean_minutes_per_day", "turnover_usdt_per_minute", "volume_btc_per_minute", "trades_per_second",
               "variance_bps2_per_minute", "zero_volume_pct", "valid_second_returns", "turnover_days",
               "variance_days", "mean_turnover_intensity", "mean_variance_intensity"]
    return pd.DataFrame(rows, columns=columns)


def session_weekend_tables(daily_sessions):
    """Pair weekend and weekday rates within full weeks, then bootstrap whole weeks."""
    work = daily_sessions.copy()
    work["week_start"] = work["date"] - pd.to_timedelta(work["date"].dt.dayofweek, unit="D")
    rows = []
    for key, group in work.groupby(["scheme", "session", "week_start"]):
        if len(group) != 7:
            continue
        row = {"scheme": key[0], "session": key[1], "week_start": key[2]}
        weekdays = group[group["day_type"] == "Weekday"]
        weekends = group[group["day_type"] == "Weekend"]
        for source, target in [("turnover_usdt_per_minute", "turnover_ratio"),
                               ("variance_bps2_per_minute", "variance_ratio")]:
            denominator = weekdays[source].mean()
            row[target] = np.nan
            if denominator > 0:
                row[target] = weekends[source].mean() / denominator
        rows.append(row)
    weekly = pd.DataFrame(rows, columns=["scheme", "session", "week_start", "turnover_ratio", "variance_ratio"])
    summaries = []
    for key, group in weekly.groupby(["scheme", "session"]):
        summary = weekly_summary(group, ["turnover_ratio", "variance_ratio"])
        summary["scheme"] = key[0]
        summary["session"] = key[1]
        summaries.append(summary)
    columns = ["metric", "weeks", "median_ratio", "fraction_below_one", "ci95_low", "ci95_high", "scheme", "session"]
    combined = pd.DataFrame(columns=columns)
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
    return weekly, combined


def session_hourly_table(minute, daily, calendar):
    """Compare UTC-hour profiles by DST regime, day type and year using equal day weights."""
    dates = daily.loc[daily["complete_day"], "date"]
    work = minute.loc[minute.index.normalize().isin(dates), ["quote_volume", "rv2"]].copy()
    work["date"] = work.index.normalize()
    work["hour_utc"] = work.index.hour
    hours = work.groupby(["date", "hour_utc"])[["quote_volume", "rv2"]].sum().reset_index()
    for source, target in [("quote_volume", "turnover_share_pct"), ("rv2", "variance_share_pct")]:
        total = hours.groupby("date")[source].transform("sum")
        hours[target] = 100 * hours[source] / total.where(total > 0)
    regimes = calendar.drop_duplicates("date").set_index("date")["dst_regime"]
    hours["dst_regime"] = hours["date"].map(regimes)
    hours["day_type"] = "Weekday"
    hours.loc[hours["date"].dt.dayofweek >= 5, "day_type"] = "Weekend"
    periods = [("All", hours)]
    for year, group in hours.groupby(hours["date"].dt.year):
        periods.append((str(year), group))
    rows = []
    for year, period in periods:
        for key, group in period.groupby(["dst_regime", "day_type", "hour_utc"]):
            rows.append({"year": year, "dst_regime": key[0], "day_type": key[1], "hour_utc": key[2],
                         "days": len(group), "turnover_days": int(group["turnover_share_pct"].count()),
                         "variance_days": int(group["variance_share_pct"].count()),
                         "turnover_share_pct": group["turnover_share_pct"].mean(),
                         "variance_share_pct": group["variance_share_pct"].mean()})
    columns = ["year", "dst_regime", "day_type", "hour_utc", "days", "turnover_days", "variance_days",
               "turnover_share_pct", "variance_share_pct"]
    return pd.DataFrame(rows, columns=columns)


def validation_tables(minute, second_daily, tables, config):
    """Build chronological forecasts and block-based checks of the main patterns."""
    periods = validation_periods(config)
    pairs = correlation_days(minute, second_daily, periods)
    audit_rows, models, scores, losses = forecast_tables(minute, periods)
    rows = []
    for period in periods.itertuples(index=False):
        dates = pd.date_range(period.start.normalize(), period.end_exclusive.ceil("D"), freq="D", inclusive="left")
        weeks = pd.date_range(period.start.normalize(), period.end_exclusive, freq="W-MON", inclusive="left")
        jobs = []
        group = pairs[pairs["stage"] == period.stage]
        for key, part in group.groupby(["frequency", "metric", "lag"]):
            values = part.set_index("date")[MOMENTS].reindex(dates, fill_value=0).to_numpy()
            jobs.append({"test_id": "corr/%s/%s/%s" % key, "values": values,
                         "kind": "correlation", "null": 0, "unit": "day"})
        sessions = tables["session_daily"]
        sessions = sessions[(sessions["date"] >= period.start) &
                            (sessions["date"] + pd.Timedelta(days=1) <= period.end_exclusive)]
        for scheme, name in [("regional", "Hong Kong"), ("regional", "London"),
                             ("regional", "New York"), ("exclusive", "London + New York")]:
            part = sessions[(sessions["scheme"] == scheme) & (sessions["session"] == name)].set_index("date")
            for metric in ["turnover_intensity", "variance_intensity"]:
                jobs.append({"test_id": "session/" + name + "/" + metric, "values": part[metric].reindex(dates).to_numpy(),
                             "kind": "mean", "null": 1, "unit": "day"})
        week_tables = [("All", tables["weekly"], ["volume_ratio", "rv_ratio"])]
        for name, zone in SESSIONS:
            part = tables["session_weekly"]
            part = part[(part["scheme"] == "regional") & (part["session"] == name)]
            week_tables.append((name, part, ["turnover_ratio", "variance_ratio"]))
        for name, part, metrics in week_tables:
            part = part.copy()
            part["week_start"] = pd.to_datetime(part["week_start"], utc=True)
            part = part[(part["week_start"] >= period.start) &
                        (part["week_start"] + pd.Timedelta(days=7) <= period.end_exclusive)]
            for metric in metrics:
                values = part.set_index("week_start")[metric].reindex(weeks).to_numpy()
                jobs.append({"test_id": "weekend/" + name + "/" + metric, "values": values,
                             "kind": "median", "null": 1, "unit": "week"})
        for horizon, part in losses[losses["stage"] == period.stage].groupby("horizon_minutes"):
            values = part.set_index("date")[["pairs", "baseline_sse", "model_sse"]].reindex(dates, fill_value=0).to_numpy()
            jobs.append({"test_id": "forecast/" + str(horizon), "values": values,
                         "kind": "forecast_gain", "null": 0, "unit": "day"})
        index_cache = {}
        for setting, day_size, week_size in [("primary", 7, 4), ("longer_blocks", 28, 8)]:
            for job in jobs:
                size = day_size
                if job["unit"] == "week":
                    size = week_size
                key = (len(job["values"]), size)
                if key not in index_cache and key[0] > 0:
                    index_cache[key] = block_indices(key[0], size)
                check = block_test(job["values"], job["kind"], size, job["null"], index_cache.get(key))
                check.update({"stage": period.stage, "test_id": job["test_id"], "statistic": job["kind"],
                              "block_unit": job["unit"], "block_setting": setting})
                rows.append(check)
        logging.info("Finished chronological validation: %s", period.stage)
    inference = holm_adjust(pd.DataFrame(rows))
    return {"validation_periods": periods, "feature_audit": audit_rows, "forecast_models": models,
            "forecast_scores": scores, "correlation_daily": pairs, "forecast_daily": losses, "inference": inference}


# Charts

def save_chart(fig, output, name, period, bottom=0.055):
    """Add the source and date range, save a PNG and close the figure."""
    fig.text(0.02, 0.02, "Binance spot BTCUSDT | UTC | " + period, fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, bottom, 1, 0.96])
    fig.savefig(output / name, dpi=200, facecolor="white")
    plt.close(fig)


def draw_charts(tables, output, period):
    """Draw the six analysis charts from the saved table definitions."""
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    daily = tables["daily"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Market overview: price, trading activity and realized volatility")
    axes[0].plot(daily["date"], daily["close"], color="#245A81")
    axes[0].set_ylabel("Close (USDT)")
    axes[1].plot(daily["date"], daily["turnover_usdt"] / 1e9, color="#228C83")
    axes[1].set_ylabel("Turnover\n(billion USDT)")
    axes[2].plot(daily["date"], daily["rv_pct"], color="#C17632")
    axes[2].set_ylabel("Daily RV (%)")
    save_chart(fig, output, "01_overview.png", period)

    hours = tables["hourly"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle("Intraday patterns: average share of each UTC day")
    for axis, metric, title in [(axes[0], "volume_share_pct", "BTC volume share (%)"),
                                (axes[1], "variance_share_pct", "Realized variance share (%)")]:
        for year, group in hours.groupby("year", sort=False):
            width = 1.3
            if year == "All":
                width = 3
            axis.plot(group["hour_utc"], group[metric], label=year, linewidth=width)
        axis.axhline(100 / 24, color="gray", linestyle="--", label="Uniform: 4.17%")
        axis.set(title=title, xlabel="Hour (UTC)", xticks=[0, 4, 8, 12, 16, 20, 23])
        axis.legend(fontsize=9)
    if hours.empty:
        axes[0].text(0.1, 0.5, "No complete UTC days", transform=axes[0].transAxes)
    save_chart(fig, output, "02_intraday.png", period)

    weeks = tables["weekly"]
    fig, axis = plt.subplots(figsize=(13, 7))
    axis.set_title("Weekend / weekday daily averages within the same complete week")
    for position, metric in enumerate(["volume_ratio", "rv_ratio"]):
        values = weeks[metric].dropna()
        if len(values) > 0:
            axis.boxplot([values], positions=[position + 1], widths=0.45, showfliers=False)
            axis.text(position + 1.28, values.median(), "Median: %.3f\nWeeks: %s" % (values.median(), len(values)),
                      fontsize=11, va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    axis.axhline(1, color="gray", linestyle="--", label="Equal activity")
    axis.set(xticks=[1, 2], xticklabels=["BTC volume", "Realized volatility"], ylabel="Weekend / weekday ratio", xlim=(0.5, 2.9))
    axis.legend()
    if weeks.empty:
        axis.text(0.2, 0.5, "No complete Monday-Sunday weeks", transform=axis.transAxes)
    save_chart(fig, output, "03_weekend.png", period)

    correlations = tables["correlations"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle("Persistence of returns and absolute returns (pooled sample)")
    for axis, frequency, unit in [(axes[0], "1s", "seconds; within UTC day"), (axes[1], "1min", "minutes; continuous blocks")]:
        for metric, label in [("return", "Return"), ("absolute_return", "Absolute return")]:
            part = correlations[(correlations["year"] == "All") & (correlations["frequency"] == frequency) & (correlations["metric"] == metric)]
            axis.plot(part["lag"].astype(str), part["correlation"], marker="o", label=label)
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(title=frequency + " candles", xlabel="Lag (" + unit + ")", ylabel="Pearson correlation")
        axis.legend()
    save_chart(fig, output, "04_autocorrelation.png", period)

    flow = correlations[(correlations["metric"] == "imbalance_vs_return") & (correlations["year"] != "All")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle("Taker-buy imbalance: current versus future return association")
    current = flow[flow["lag"] == 0]
    axes[0].bar(current["year"], current["correlation"], color="#245A81")
    axes[0].set(title="Same-minute return", ylabel="Pearson correlation")
    for lag in [1, 5, 15]:
        part = flow[flow["lag"] == lag]
        axes[1].plot(part["year"], part["correlation"], marker="o", label="Next %s min" % lag)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set(title="Future return (different y-axis scale)", ylabel="Pearson correlation")
    axes[1].legend()
    save_chart(fig, output, "05_imbalance.png", period)

    monthly = tables["monthly"]
    dates = pd.to_datetime(monthly["month"])
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Monthly trading activity (boundary months may be partial)")
    for axis, metric, label in [(axes[0], "zero_volume_pct", "Zero-volume\nseconds (%)"),
                                (axes[1], "trades_per_second", "Trades per\nobserved second"),
                                (axes[2], "mean_trade_usdt", "Mean trade\nsize (USDT)")]:
        axis.plot(dates, monthly[metric], marker="o", markersize=3, color="#228C83")
        axis.set_ylabel(label)
    save_chart(fig, output, "06_activity.png", period)


def draw_session_charts(tables, output, period):
    """Plot city windows, overlaps, DST timing, matched weekends and yearly stability."""
    summary = tables["session_summary"]
    names = []
    for name, zone in SESSIONS:
        names.append(name)
    metrics = [("mean_turnover_intensity", "Turnover intensity"),
               ("mean_variance_intensity", "Variance intensity")]
    regional = summary[(summary["year"] == "All") & (summary["scheme"] == "regional")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle("Local 09:00-17:00 windows: activity relative to each day's average minute")
    positions = np.arange(len(names))
    for axis, metric in zip(axes, metrics):
        for day_type, shift, color in [("Weekday", -0.18, "#245A81"), ("Weekend", 0.18, "#C17632")]:
            part = regional[regional["day_type"] == day_type].set_index("session").reindex(names)
            axis.bar(positions + shift, part[metric[0]], width=0.34, label=day_type, color=color)
        axis.axhline(1, color="gray", linestyle="--")
        axis.set(title=metric[1], xticks=positions, xticklabels=names, ylabel="Multiple of own-day average")
        axis.legend()
    fig.text(0.02, 0.08, "City windows include overlaps. Time labels describe activity windows; trader location is unknown.", fontsize=10)
    save_chart(fig, output, "07_regional_sessions.png", period, bottom=0.12)

    part = summary[(summary["year"] == "All") & (summary["day_type"] == "All") & (summary["scheme"] == "exclusive")]
    part = part.sort_values("mean_turnover_intensity")
    labels = []
    for name, minutes in zip(part["session"], part["mean_minutes_per_day"]):
        labels.append(name + "\n(%.1f h on days present)" % (minutes / 60))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    fig.suptitle("Exclusive time buckets: overlap activity adjusted for duration")
    for axis, metric in zip(axes, metrics):
        axis.barh(np.arange(len(part)), part[metric[0]], color="#228C83")
        axis.axvline(1, color="gray", linestyle="--")
        axis.set(title=metric[1], yticks=np.arange(len(part)), yticklabels=labels,
                 xlabel="Mean multiple of own-day average")
    if part.empty:
        axes[0].text(0.1, 0.5, "No complete UTC days", transform=axes[0].transAxes)
    save_chart(fig, output, "08_session_overlaps.png", period)

    calendar = tables["session_calendar"]
    regimes = calendar["dst_regime"].drop_duplicates().tolist()
    profiles = tables["session_hourly"]
    profiles = profiles[(profiles["year"] == "All") & (profiles["day_type"] == "Weekday")]
    profile_limit = 6.0
    if not profiles.empty:
        peak = profiles[["turnover_share_pct", "variance_share_pct"]].max().max()
        if pd.notna(peak):
            profile_limit = max(profile_limit, float(np.ceil(peak * 1.1)))
    fig, axes = plt.subplots(2, len(regimes), figsize=(5 * len(regimes), 8), squeeze=False)
    fig.suptitle("Daylight saving: local windows in UTC and observed weekday profiles")
    for i, regime in enumerate(regimes):
        example = calendar[calendar["dst_regime"] == regime].drop_duplicates("session").set_index("session")
        for j, name in enumerate(names):
            start = example.loc[name, "start_utc"].hour
            end = example.loc[name, "end_utc"].hour
            axes[0, i].barh(j, end - start, left=start, height=0.55, color="#245A81")
            axes[0, i].text((start + end) / 2, j, "%02d-%02d" % (start, end), color="white", ha="center", va="center")
        axes[0, i].set(title=regime.replace(" / ", "\n"), yticks=[0, 1, 2], yticklabels=names,
                       xlim=(0, 24), xticks=[0, 6, 12, 18, 24], xlabel="UTC hour; end exclusive")
        part = profiles[profiles["dst_regime"] == regime]
        axes[1, i].plot(part["hour_utc"], part["turnover_share_pct"], label="Turnover", color="#245A81")
        axes[1, i].plot(part["hour_utc"], part["variance_share_pct"], label="Variance", color="#C17632")
        axes[1, i].axhline(100 / 24, color="gray", linestyle="--")
        days = 0
        if not part.empty:
            days = int(part["days"].iloc[0])
        axes[1, i].set(title="%s complete weekdays" % days, xlabel="UTC hour", ylabel="Mean share of UTC day (%)",
                       xlim=(0, 23), ylim=(0, profile_limit), xticks=[0, 6, 12, 18, 23])
        axes[1, i].legend(fontsize=10)
    fig.text(0.02, 0.08, "Regimes cover different calendar dates. These profiles do not estimate a causal DST effect.", fontsize=10)
    save_chart(fig, output, "09_dst_profiles.png", period, bottom=0.12)

    weekend = tables["session_weekend_summary"]
    weekend = weekend[weekend["scheme"] == "regional"]
    ratio_limit = 1.15
    if not weekend.empty:
        peak = weekend[["median_ratio", "ci95_high"]].max().max()
        if pd.notna(peak):
            ratio_limit = max(ratio_limit, float(peak) * 1.1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    fig.suptitle("Same-week weekend / weekday rates within each local window")
    for axis, metric, title in [(axes[0], "turnover_ratio", "Turnover per minute"),
                                (axes[1], "variance_ratio", "Realized variance per minute")]:
        part = weekend[weekend["metric"] == metric].set_index("session").reindex(names)
        axis.plot(part["median_ratio"], positions, "o", color="#245A81")
        axis.hlines(positions, part["ci95_low"], part["ci95_high"], color="#245A81", linewidth=2)
        axis.axvline(1, color="gray", linestyle="--")
        axis.set(title=title, yticks=positions, yticklabels=names, xlabel="Median weekly ratio and exploratory 95% interval",
                 ylim=(-0.6, 2.6), xlim=(0, ratio_limit))
    fig.text(0.02, 0.08, "Complete weeks; 1,999 bootstrap draws in four-week blocks. Variance ratios differ from volatility ratios.", fontsize=10)
    save_chart(fig, output, "10_session_weekends.png", period, bottom=0.12)

    yearly = summary[(summary["year"] != "All") & (summary["scheme"] == "regional") & (summary["day_type"] == "Weekday")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle("Yearly stability of weekday session patterns")
    for axis, metric in zip(axes, metrics):
        for name in names:
            part = yearly[yearly["session"] == name]
            axis.plot(part["year"], part[metric[0]], marker="o", label=name)
        axis.axhline(1, color="gray", linestyle="--")
        axis.set(title=metric[1], xlabel="Year (boundary years may be partial)", ylabel="Mean multiple of own-day average")
        axis.legend()
    save_chart(fig, output, "11_session_yearly.png", period)


def draw_validation_charts(tables, output, period):
    """Plot adjusted association checks, frozen forecast scores and seasonal stability."""
    tests = tables["inference"]
    primary = tests[tests["block_setting"] == "primary"]
    stages = ["train", "validation", "retrospective_test"]
    labels = ["2023-24\nTrain", "2025\nValidation", "2026\nRetrospective test"]
    positions = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    fig.suptitle("Imbalance and future returns: chronological association checks")
    for axis, horizon in zip(axes, [1, 5, 15]):
        part = primary[primary["test_id"] == "corr/1min/imbalance_vs_return/" + str(horizon)]
        part = part.set_index("stage").reindex(stages)
        axis.vlines(positions, part["ci95_low"], part["ci95_high"], color="#245A81")
        for i, row in enumerate(part.itertuples()):
            face = "white"
            if row.supported_at_both_block_lengths == True:
                face = "#245A81"
            axis.plot(i, row.estimate, "o", markerfacecolor=face, markeredgecolor="#245A81")
        axis.axhline(0, color="gray", linestyle="--")
        axis.set(title="Next %s minute(s)" % horizon, xticks=positions, xticklabels=labels, ylabel="Pearson correlation")
    fig.text(0.02, 0.08, "7-day block intervals. Filled markers pass Holm adjustment with both 7-day and 28-day blocks.", fontsize=10)
    save_chart(fig, output, "12_validated_associations.png", period, bottom=0.14)

    scores = tables["forecast_scores"]
    selected = scores[scores["selected_on_validation"] == True]
    fig, axis = plt.subplots(figsize=(13, 7))
    axis.set_title("Frozen forecasts: improvement over the training-mean baseline")
    for stage, shift, color in [("validation", -0.18, "#245A81"), ("retrospective_test", 0.18, "#C17632")]:
        part = selected[selected["stage"] == stage].set_index("horizon_minutes").reindex([1, 5, 15])
        axis.bar(positions + shift, 100 * part["r2_vs_train_mean"], width=0.34, color=color, label=stage.replace("_", " "))
        checks = primary[primary["stage"] == stage].set_index("test_id").reindex(["forecast/1", "forecast/5", "forecast/15"])
        axis.vlines(positions + shift, 100 * checks["ci95_low"], 100 * checks["ci95_high"], color="#333333", linewidth=1.5)
    axis.axhline(0, color="gray", linestyle="--")
    axis.set(xticks=positions, xticklabels=["1 minute", "5 minutes", "15 minutes"], ylabel="Reduction in mean squared error (%)")
    axis.legend()
    fig.text(0.02, 0.08, "Fit: 2023-24. Model choice: 2025 MSE. Parameters stay fixed for 2026; this is retrospective validation.\n"
             "Error bars: pointwise 7-day block 95% intervals. Adjusted tests are saved in inference.csv.", fontsize=10)
    save_chart(fig, output, "13_chronological_forecasts.png", period, bottom=0.15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Seasonal pattern stability across chronological periods")
    for metric, label in [("turnover_intensity", "Turnover"), ("variance_intensity", "Variance")]:
        part = primary[primary["test_id"] == "session/London + New York/" + metric]
        part = part.set_index("stage").reindex(stages)
        line = axes[0].plot(positions, part["estimate"], "o-", label=label)[0]
        axes[0].vlines(positions, part["ci95_low"], part["ci95_high"], color=line.get_color())
    for name, zone in SESSIONS:
        part = primary[primary["test_id"] == "weekend/" + name + "/turnover_ratio"]
        part = part.set_index("stage").reindex(stages)
        line = axes[1].plot(positions, part["estimate"], "o-", label=name)[0]
        axes[1].vlines(positions, part["ci95_low"], part["ci95_high"], color=line.get_color())
    for axis in axes:
        axis.axhline(1, color="gray", linestyle="--")
        axis.set(xticks=positions, xticklabels=labels)
        axis.legend(fontsize=10)
    axes[0].set(title="London + New York overlap", ylabel="Mean multiple of own-day activity")
    axes[1].set(title="Weekend / weekday turnover per minute", ylabel="Median same-week ratio")
    axes[0].legend(loc="upper left", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=10)
    fig.text(0.02, 0.08, "Pointwise block intervals. Adjusted tests and longer-block sensitivity are saved in inference.csv.", fontsize=10)
    save_chart(fig, output, "14_pattern_stability.png", period, bottom=0.14)


# CLI

def main(argv=None):
    """Check the data, build the analysis tables and charts, then save the completion summary."""
    parser = argparse.ArgumentParser(description="Option B: BTCUSDT trading patterns")
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--output-dir", type=Path, help="Output folder, relative to the current working directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lock = None
    try:
        config = load_config(args.config)
        output = args.output_dir
        if output is None:
            output = config["report_dir"] / "analysis"
        output = output.resolve()
        data_root = config["data_dir"].resolve()
        if output == data_root or data_root in output.parents:
            raise ValueError("Analysis output must be outside the source data directory.")
        # Use the same lock as the downloader so files stay fixed during analysis.
        lock = create_lock(config["data_dir"])
        logging.info("Checking local data before analysis")
        quality, checked_days, gaps = audit(config)
        if not quality["ready_for_analysis"]:
            logging.error("Data quality is FAIL; analysis stopped. %s", quality)
            return 2
        if quality["quality_status"] == "WARNING":
            logging.warning("Known missing seconds: %s; returns will not cross the gap", quality["known_missing_seconds"])
        minute, hourly, daily, totals, second_daily = load_analysis_data(config)
        add_minute_correlations(minute, totals)
        weekly = weekly_table(daily)
        tables = {"daily": daily, "yearly": activity_table(daily, "year"),
                  "monthly": activity_table(daily, "month"), "hourly": hourly_table(daily, hourly),
                  "weekly": weekly, "weekend_summary": weekly_summary(weekly),
                  "correlations": correlation_table(totals), "robustness": robustness_table(daily, hourly),
                  "extreme_days": daily.nlargest(10, "rv_pct")}
        logging.info("Comparing regional windows, overlaps, DST regimes and weekends")
        calendar = session_calendar(daily)
        daily_sessions = session_daily_table(minute, daily, calendar)
        session_weeks, session_weekends = session_weekend_tables(daily_sessions)
        tables["session_calendar"] = calendar
        tables["session_daily"] = daily_sessions
        tables["session_summary"] = session_summary_table(daily_sessions)
        tables["session_weekly"] = session_weeks
        tables["session_weekend_summary"] = session_weekends
        tables["session_hourly"] = session_hourly_table(minute, daily, calendar)
        logging.info("Checking chronological forecasts and dependence-aware inference")
        tables.update(validation_tables(minute, second_daily, tables, config))
        table_dir = output / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(table_dir / (name + ".csv"), index=False, encoding="utf-8-sig")
        period = iso(config["start"]) + " to " + iso(config["end"]) + " (end exclusive)"
        draw_charts(tables, output / "figures", period)
        draw_session_charts(tables, output / "figures", period)
        draw_validation_charts(tables, output / "figures", period)
        summary = {"generated_at": now_utc(), "status": "COMPLETE", "quality": quality,
                   "observed_minutes": len(minute), "complete_minutes": int((minute["n"] == 60).sum()),
                   "complete_days_for_seasonality": int(daily["complete_day"].sum()),
                   "complete_weeks_for_comparison": len(weekly),
                   "session_daily_rows": len(daily_sessions),
                   "session_tables": 6, "figure_count": 14, "table_count": len(tables),
                   "validation_design": "2023-24 train; 2025 validation; 2026 retrospective test",
                   "bootstrap_draws": BOOTSTRAP_DRAWS,
                   "inference_rows": len(tables["inference"]),
                   "notes": ["Exploratory analysis, not an unseen-test-set result or a backtest.",
                             "No missing seconds or minutes are filled. Returns do not cross gaps.",
                             "Hourly shares use complete UTC days and give every day equal weight.",
                             "Weekly ratios use seven complete UTC days. RV is not annualized.",
                             "Second autocorrelation pairs stay within UTC days; minute pairs stay within continuous blocks.",
                             "Weekly summaries use four-week circular blocks; inference.csv adds eight-week sensitivity and Holm adjustment.",
                             "Top-1% volatility exclusion is a descriptive sensitivity check, not a trading rule.",
                             "Boundary months and years may be partial; see activity tables for actual dates.",
                             "Hong Kong, London and New York windows use local [09:00, 17:00) and historical timezone rules.",
                             "Session labels do not identify trader locations or official cryptocurrency market hours.",
                             "Session comparisons exclude incomplete UTC days. Regional windows include overlaps; exclusive buckets partition each day.",
                             "These city windows stay within the same UTC date; their local weekday matches UTC. Outside uses UTC day type.",
                             "Intensity divides a bucket's per-minute activity by its own UTC-day average, then averages days equally.",
                             "Session weekend ratios compare per-minute turnover and variance within complete weeks; variance is squared volatility.",
                             "DST profiles describe different seasonal samples and do not identify a causal DST effect.",
                             "Forecast inputs are finished-minute return and imbalance; full-day normalized outputs remain descriptive.",
                             "Training scalers and coefficients stay frozen. Validation selects among the mean, return and imbalance models by MSE.",
                             "Forecast targets cannot extend beyond their chronological stage. All models use the same valid feature-target rows.",
                             "Correlation and daily pattern intervals use 7-day circular blocks with 28-day sensitivity; weeks use 4 and 8.",
                             "Inference uses basic bootstrap intervals and approximate centered p-values; Holm families include all planned tests per stage and block setting.",
                             "At least four usable blocks are required. Blocks assume sufficient local stability; tests remain retrospective and exploratory.",
                             "2026 data was inspected before this validation design and is not an untouched holdout."]}
        # Write COMPLETE after all tables and charts have been saved.
        write_json(output / "summary.json", summary)
        logging.info("Analysis complete: %s", output)
        print("Tables: " + str(table_dir))
        print("Charts: " + str(output / "figures"))
        return 0
    except KeyboardInterrupt:
        logging.warning("Analysis interrupted; rerun the same command to restart")
        return 130
    except Exception as error:
        logging.error("%s: %s", type(error).__name__, error)
        return 1
    finally:
        remove_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())
