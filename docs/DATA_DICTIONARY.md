# Canonical data dictionary

One row represents one observed Binance spot BTCUSDT one-second candle. Raw data has 12 exchange fields. The exchange's unused final `ignore` field remains in raw payloads but is omitted from canonical Parquet.

| Column | Canonical Arrow type | Unit / interpretation |
|---|---|---|
| `open_time` | `timestamp[us, tz=UTC]` | Start of candle; unique whole-second identifier within a valid partition |
| `open` | `float64` | First trade price in the second, USDT per BTC |
| `high` | `float64` | Highest trade price in the second, USDT per BTC |
| `low` | `float64` | Lowest trade price in the second, USDT per BTC |
| `close` | `float64` | Last trade price in the second, USDT per BTC |
| `volume` | `float64` | Traded base-asset amount, BTC |
| `close_time` | `timestamp[us, tz=UTC]` | Exchange candle end; a default REST response has millisecond resolution |
| `quote_volume` | `float64` | Traded quote-asset amount, USDT |
| `trade_count` | `int64` | Number of trades reported by the exchange |
| `taker_buy_base_volume` | `float64` | BTC amount bought by takers |
| `taker_buy_quote_volume` | `float64` | USDT amount bought by takers |

Every canonical column is required. Null values fail the data-quality checks. Raw REST timestamps are interpreted explicitly as milliseconds. Microsecond storage resolution does not add source precision: for example, a source close time ending `.999` becomes `.999000`, not `.999999`. Column types are checked with ordinary pandas dtype checks.

Prices and amounts use float64 for analytical efficiency, not exact financial accounting. The original exchange decimal strings are retained in compressed raw payloads for reprocessing at decimal precision if required. Taker-volume comparison allows a tolerance of `max(1e-8, abs(total_volume) * 1e-12)` to avoid insignificant numerical roundoff.

The dataset does not contain bid/ask spread, order-book depth, individual trade ordering, execution latency or venue-wide consolidated market prices. A one-second candle cannot reveal the order in which its high and low occurred. Empty-trade candles and missing records are separate concepts.

## Quality output definitions

- **Expected seconds:** whole seconds in the requested `[start, end)` window.
- **Observed unique seconds:** distinct on-grid open times inside the window from trusted files.
- **Missing seconds:** expected minus observed unique seconds. An untrusted file leaves its entire window uncovered.
- **Known missing seconds:** missing observations inside the documented 2023-03-24 12:39:42–14:00:00 UTC interval, from a partition that passes file checks.
- **Unexpected missing seconds:** all other missing observations, including the entire window of a missing or untrusted partition.
- **Duplicate seconds:** extra rows after the first row at the same open timestamp; duplicates do not compensate for missing seconds.
- **Coverage fraction:** observed unique seconds divided by expected seconds.
- **Zero-volume seconds:** observed records with zero BTC volume, preserved as records rather than missing data.
- **Daily quality:** the same checks for each requested UTC-day window, including partial days.

The audit saves `quality.json`, `daily_quality.csv` and `gaps.csv`. Gap rows include `status` and `reason`; a gap crossing the known boundary is split so unexpected seconds remain visible. Actual coverage is never adjusted upward for the exception.

`PASS` means complete coverage and no errors. `WARNING` accepts only the documented source gap, keeps `ready_for_analysis=true`, and requires gap-aware analysis. Any other missing data, row violation or file issue remains `FAIL`; untrusted daily files retain `UNTRUSTED`. Rule counts can overlap on one row. No observations are imputed or rewritten.
