# BTC/USDT Trading Patterns

Research for option B of the 21Shares take-home project: identify trading patterns in Binance spot BTC/USDT one-second candles from January 2023.

The pipeline downloads public API data, stores daily Parquet files, checks data quality and studies activity, volatility and return relationships. The sample ends on **5 November 2026**, with `2026-11-06T00:00:00Z` as the exclusive endpoint.

- [Presentation](presentation/BTC-21shares.pdf)
- [Analysis tables](results/analysis/tables) and [figures](results/analysis/figures)
- [Descriptive tables](results/eda/tables) and [figures](results/eda/figures)
- [Quality results](results/quality/quality.json)
- [Data dictionary](docs/DATA_DICTIONARY.md)

## Main findings

| Finding | Full-sample result |
|---|---|
| Intraday concentration | 14:00-16:00 UTC averages 13.91% of daily BTC volume and 16.66% of daily variance |
| Regional overlap | London-New York overlap turnover and variance per minute average 1.60 and 1.89 times their own day's average |
| Weekend effect | Median within-week weekend/weekday ratios: 0.463 for BTC volume and 0.548 for daily realised volatility |
| Volatility persistence | One-minute absolute-return lag-one correlation: 0.383; signed-return correlation is close to zero |
| Trade imbalance | Correlation with same-minute returns: 0.421; with next-minute returns: 0.0125 |

Overlap activity, absolute-return persistence and lower regional weekend turnover receive support across all three chronological periods under both block-length settings. The next-minute imbalance association does not pass the longer-block adjusted test in 2026. Simple return forecasts show no robust MSE improvement in that period.

The final period is **retrospective**: the 2026 data was inspected before the validation design was finalised. These results describe market patterns and limited forecasting evidence; they do not establish trading profitability or causality.

![Chronological pattern comparisons](results/analysis/figures/14_pattern_stability.png)

## Repository structure

| Path | Responsibility |
|---|---|
| `run.py` | Configuration, command-line entry and task selection |
| `download.py` | Public API requests, pagination, raw cache, daily Parquet and download recovery |
| `quality.py` | File integrity, timestamp coverage and OHLCV checks |
| `analysis.py` | Aggregation, pattern research, chronological validation and charts |
| `scripts/eda.py` | Descriptive charts from the existing full-run summary tables |
| `tests/test_run.py` | Offline tests for downloads, quality checks and analysis calculations |
| `configs/` | One-day demo and full-sample configuration |
| `results/` | Saved quality checks, aggregate tables and research figures |
| `presentation/` | Final presentation |

Raw data, the virtual environment, logs and generated working outputs are excluded from Git. `results/` contains the saved full-run outputs; commands write new outputs to `reports/`.

## Setup and commands

Use Python 3.9-3.12. The project was tested on Windows with Python 3.9.21.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Activate the environment before running the following commands, or use its full Python path. On macOS/Linux, the interpreter is `.venv/bin/python`.

```powershell
# Inspect the one-day request plan without making API calls.
python run.py --config configs/demo.json plan

# Download and check the one-day sample.
python run.py --config configs/demo.json run

# Download and check the full sample.
python run.py --config configs/full.json run

# Recheck existing files without downloading.
python run.py --config configs/full.json check

# Run research after the quality checks pass.
python analysis.py --config configs/full.json

# Rebuild descriptive charts from the full analysis tables.
python scripts/eda.py

# Run the offline tests.
python -m pytest -q
```

`run` performs `fetch` followed by `check`. `fetch` can also run separately. Repeating the command resumes cached pages and reuses completed partitions whose hashes match their receipts. Override dates and directories with `--start`, `--end`, `--data-dir` and `--report-dir`. Analysis also accepts `--output-dir`.

The full range needs at least 122,235 API requests. Default pacing alone takes about 8.49 hours, before latency, retries and processing. A one-day demo checks the setup but cannot establish multi-year patterns.

To rebuild only the descriptive charts from committed results, copy `results/analysis/tables/` to `reports/full/analysis/tables/`, then run `python scripts/eda.py`. This script uses the fixed full-sample dates shown in the presentation.

## Data source and quality

The source is Binance spot `/api/v3/klines`, symbol `BTCUSDT`, interval `1s`, through `https://data-api.binance.vision`. No API key is required. Windows use UTC `[start, end)` boundaries. REST timestamps are parsed as milliseconds. The downloader does not ingest historical ZIP files, whose timestamp convention changed in 2025.

One day of seconds is processed at a time. Compressed raw responses are cached before conversion. Daily Parquet receipts record the source, window and SHA-256 hash. Checks cover integrity, schema, duplicate/off-grid/out-of-order timestamps, missing seconds, null/non-finite values, price and volume bounds, OHLC ordering, close times and taker-volume consistency.

| Coverage metric | Full sample |
|---|---:|
| Trusted daily partitions | 1,405 |
| Expected seconds | 121,392,000 |
| Observed seconds | 121,387,182 |
| Actual coverage | 99.996031% |
| Known missing seconds | 4,818 |
| Unexpected missing seconds | 0 |
| Observed zero-volume seconds | 8,833,704 |
| Complete analysis days | 1,404 |
| Complete comparison weeks | 199 |

The saved audit has status `WARNING`. All row checks and file-integrity checks pass.

### Known source gap

The accepted gap is exactly **`[2023-03-24T12:39:42Z, 2023-03-24T14:00:00Z)`**, or 4,818 seconds. The original cached response contains the timestamp jump within a single page. A repeated API request and the official daily archive showed the same missing interval.

Binance's [incident explanation](https://www.binance.com/en/blog/from-our-ceo/6789340645608890113) describes the spot trading halt; its [recovery announcement](https://www.binance.com/en/support/announcement/detail/813a31506e9f478ea8c1058b425df87a) gives 14:00 UTC as the resumption time. The candle gap starts later: 4,338 zero-volume, zero-trade candles remain from 11:27:24 through 12:39:41. The source does not explain this difference.

The [daily ZIP](https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2023-03-24.zip), verified against its [checksum](https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2023-03-24.zip.CHECKSUM), has the same unique timestamps and 36 additional identical duplicate rows. The project retains its REST dataset.

| Status | Rule |
|---|---|
| `PASS` | Complete requested coverage and no file or row errors |
| `WARNING` | Missing seconds are confined to the documented interval; all other checks pass |
| `FAIL` | Any other missing seconds, row violation, missing file or untrusted file |

The exception never excuses a corrupt or missing partition. Actual coverage remains unchanged. Observed zero-volume records are retained, missing seconds stay missing, and returns never cross a gap. Analysis stops on `FAIL` and handles the known `WARNING` explicitly.

## Research methods

- **Returns and volatility:** one-second log returns require adjacent timestamps. Daily realised volatility is `100 * sqrt(sum(r_second ** 2))`, without annualisation. Minute returns require two complete adjacent minutes. The first unavailable return is excluded.
- **Intraday seasonality:** normalise each complete UTC day separately, then average hourly volume and variance shares with equal day weights. The halt date is excluded.
- **Regional windows:** Hong Kong, London and New York use local 09:00-17:00 with historical timezone and DST rules. Regional windows can overlap; exclusive buckets partition the day. Intensity divides activity per minute by its own day's average per minute, then averages across days. Labels identify time windows rather than trader locations. DST profiles compare different seasonal samples and do not identify a causal effect.
- **Weekend effect:** within a complete Monday-Sunday week, compare weekend and weekday daily averages. Session comparisons use per-minute rates and require the bucket to exist on all seven days.
- **Autocorrelation:** signed returns measure direction persistence; absolute returns measure movement-size persistence. Minute pairs stay within continuous blocks. Second-level pairs stay within a UTC day.
- **Trade imbalance:** `2 * taker_buy_quote_volume / quote_volume - 1`. Zero-turnover minutes have undefined imbalance. Same-minute association is reported separately from forward association.
- **Descriptive checks:** monthly heatmaps, complete-day distributions and same-day turnover/RV scatter plots describe the data. Extreme-day sensitivity excludes dates above the full-sample 99th percentile of daily RV; this threshold is used only for descriptive comparisons.

The main analysis holds one day's seconds at a time and retains minute aggregates in memory. It writes 22 CSV tables, 14 PNG figures and a completion summary. The descriptive script adds three CSV tables and five PNG figures. Monthly/yearly tables report actual coverage: 2026 is partial and November contains five days.

## Chronological validation

| Stage | Period | Use |
|---|---|---|
| Training | 2023-2024 | Fit feature scales, intercepts and slopes |
| Validation | 2025 | Select the lowest-MSE candidate for each horizon |
| Retrospective test | 2026 to the data endpoint | Evaluate the selected forecast with frozen training parameters |

Forecast horizons are 1, 5 and 15 minutes. Candidates are the training-mean baseline, a return-only linear model and an imbalance-only linear model. All candidates use the same valid feature-target rows. Features come from completed minutes; targets cannot extend past their stage boundary or across a gap. Full-day normalisation and full-sample thresholds are excluded from forecast inputs.

Forecast improvement is `100 * (1 - model MSE / baseline MSE)` percent. It measures prediction error improvement, not a trading return. Training and validation inference is exploratory because those periods are used for fitting and selection.

### Dependence and multiple comparisons

Daily sufficient statistics preserve paired observations and overlapping targets. Inference resamples contiguous calendar blocks within each stage, retaining missing units in their calendar positions.

| Setting | Value |
|---|---|
| Daily block lengths | 7 and 28 days |
| Weekly block lengths | 4 and 8 weeks |
| Bootstrap | Circular blocks; 1,999 draws; seed 42 |
| Intervals and p-values | Basic bootstrap intervals; approximate centred two-sided p-values with plus-one correction |
| Minimum support | At least four block lengths of usable units and 95% valid draws |
| Adjustment | Holm across 39 planned checks within each stage and block setting |

The family contains 20 correlation checks, eight regional/overlap intensity checks, eight weekend checks and three selected-forecast checks. Undefined checks retain their place in the family. The saved `inference.csv` contains 234 rows: 39 checks across three stages and two block settings. An `ok` status means the calculation was valid. Statistical support additionally requires adjusted p-values below 0.05 with consistent effect direction under both block settings.

## Saved outputs

| Location | Contents |
|---|---|
| `results/quality/` | Overall audit, daily checks and missing intervals |
| `results/analysis/tables/` | Activity, hourly shares, weekend ratios, correlations, regional windows and validation |
| `results/analysis/figures/` | Research and validation charts |
| `results/analysis/summary.json` | Run timestamp, source quality, usable counts and calculation settings |
| `results/eda/` | Heatmaps, volatility calendar, distributions and same-day activity/volatility plots |

`forecast_models.csv` records training parameters. `forecast_scores.csv` retains every candidate's scores and selection status. `feature_audit.csv` records information timing and target boundaries. `correlation_daily.csv` and `forecast_daily.csv` retain daily inputs to inference.

## Recovery and limitations

After interruption, rerun the same download command. If a hard kill leaves `data/<run>/.writer.lock`, check that its PID is no longer running before removing that file. A hash mismatch stops processing; preserve the file and rebuild from trusted cached pages.

Exit codes: 0 for success or accepted `WARNING`, 1 for runtime/configuration errors, 2 for quality `FAIL` or invalid CLI syntax, and 130 for interruption.

The data comes from one exchange and contains no order-book depth, spread, trader location or within-candle trade ordering. Float64 supports analysis; original decimal strings remain in raw caches. Intraday seasonality has not been fully removed from volatility-persistence estimates. Forward volume-to-volatility prediction and event recovery paths remain future research. An untouched evaluation requires freezing the method before collecting new data.

Source documentation: [Binance REST API](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md) and [public market-data endpoints](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md). Source data remains subject to the provider's terms.