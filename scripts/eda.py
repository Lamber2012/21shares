"""Plot descriptive statistics from the full-run analysis tables."""

from pathlib import Path
import json
import calendar
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "reports/full/analysis/tables"
OUTPUT = ROOT / "reports/full/eda"
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "figures").mkdir(exist_ok=True)
(OUTPUT / "tables").mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200})
daily = pd.read_csv(SOURCE / "daily.csv", parse_dates=["date"])
daily["date"] = pd.to_datetime(daily["date"], utc=True)
complete = daily[daily["complete_day"]].copy()
years = sorted(daily["year"].unique())
footer = "Binance spot BTCUSDT | 2023-01-01 to 2026-11-06 UTC (end exclusive) | Descriptive analysis"


def save(fig, name):
    """Leave room for the source and save a clear PNG."""
    fig.text(0.01, 0.012, footer, color="#64748b", fontsize=9)
    fig.tight_layout(rect=[0, 0.045, 1, 0.96])
    fig.savefig(OUTPUT / "figures" / name, facecolor="white")
    plt.close(fig)


def heatmap(ax, values, title, cmap, labels, vmax=None):
    """Keep absent months blank and show the observed values."""
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("#edf0f4")
    image = ax.imshow(values, aspect="auto", cmap=colors, vmin=0, vmax=vmax)
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_yticks(range(len(years)), years)
    ax.set_xticks(range(len(labels)), labels)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                color = "white" if image.norm(value) > 0.60 else "#162b3c"
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=9, color=color)
    ax.figure.colorbar(image, ax=ax, fraction=0.022, pad=0.02)


def monthly_heatmaps():
    """Compare monthly typical volatility and observed zero-volume seconds."""
    rows = []
    for (year, month), group in daily.groupby([daily["date"].dt.year, daily["date"].dt.month]):
        full = group[group["complete_day"]]
        rows.append({"year": year, "month": month, "observed_days": len(group),
                     "complete_days": len(full), "median_daily_rv_pct": full["rv_pct"].median(),
                     "zero_volume_pct": 100 * group["zero_volume_seconds"].sum() / group["rows"].sum()})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "tables/monthly_eda.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    for ax, metric, title, cmap in zip(axes, ["median_daily_rv_pct", "zero_volume_pct"],
            ["Median daily realized volatility (%) - complete days", "Zero-volume seconds (%) - observed seconds"],
            ["YlOrRd", "YlGnBu"]):
        matrix = table.pivot(index="year", columns="month", values=metric).reindex(index=years, columns=range(1, 13))
        heatmap(ax, matrix.to_numpy(), title, cmap, [calendar.month_abbr[i] for i in range(1, 13)])
    fig.suptitle("Monthly data profile | Nov 2026 has 5 days; grey cells have no data", x=0.01, ha="left", fontsize=17, fontweight="bold")
    save(fig, "01_monthly_heatmaps.png")


def hourly_heatmaps():
    """Show equal-day-weighted hour shares for each year."""
    table = pd.read_csv(SOURCE / "hourly.csv")
    table = table[table["year"] != "All"].copy()
    table["year"] = table["year"].astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    vmax = max(table["volume_share_pct"].max(), table["variance_share_pct"].max())
    for ax, metric, title in zip(axes, ["volume_share_pct", "variance_share_pct"],
            ["BTC volume share by UTC hour (%)", "Realized variance share by UTC hour (%)"]):
        matrix = table.pivot(index="year", columns="hour_utc", values=metric).reindex(index=years, columns=range(24))
        heatmap(ax, matrix.to_numpy(), title, "YlOrRd", list(range(24)), vmax)
        ax.set_xlabel("UTC hour starting at")
    fig.suptitle("Intraday structure across years | Uniform-hour reference = 4.17%", x=0.01, ha="left", fontsize=17, fontweight="bold")
    save(fig, "02_hourly_heatmaps.png")


def calendar_heatmap():
    """Place each complete day's volatility on its calendar date."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), gridspec_kw={"hspace": 0.65})
    vmax = float(complete["rv_pct"].max())
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#edf0f4")
    for year, ax in zip(years, axes):
        first = pd.Timestamp(f"{year}-01-01", tz="UTC")
        left = first - pd.Timedelta(days=first.weekday())
        last = pd.Timestamp(f"{year}-12-31", tz="UTC")
        weeks = int((last - left).days // 7 + 1)
        matrix = np.full((7, weeks), np.nan)
        group = complete[complete["year"] == year]
        for row in group.itertuples(index=False):
            matrix[row.date.weekday(), (row.date - left).days // 7] = row.rv_pct
        image = ax.imshow(matrix, cmap=cmap, norm=Normalize(0, vmax), aspect="auto")
        ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=9)
        positions = []
        for month in range(1, 13):
            positions.append((pd.Timestamp(year=int(year), month=month, day=1, tz="UTC") - left).days // 7)
        ax.set_xticks(positions, list(calendar.month_abbr)[1:], fontsize=10)
        ax.set_title(f"{year} | {len(group):,} complete days", loc="left", fontsize=12, fontweight="bold")
    fig.suptitle("Daily realized volatility (%) | Grey = unavailable, incomplete or outside year", x=0.01, ha="left", fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.91, top=0.92, bottom=0.065)
    cax = fig.add_axes([0.935, 0.16, 0.012, 0.65])
    fig.colorbar(image, cax=cax, label="Daily RV (%)")
    fig.text(0.01, 0.015, footer, color="#64748b", fontsize=9)
    fig.savefig(OUTPUT / "figures/03_volatility_calendar.png", facecolor="white")
    plt.close(fig)


def distributions():
    """Describe complete-day distributions without treating days as independent tests."""
    specs = [("turnover_usdt", "Daily turnover (USDT billion)", 1e9, "#167c94"),
             ("rv_pct", "Daily realized volatility (%)", 1, "#b76432"),
             ("zero_volume_pct", "Zero-volume seconds per day (%)", 1, "#478a75"),
             ("trades", "Daily trades (million)", 1e6, "#667ab2")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    summary = []
    for ax, (field, label, unit, color) in zip(axes.ravel(), specs):
        values = complete[field] / unit
        ax.hist(values, bins=45, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.axvline(values.median(), color="#172b3a", linestyle="--", linewidth=1.5, label=f"Median {values.median():.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Complete days")
        ax.legend(frameon=False, fontsize=10)
        ax.grid(axis="y", alpha=0.14)
        row = {"metric": field, "days": len(values)}
        for q in [0, .01, .25, .5, .75, .95, .99, 1]:
            row[f"p{int(100*q):02d}"] = float(complete[field].quantile(q))
        row["mean"] = float(complete[field].mean())
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUTPUT / "tables/daily_distribution.csv", index=False)
    fig.suptitle("Daily distributions | 1,404 complete UTC days; all extremes retained", x=0.01, ha="left", fontsize=17, fontweight="bold")
    save(fig, "04_daily_distributions.png")


def activity_scatter():
    """Plot same-day turnover and volatility, with separate year labels."""
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#167c94", "#b76432", "#478a75", "#667ab2"]
    rows = []
    for year, color in zip(years, colors):
        group = complete[complete["year"] == year]
        ax.scatter(group["turnover_usdt"] / 1e9, group["rv_pct"], s=19, alpha=0.48, color=color, label=f"{year} (n={len(group)})")
        rows.append({"year": int(year), "complete_days": len(group),
                     "pearson_log_turnover_log_rv": float(np.corrcoef(np.log(group["turnover_usdt"]), np.log(group["rv_pct"]))[0, 1])})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Daily USDT turnover (billion, log scale)")
    ax.set_ylabel("Daily realized volatility (%, log scale)")
    ax.set_title("Same-day activity and volatility | Association only; no future target", loc="left", fontsize=17, fontweight="bold", pad=18)
    ax.grid(alpha=0.15, which="both")
    ax.legend(title="Year", frameon=False)
    pd.DataFrame(rows).to_csv(OUTPUT / "tables/same_day_association.csv", index=False)
    save(fig, "05_activity_volatility_scatter.png")


if __name__ == "__main__":
    monthly_heatmaps()
    hourly_heatmaps()
    calendar_heatmap()
    distributions()
    activity_scatter()
    summary = {"source": "Existing full analysis CSV tables", "complete_days": len(complete),
               "figure_count": 5, "table_count": 3, "raw_data_changed": False,
               "notes": ["Descriptive summaries only; no new predictive claims.",
                         "Monthly zero-volume rates include all observed seconds.",
                         "Distributions, calendar RV and scatter use complete UTC days.",
                         "Missing future months remain blank; 2026 is partial."]}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
