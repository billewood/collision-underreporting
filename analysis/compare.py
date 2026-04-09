"""
Compare dispatch incidents against SWITRS records to quantify underreporting.

Loads extracted dispatch incidents and SWITRS collision records for a city
and date range, aggregates by month, and outputs a gap ratio table plus
a time-series chart.

Usage:
    python -m analysis.compare --city el_cerrito --start 2025-10-01 --end 2026-01-01
"""

import json
from datetime import date
from pathlib import Path

import click
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_dispatch_incidents(
    city: str,
    start: date | None,
    end: date | None,
    data_dir: Path,
    min_confidence: float = 0.5,
) -> pd.DataFrame:
    """Load extracted incident JSONL files for a city."""
    incidents_dir = data_dir / "incidents" / city
    if not incidents_dir.exists():
        return pd.DataFrame()

    rows = []
    for jsonl_path in sorted(incidents_dir.glob("*.jsonl")):
        for line in jsonl_path.read_text().splitlines():
            if not line:
                continue
            try:
                inc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if inc.get("incident_type") == "parse_error":
                continue
            if inc.get("confidence", 1.0) < min_confidence:
                continue
            rows.append(inc)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["block_start_utc"], errors="coerce", utc=True).dt.date

    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]

    return df


def load_switrs_incidents(
    city: str,
    start: date | None,
    end: date | None,
    data_dir: Path,
) -> pd.DataFrame:
    from switrs.pull import load_city_switrs
    return load_city_switrs(city, start=start, end=end, data_dir=data_dir)


def aggregate_by_month(df: pd.DataFrame, date_col: str, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "count", "source"])
    df = df.copy()
    df["month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M")
    counts = df.groupby("month").size().reset_index(name="count")
    counts["source"] = source
    return counts


def build_comparison_table(
    dispatch_df: pd.DataFrame,
    switrs_df: pd.DataFrame,
) -> pd.DataFrame:
    dispatch_monthly = aggregate_by_month(dispatch_df, "date", "dispatch")
    switrs_monthly = aggregate_by_month(switrs_df, "collision_date", "switrs")

    dispatch_monthly = dispatch_monthly.rename(columns={"count": "dispatch_incidents"})
    switrs_monthly = switrs_monthly.rename(columns={"count": "switrs_reports"})

    merged = pd.merge(
        dispatch_monthly[["month", "dispatch_incidents"]],
        switrs_monthly[["month", "switrs_reports"]],
        on="month",
        how="outer",
    ).fillna(0)

    merged["dispatch_incidents"] = merged["dispatch_incidents"].astype(int)
    merged["switrs_reports"] = merged["switrs_reports"].astype(int)
    merged["gap_ratio"] = merged.apply(
        lambda r: round(r["dispatch_incidents"] / r["switrs_reports"], 2)
        if r["switrs_reports"] > 0
        else float("inf"),
        axis=1,
    )
    return merged.sort_values("month")


def print_summary(table: pd.DataFrame, city: str) -> None:
    click.echo(f"\n=== Collision Underreporting Analysis: {city} ===\n")
    click.echo(
        f"{'Month':<12} {'Dispatch':>10} {'SWITRS':>10} {'Gap Ratio':>12}"
    )
    click.echo("-" * 46)
    for _, row in table.iterrows():
        gap = f"{row['gap_ratio']:.2f}x" if row["gap_ratio"] != float("inf") else "∞"
        click.echo(
            f"{str(row['month']):<12} {int(row['dispatch_incidents']):>10} "
            f"{int(row['switrs_reports']):>10} {gap:>12}"
        )

    total_dispatch = table["dispatch_incidents"].sum()
    total_switrs = table["switrs_reports"].sum()
    overall_gap = (
        round(total_dispatch / total_switrs, 2) if total_switrs > 0 else float("inf")
    )
    click.echo("-" * 46)
    gap_str = f"{overall_gap:.2f}x" if overall_gap != float("inf") else "∞"
    click.echo(
        f"{'TOTAL':<12} {int(total_dispatch):>10} {int(total_switrs):>10} {gap_str:>12}"
    )
    click.echo()


def plot_comparison(table: pd.DataFrame, city: str, out_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        click.echo("matplotlib not installed — skipping chart")
        return

    months = [str(m) for m in table["month"]]
    x = range(len(months))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Collision Underreporting: {city}", fontsize=14)

    ax1.bar([i - 0.2 for i in x], table["dispatch_incidents"], 0.4,
            label="Dispatch (EMS radio)", color="#e74c3c", alpha=0.85)
    ax1.bar([i + 0.2 for i in x], table["switrs_reports"], 0.4,
            label="SWITRS (official)", color="#3498db", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha="right")
    ax1.set_ylabel("Incidents")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    gap_finite = [g if g != float("inf") else 0 for g in table["gap_ratio"]]
    ax2.plot(x, gap_finite, marker="o", color="#e74c3c", linewidth=2)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="1:1 (no gap)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, rotation=45, ha="right")
    ax2.set_ylabel("Gap ratio (dispatch / SWITRS)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    click.echo(f"Chart saved to {out_path}")


@click.command()
@click.option("--city", required=True)
@click.option("--start", "start_str", default=None, help="YYYY-MM-DD")
@click.option("--end", "end_str", default=None, help="YYYY-MM-DD")
@click.option("--min-confidence", default=0.5, show_default=True,
              help="Minimum incident confidence score to include")
@click.option("--chart", is_flag=True, default=False, help="Save a time-series chart")
@click.option("--data-dir", default="data", show_default=True)
def main(city, start_str, end_str, min_confidence, chart, data_dir):
    """Compare dispatch incidents vs SWITRS records and report the gap."""
    start = date.fromisoformat(start_str) if start_str else None
    end = date.fromisoformat(end_str) if end_str else None
    data = Path(data_dir)

    dispatch_df = load_dispatch_incidents(city, start, end, data, min_confidence)
    switrs_df = load_switrs_incidents(city, start, end, data)

    if dispatch_df.empty:
        click.echo("No dispatch incidents found. Run extract first.")
        return
    if switrs_df.empty:
        click.echo(
            "No SWITRS data found. Export from tims.berkeley.edu and place in "
            f"data/switrs/{city}/\n"
            "Showing dispatch-only summary:"
        )

    table = build_comparison_table(dispatch_df, switrs_df)
    print_summary(table, city)

    if chart:
        out = data / "analysis" / f"{city}_gap.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(table, city, out)


if __name__ == "__main__":
    main()
