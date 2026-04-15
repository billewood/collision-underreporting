"""
Download as many missing historical dates as possible within the daily
Broadcastify quota (499 files per 24-hour period).

Working backwards from yesterday, downloads complete dates until the quota
is exhausted. Already-downloaded files are skipped automatically, so the
script is safe to re-run mid-quota-period.

Usage:
  python scripts/download_next.py
  python scripts/download_next.py --city el_cerrito --quota 499 --dry-run
"""

import sys
from datetime import date, timedelta
from pathlib import Path
import subprocess

import click

BROADCASTIFY_DAILY_QUOTA = 499


@click.command()
@click.option("--city", default="el_cerrito", show_default=True)
@click.option("--earliest", default="2025-04-13", show_default=True,
              help="Don't search for missing dates before this date")
@click.option("--quota", default=BROADCASTIFY_DAILY_QUOTA, show_default=True,
              help="Max files to download this run")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--dry-run", is_flag=True, help="Print what would be downloaded without downloading")
def main(city, earliest, quota, data_dir, dry_run):
    """Download missing historical dates up to the daily Broadcastify quota."""
    audio_root = Path(data_dir) / "audio" / city
    earliest_date = date.fromisoformat(earliest)

    downloaded_dates = {
        date(int(d.name[:4]), int(d.name[4:6]), int(d.name[6:8]))
        for d in audio_root.iterdir()
        if d.is_dir() and len(d.name) == 8 and d.name.isdigit()
    } if audio_root.exists() else set()

    cursor = date.today() - timedelta(days=1)
    files_this_run = 0
    dates_queued = []

    # Collect missing dates until we estimate we'd hit the quota
    while cursor >= earliest_date:
        if cursor not in downloaded_dates:
            dates_queued.append(cursor)
        cursor -= timedelta(days=1)

    if not dates_queued:
        click.echo("No missing dates found — historical archive is complete!")
        return

    if dry_run:
        click.echo(f"Missing dates (most recent first): {len(dates_queued)} total")
        for d in dates_queued[:15]:
            click.echo(f"  {d}")
        if len(dates_queued) > 15:
            click.echo(f"  ... and {len(dates_queued) - 15} more")
        return

    click.echo(f"Quota: {quota} files | {len(dates_queued)} missing dates to work through")

    for target_date in dates_queued:
        if files_this_run >= quota:
            click.echo(f"\nQuota reached ({files_this_run}/{quota} files). Stopping.")
            break

        date_dir = audio_root / target_date.strftime("%Y%m%d")
        files_before = set(date_dir.glob("*.mp3")) if date_dir.exists() else set()

        click.echo(f"\nDownloading {target_date} ({files_this_run}/{quota} files used so far)...")
        result = subprocess.run(
            [
                sys.executable, "scripts/run_pipeline.py",
                "--city", city,
                "--start", str(target_date),
                "--end", str(target_date),
                "--steps", "download",
                "--data-dir", data_dir,
            ],
            cwd=Path(__file__).parent.parent,
        )

        files_after = set(date_dir.glob("*.mp3")) if date_dir.exists() else set()
        new_files = len(files_after - files_before)
        files_this_run += new_files

        if result.returncode != 0:
            click.echo(f"Download failed for {target_date} (exit {result.returncode}) — stopping.")
            break

        click.echo(f"  {new_files} files downloaded | running total: {files_this_run}/{quota}")

    click.echo(f"\nDone. {files_this_run} files downloaded this run.")


if __name__ == "__main__":
    main()
