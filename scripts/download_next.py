"""
Download the next missing historical date, working backwards from yesterday.

Run daily via cron to systematically fill in historical audio gaps. Each
invocation downloads exactly one missing date, walking backwards in time.

Example progression:
  Day 1: yesterday = Apr 13 → missing → download Apr 13
  Day 2: yesterday = Apr 14 → downloaded → Apr 13 downloaded → Apr 12 downloaded → Apr 11 missing → download Apr 11
  ...

Usage:
  python scripts/download_next.py --city el_cerrito
  python scripts/download_next.py --city el_cerrito --earliest 2025-04-13
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import click


@click.command()
@click.option("--city", default="el_cerrito", show_default=True)
@click.option("--earliest", default="2025-04-13", show_default=True,
              help="Don't search for missing dates before this date")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--dry-run", is_flag=True, help="Print the date that would be downloaded, don't download")
def main(city, earliest, data_dir, dry_run):
    """Download the next missing historical date, working backwards from yesterday."""
    audio_root = Path(data_dir) / "audio" / city
    earliest_date = date.fromisoformat(earliest)

    downloaded = {
        date(int(d.name[:4]), int(d.name[4:6]), int(d.name[6:8]))
        for d in audio_root.iterdir()
        if d.is_dir() and len(d.name) == 8 and d.name.isdigit()
    } if audio_root.exists() else set()

    cursor = date.today() - timedelta(days=1)

    while cursor >= earliest_date:
        if cursor not in downloaded:
            click.echo(f"Next missing date: {cursor}")
            if dry_run:
                return
            from scripts.run_pipeline import main as run_pipeline
            from click.testing import CliRunner
            # Invoke via subprocess to get clean output and sleep inhibition
            import subprocess
            result = subprocess.run(
                [
                    sys.executable, "scripts/run_pipeline.py",
                    "--city", city,
                    "--start", str(cursor),
                    "--end", str(cursor),
                    "--steps", "download",
                    "--data-dir", data_dir,
                ],
                cwd=Path(data_dir).parent if data_dir != "data" else Path.cwd(),
            )
            sys.exit(result.returncode)
        cursor -= timedelta(days=1)

    click.echo("No missing dates found — historical archive is complete!")


if __name__ == "__main__":
    main()
