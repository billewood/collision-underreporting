"""
Download Broadcastify archive MP3 blocks for a given feed and date range.
Adapted from github.com/NotJoeMartinez/broadcastify-cli.

Usage:
    python -m broadcastify.download --city el_cerrito --start 2025-10-01 --end 2025-10-07

Output:
    data/audio/{city}/{YYYYMMDD}/{archive_id}.mp3
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import click
import requests
import yaml

from broadcastify.auth import auth_headers, get_cookie

ARCHIVE_LIST_URL = "https://www.broadcastify.com/archives/ajax.php"
ARCHIVE_DOWNLOAD_URL = "https://www.broadcastify.com/archives/downloadv2"


def _load_city_config(city: str) -> dict:
    cfg_path = Path("config/cities.yaml")
    cities = yaml.safe_load(cfg_path.read_text())
    if city not in cities:
        raise ValueError(f"Unknown city '{city}'. Available: {list(cities.keys())}")
    cfg = cities[city]
    if not cfg.get("broadcastify_feed_id"):
        raise ValueError(f"No broadcastify_feed_id configured for '{city}'.")
    return cfg


def _list_archive_ids(feed_id: int, dt: date, headers: dict) -> list[str]:
    """Return list of archive IDs available for a feed on a given date."""
    date_str = dt.strftime("%-m/%-d/%Y")  # M/D/YYYY — no leading zeros
    resp = requests.get(
        ARCHIVE_LIST_URL,
        params={"feedId": feed_id, "date": date_str},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("data"):
        return []
    return [str(row[0]) for row in data["data"]]


def _download_block(
    feed_id: int,
    archive_id: str,
    dt: date,
    out_dir: Path,
    headers: dict,
) -> Path | None:
    """Download a single 30-minute archive block. Returns output path or None on skip."""
    url_date = dt.strftime("%Y%m%d")
    url = f"{ARCHIVE_DOWNLOAD_URL}/{feed_id}/{url_date}/{archive_id}"

    # Filename mirrors broadcastify-cli convention: {url_date}-{archive_id}-{feed_id}.mp3
    filename = f"{url_date}-{archive_id}-{feed_id}.mp3"
    out_path = out_dir / filename

    if out_path.exists():
        return out_path  # already downloaded

    resp = requests.get(url, headers=headers, timeout=60, stream=True)
    if resp.status_code == 401:
        raise PermissionError("Auth failed — re-run with --relogin to refresh cookie.")
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    return out_path


def download_range(
    city: str,
    start: date,
    end: date,
    jobs: int = 4,
    relogin: bool = False,
    data_dir: Path = Path("data"),
) -> list[Path]:
    """
    Download all archive blocks for a city between start and end (inclusive).
    Returns list of downloaded file paths.
    """
    cfg = _load_city_config(city)
    feed_id = cfg["broadcastify_feed_id"]
    cookie = get_cookie(force_login=relogin)
    headers = auth_headers(cookie)

    downloaded = []
    current = start
    while current <= end:
        date_dir = data_dir / "audio" / city / current.strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        archive_ids = _list_archive_ids(feed_id, current, headers)
        if not archive_ids:
            click.echo(f"  {current}: no archives found")
            current += timedelta(days=1)
            continue

        click.echo(f"  {current}: {len(archive_ids)} blocks")

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    _download_block, feed_id, aid, current, date_dir, headers
                ): aid
                for aid in archive_ids
            }
            for future in as_completed(futures):
                aid = futures[future]
                try:
                    path = future.result()
                    if path:
                        downloaded.append(path)
                        click.echo(f"    ✓ {path.name}")
                except Exception as exc:
                    click.echo(f"    ✗ {aid}: {exc}", err=True)

        time.sleep(0.5)  # polite pause between days
        current += timedelta(days=1)

    return downloaded


@click.command()
@click.option("--city", required=True, help="City key from config/cities.yaml")
@click.option("--start", "start_str", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_str", required=True, help="End date YYYY-MM-DD (inclusive)")
@click.option("--jobs", default=4, show_default=True, help="Parallel download threads")
@click.option("--relogin", is_flag=True, default=False, help="Force re-authentication")
@click.option(
    "--data-dir", default="data", show_default=True, help="Root data directory"
)
def main(city, start_str, end_str, jobs, relogin, data_dir):
    """Download Broadcastify archive blocks for a city and date range."""
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    click.echo(f"Downloading {city} | {start} → {end} | {jobs} threads")
    paths = download_range(city, start, end, jobs=jobs, relogin=relogin, data_dir=Path(data_dir))
    click.echo(f"\nDone. {len(paths)} blocks downloaded.")


if __name__ == "__main__":
    main()
