"""
Download Broadcastify archive MP3 blocks for a given feed and date range.
Adapted from github.com/NotJoeMartinez/broadcastify-cli.

Usage:
    python -m broadcastify.download --city el_cerrito --start 2025-10-01 --end 2025-10-07

Output:
    data/audio/{city}/{YYYYMMDD}/{archive_id}.mp3
"""

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import click
import requests
import yaml

from broadcastify.auth import auth_headers, get_cookie

_SENTINEL = None  # signals end of download stream

ARCHIVE_LIST_URL = "https://www.broadcastify.com/archives/ajax.php"
ARCHIVE_DOWNLOAD_URL = "https://www.broadcastify.com/archives/downloadv2"

# Seconds to wait between individual block downloads (avoids 429s)
INTER_REQUEST_DELAY = 1.5
# Retry settings for 429 / transient errors
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 10  # seconds — doubles each retry (10, 20, 40, 80, 160)


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
    """Download a single 30-minute archive block with retry on 429. Returns output path."""
    url_date = dt.strftime("%Y%m%d")
    url = f"{ARCHIVE_DOWNLOAD_URL}/{feed_id}/{url_date}/{archive_id}"

    filename = f"{url_date}-{archive_id}-{feed_id}.mp3"
    out_path = out_dir / filename

    if out_path.exists():
        return out_path  # already downloaded

    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=headers, timeout=60, stream=True)

        if resp.status_code == 401:
            raise PermissionError("Auth failed — re-run with --relogin to refresh cookie.")

        if resp.status_code == 429:
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            click.echo(f"    429 rate-limited — waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}", err=True)
            time.sleep(wait)
            continue

        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return out_path

    raise RuntimeError(f"Failed to download {archive_id} after {MAX_RETRIES} retries (persistent 429)")


def download_iter(
    city: str,
    dt: date,
    headers: dict,
    feed_id: int,
    data_dir: Path,
) -> "queue.Queue[Path | None]":
    """
    Download all blocks for a single day in a background thread, yielding
    each path via a Queue as soon as it's ready. Caller reads until it
    receives the _SENTINEL (None).

    This lets the main thread start transcribing block N while block N+1
    is still downloading, hiding the inter-request delay inside GPU time.
    """
    q: queue.Queue = queue.Queue()

    def _worker():
        date_dir = data_dir / "audio" / city / dt.strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        archive_ids = _list_archive_ids(feed_id, dt, headers)
        if not archive_ids:
            click.echo(f"  {dt}: no archives found")
            q.put(_SENTINEL)
            return

        already = [
            aid for aid in archive_ids
            if (date_dir / f"{dt.strftime('%Y%m%d')}-{aid}-{feed_id}.mp3").exists()
        ]
        todo = [aid for aid in archive_ids if aid not in already]
        click.echo(
            f"  {dt}: {len(archive_ids)} blocks "
            f"({len(already)} cached, {len(todo)} to download)"
        )

        # Cached blocks go straight into the queue so they can be transcribed
        # immediately while new ones are fetched
        for aid in already:
            q.put(date_dir / f"{dt.strftime('%Y%m%d')}-{aid}-{feed_id}.mp3")

        for aid in todo:
            try:
                path = _download_block(feed_id, aid, dt, date_dir, headers)
                if path:
                    click.echo(f"    ✓ {path.name}")
                    q.put(path)
            except Exception as exc:
                click.echo(f"    ✗ {aid}: {exc}", err=True)
            time.sleep(INTER_REQUEST_DELAY)

        q.put(_SENTINEL)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return q


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

        already = sum(
            1 for aid in archive_ids
            if (date_dir / f"{current.strftime('%Y%m%d')}-{aid}-{feed_id}.mp3").exists()
        )
        todo = [aid for aid in archive_ids
                if not (date_dir / f"{current.strftime('%Y%m%d')}-{aid}-{feed_id}.mp3").exists()]
        click.echo(f"  {current}: {len(archive_ids)} blocks ({already} cached, {len(todo)} to download)")

        if jobs == 1:
            # Sequential with per-request delay — safest for rate limits
            for aid in todo:
                try:
                    path = _download_block(feed_id, aid, current, date_dir, headers)
                    if path:
                        downloaded.append(path)
                        click.echo(f"    ✓ {path.name}")
                except Exception as exc:
                    click.echo(f"    ✗ {aid}: {exc}", err=True)
                time.sleep(INTER_REQUEST_DELAY)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {
                    pool.submit(
                        _download_block, feed_id, aid, current, date_dir, headers
                    ): aid
                    for aid in todo
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

        # Add already-cached files to the returned list
        for aid in archive_ids:
            p = date_dir / f"{current.strftime('%Y%m%d')}-{aid}-{feed_id}.mp3"
            if p.exists() and p not in downloaded:
                downloaded.append(p)

        time.sleep(2)  # pause between days
        current += timedelta(days=1)

    return downloaded


@click.command()
@click.option("--city", required=True, help="City key from config/cities.yaml")
@click.option("--start", "start_str", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_str", required=True, help="End date YYYY-MM-DD (inclusive)")
@click.option("--jobs", default=1, show_default=True, help="Parallel download threads (1 = sequential, safest for rate limits)")
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
