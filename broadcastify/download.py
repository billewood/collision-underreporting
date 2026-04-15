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

ARCHIVE_LIST_URL = "https://www.broadcastify.com/archives/api/archives.php"
ARCHIVE_DOWNLOAD_URL = "https://www.broadcastify.com/archives/download"

# Seconds to wait between individual block downloads (avoids 429s)
INTER_REQUEST_DELAY = 3.0
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
    # New API returns {"archives": [{"id": "...", "start": "...", ...}, ...]}
    archives = data.get("archives", [])
    if archives:
        return [str(a["id"]) for a in archives]
    # Fallback: old API returned {"data": [[id, ...], ...]}
    if data.get("data"):
        return [str(row[0]) for row in data["data"]]
    return []


def _download_block(
    feed_id: int,
    archive_id: str,
    dt: date,
    out_dir: Path,
    headers: dict,
) -> Path | None:
    """Download a single 30-minute archive block with retry on 429. Returns output path."""
    url_date = dt.strftime("%Y%m%d")
    url = f"{ARCHIVE_DOWNLOAD_URL}/{archive_id}"

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
    max_blocks: int | None = None,
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
        if max_blocks is not None:
            todo = todo[:max(0, max_blocks - len(already))]

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
                    click.echo(f"    OK {path.name}")
                    q.put(path)
            except Exception as exc:
                click.echo(f"    FAIL {aid}: {exc}", err=True)
            time.sleep(INTER_REQUEST_DELAY)

        q.put(_SENTINEL)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return q


def download_range(
    city: str,
    start: date,
    end: date,
    jobs: int = 1,
    relogin: bool = False,
    data_dir: Path = Path("data"),
    max_blocks: int | None = None,
) -> list[Path]:
    """
    Download archive blocks for a city between start and end (inclusive).
    max_blocks: if set, stop after downloading this many blocks total (useful for probing).
    Returns list of downloaded file paths.
    """
    cfg = _load_city_config(city)
    feed_id = cfg["broadcastify_feed_id"]
    cookie = get_cookie(force_login=relogin)
    headers = auth_headers(cookie)

    downloaded = []
    current = start
    while current <= end:
        if max_blocks is not None and len(downloaded) >= max_blocks:
            break

        date_dir = data_dir / "audio" / city / current.strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        archive_ids = _list_archive_ids(feed_id, current, headers)
        if not archive_ids:
            click.echo(f"  {current}: no archives found")
            current += timedelta(days=1)
            continue

        already = [
            aid for aid in archive_ids
            if (date_dir / f"{current.strftime('%Y%m%d')}-{aid}-{feed_id}.mp3").exists()
        ]
        todo = [aid for aid in archive_ids if aid not in already]

        remaining = (max_blocks - len(downloaded)) if max_blocks is not None else None
        if remaining is not None:
            todo = todo[:remaining]

        click.echo(
            f"  {current}: {len(archive_ids)} blocks "
            f"({len(already)} cached, downloading {len(todo)}"
            + (f" of {len(archive_ids) - len(already)} remaining" if remaining is not None and remaining < len(archive_ids) - len(already) else "")
            + ")"
        )

        # Cached blocks count toward max_blocks
        for aid in already:
            p = date_dir / f"{current.strftime('%Y%m%d')}-{aid}-{feed_id}.mp3"
            if p not in downloaded:
                downloaded.append(p)

        for aid in todo:
            if max_blocks is not None and len(downloaded) >= max_blocks:
                break
            try:
                path = _download_block(feed_id, aid, current, date_dir, headers)
                if path:
                    downloaded.append(path)
                    click.echo(f"    OK {path.name}")
            except Exception as exc:
                click.echo(f"    FAIL {aid}: {exc}", err=True)
            time.sleep(INTER_REQUEST_DELAY)

        time.sleep(2)  # pause between days
        current += timedelta(days=1)

    return downloaded


@click.command()
@click.option("--city", required=True, help="City key from config/cities.yaml")
@click.option("--start", "start_str", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_str", required=True, help="End date YYYY-MM-DD (inclusive)")
@click.option("--jobs", default=1, show_default=True, help="Parallel download threads (1 = sequential, safest for rate limits)")
@click.option("--relogin", is_flag=True, default=False, help="Force re-authentication")
@click.option("--max-blocks", default=None, type=int, help="Stop after downloading this many blocks (useful for probing)")
@click.option("--data-dir", default="data", show_default=True, help="Root data directory")
def main(city, start_str, end_str, jobs, relogin, max_blocks, data_dir):
    """Download Broadcastify archive blocks for a city and date range."""
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    click.echo(f"Downloading {city} | {start} → {end} | {jobs} threads")
    paths = download_range(city, start, end, jobs=jobs, relogin=relogin,
                           data_dir=Path(data_dir), max_blocks=max_blocks)
    click.echo(f"\nDone. {len(paths)} blocks downloaded.")


if __name__ == "__main__":
    main()
