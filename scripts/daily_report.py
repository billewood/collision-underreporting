"""
Send a daily email summary of the overnight download run.

Reads data/pipeline.log, extracts stats from the most recent
download_next.py run, and sends a summary via Resend.

Usage:
  python scripts/daily_report.py
"""

import os
import re
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

RESEND_API_URL = "https://api.resend.com/emails"
LOG_PATH = Path("/media/bill/FivetbEphys/collision data/pipeline.log")


def _send_email(subject: str, html: str):
    api_key = os.environ.get("RESEND_API_KEY")
    to = os.environ.get("EMAIL_TO")
    from_addr = os.environ.get("EMAIL_FROM", "Collision Pipeline <onboarding@resend.dev>")

    if not api_key:
        print("RESEND_API_KEY not set — skipping email")
        return
    if not to:
        print("EMAIL_TO not set — skipping email")
        return

    resp = requests.post(
        RESEND_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"from": from_addr, "to": to, "subject": subject, "html": html},
        timeout=15,
    )
    resp.raise_for_status()
    print(f"Email sent: {resp.json().get('id')}")


def _parse_last_run(log_text: str) -> dict:
    """Extract stats from the most recent download_next.py run in the log."""
    # Split into runs by the "Quota:" header line
    runs = re.split(r"(?=Quota: \d+ files)", log_text)
    if not runs or not any("Quota:" in r for r in runs):
        return {}

    last_run = [r for r in runs if "Quota:" in r][-1]

    quota_match = re.search(r"Quota: (\d+) files \| (\d+) missing dates", last_run)
    total_match = re.search(r"Done\. (\d+) files downloaded this run", last_run)
    quota_stop = "Quota reached" in last_run
    no_missing = "No missing dates found" in last_run

    dates_downloaded = re.findall(r"Downloading (\d{4}-\d{2}-\d{2})", last_run)
    failed_dates = re.findall(r"Download failed for (\d{4}-\d{2}-\d{2})", last_run)

    per_date = re.findall(
        r"(\d{4}-\d{2}-\d{2}) \(\d+/\d+ files used so far\).*?(\d+) files downloaded \| running total: (\d+)/\d+",
        last_run,
        re.DOTALL,
    )

    return {
        "quota": int(quota_match.group(1)) if quota_match else None,
        "missing_total": int(quota_match.group(2)) if quota_match else None,
        "files_downloaded": int(total_match.group(1)) if total_match else 0,
        "quota_reached": quota_stop,
        "complete": no_missing,
        "dates_attempted": dates_downloaded,
        "dates_failed": failed_dates,
        "per_date": per_date,
    }


def _build_html(stats: dict) -> str:
    today = date.today().strftime("%B %d, %Y")
    files = stats.get("files_downloaded", 0)
    missing = stats.get("missing_total", "?")
    dates_attempted = stats.get("dates_attempted", [])
    dates_failed = stats.get("dates_failed", [])
    dates_ok = [d for d in dates_attempted if d not in dates_failed]

    if stats.get("complete"):
        status_line = "✅ Historical archive is complete — no missing dates remaining."
        status_color = "#2e7d32"
    elif dates_failed:
        status_line = f"⚠️ Download stopped early — {len(dates_failed)} date(s) failed."
        status_color = "#b71c1c"
    elif stats.get("quota_reached"):
        status_line = f"✅ Quota reached — {files} files downloaded across {len(dates_ok)} dates."
        status_color = "#1565c0"
    elif files == 0:
        status_line = "⚠️ No files downloaded this run."
        status_color = "#b71c1c"
    else:
        status_line = f"✅ {files} files downloaded."
        status_color = "#2e7d32"

    dates_html = ""
    if dates_ok:
        dates_html = "<ul>" + "".join(f"<li>{d}</li>" for d in dates_ok) + "</ul>"
    if dates_failed:
        dates_html += "<p><strong>Failed:</strong></p><ul style='color:#b71c1c'>" + \
                      "".join(f"<li>{d}</li>" for d in dates_failed) + "</ul>"

    return f"""
    <div style="font-family:sans-serif;max-width:520px;margin:0 auto;padding:24px">
      <h2 style="margin-top:0">Collision Pipeline — {today}</h2>
      <p style="color:{status_color};font-weight:bold">{status_line}</p>
      <table style="border-collapse:collapse;width:100%">
        <tr><td style="padding:6px 0;color:#555">Files downloaded</td>
            <td style="padding:6px 0;font-weight:bold">{files}</td></tr>
        <tr><td style="padding:6px 0;color:#555">Dates remaining</td>
            <td style="padding:6px 0">{missing}</td></tr>
        <tr><td style="padding:6px 0;color:#555">Dates downloaded</td>
            <td style="padding:6px 0">{len(dates_ok)}</td></tr>
      </table>
      {f'<h3>Dates downloaded</h3>{dates_html}' if dates_html else ''}
      <hr style="margin-top:24px;border:none;border-top:1px solid #eee">
      <p style="color:#aaa;font-size:12px">collision-underreporting pipeline · el_cerrito</p>
    </div>
    """


def main():
    if not LOG_PATH.exists():
        print(f"Log not found: {LOG_PATH}")
        return

    log_text = LOG_PATH.read_text()
    stats = _parse_last_run(log_text)

    if not stats:
        print("No download run found in log.")
        return

    today = date.today().strftime("%Y-%m-%d")
    files = stats.get("files_downloaded", 0)
    subject = f"Pipeline {today} — {files} files downloaded"

    html = _build_html(stats)
    _send_email(subject, html)


if __name__ == "__main__":
    main()
