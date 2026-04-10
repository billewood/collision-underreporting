"""
Top-level pipeline orchestrator.

Runs any combination of pipeline steps for a city and date range.

Usage examples:

  # Full pipeline, 1-week test run
  python scripts/run_pipeline.py \\
    --city el_cerrito --start 2025-10-01 --end 2025-10-07 \\
    --steps download,transcribe,extract

  # Transcribe + extract only (audio already downloaded)
  python scripts/run_pipeline.py \\
    --city el_cerrito --start 2025-10-01 --end 2025-10-07 \\
    --steps transcribe,extract --model medium.en --device cuda

  # Analysis only
  python scripts/run_pipeline.py \\
    --city el_cerrito --start 2025-10-01 --end 2025-10-31 \\
    --steps analyze --chart
"""

import platform
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import click

VALID_STEPS = {"download", "transcribe", "extract", "analyze"}


def _hardware_summary() -> str:
    """One-line hardware summary shown at pipeline startup."""
    system = platform.system()
    machine = platform.machine()

    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
        except Exception:
            return ""

    if system == "Darwin" and machine == "arm64":
        chip = run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
        mem_bytes = int(run(["sysctl", "-n", "hw.memsize"]) or 0)
        mem = f"{mem_bytes / 1024**3:.0f} GB" if mem_bytes else "?"
        return f"{chip}  |  {mem} unified memory  |  recommend: --model large-v3 --device cpu"

    if shutil.which("nvidia-smi"):
        gpu = run(["nvidia-smi", "--query-gpu=name,memory.total",
                   "--format=csv,noheader"]).split("\n")[0]
        try:
            import torch
            cuda = "CUDA available" if torch.cuda.is_available() else "CUDA unavailable"
        except ImportError:
            cuda = "torch not installed"
        return f"{gpu}  |  {cuda}  |  recommend: --model medium.en --device cuda"

    cpu = run(["bash", "-c",
               "grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2"]).strip()
    return f"{cpu or platform.processor() or 'CPU'}  |  no GPU detected  |  recommend: --model small.en --device cpu"


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


@click.command()
@click.option("--city", required=True, help="City key from config/cities.yaml")
@click.option("--start", "start_str", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_str", required=True, help="End date YYYY-MM-DD (inclusive)")
@click.option(
    "--steps",
    default="download,transcribe,extract",
    show_default=True,
    help="Comma-separated steps: download,transcribe,extract,analyze",
)
@click.option("--model", "whisper_model", default="medium.en", show_default=True,
              help="Whisper model (medium.en for GTX970, large-v3 for M3 Pro)")
@click.option("--device", default=None,
              help="cuda / mps / cpu (auto-detected if omitted)")
@click.option("--jobs", default=4, show_default=True,
              help="Parallel download threads")
@click.option("--relogin", is_flag=True, default=False,
              help="Force Broadcastify re-authentication")
@click.option("--overwrite", is_flag=True, default=False,
              help="Re-process files that already exist")
@click.option("--min-confidence", default=0.5, show_default=True,
              help="Minimum confidence score for incidents (analyze step)")
@click.option("--chart", is_flag=True, default=False,
              help="Save comparison chart (analyze step)")
@click.option("--data-dir", default="data", show_default=True)
def main(
    city, start_str, end_str, steps, whisper_model, device,
    jobs, relogin, overwrite, min_confidence, chart, data_dir,
):
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    data = Path(data_dir)

    step_list = [s.strip().lower() for s in steps.split(",")]
    invalid = set(step_list) - VALID_STEPS
    if invalid:
        click.echo(f"Unknown steps: {invalid}. Valid: {VALID_STEPS}", err=True)
        sys.exit(1)

    days = (end - start).days + 1
    click.echo(f"Hardware:  {_hardware_summary()}")
    click.echo(f"Pipeline:  {city} | {start} → {end} ({days} days) | steps: {step_list}")
    click.echo(f"Data dir:  {data.resolve()}\n")

    # ── Download ─────────────────────────────────────────────────────────────
    if "download" in step_list:
        click.echo("── Step 1: Download ─────────────────────────────────────────")
        from broadcastify.download import download_range
        paths = download_range(
            city, start, end, jobs=jobs, relogin=relogin, data_dir=data
        )
        click.echo(f"  Downloaded {len(paths)} blocks\n")

    # ── Transcribe ───────────────────────────────────────────────────────────
    if "transcribe" in step_list:
        click.echo("── Step 2: Transcribe ───────────────────────────────────────")
        from broadcastify.transcribe import transcribe_date
        total = 0
        for dt in _date_range(start, end):
            paths = transcribe_date(
                city, dt,
                model_size=whisper_model,
                device=device,
                data_dir=data,
                overwrite=overwrite,
            )
            total += len(paths)
        click.echo(f"  Transcribed {total} blocks\n")

    # ── Extract ──────────────────────────────────────────────────────────────
    if "extract" in step_list:
        click.echo("── Step 3: Extract incidents ────────────────────────────────")
        from broadcastify.extract import extract_date
        total = 0
        for dt in _date_range(start, end):
            incidents = extract_date(city, dt, data_dir=data, overwrite=overwrite)
            total += len(incidents)
        click.echo(f"  Extracted {total} incidents total\n")

    # ── Analyze ──────────────────────────────────────────────────────────────
    if "analyze" in step_list:
        click.echo("── Step 4: Analyze ──────────────────────────────────────────")
        from analysis.compare import (
            build_comparison_table,
            load_dispatch_incidents,
            load_switrs_incidents,
            plot_comparison,
            print_summary,
        )
        dispatch_df = load_dispatch_incidents(city, start, end, data, min_confidence)
        switrs_df = load_switrs_incidents(city, start, end, data)
        table = build_comparison_table(dispatch_df, switrs_df)
        print_summary(table, city)
        if chart:
            from pathlib import Path as P
            out = data / "analysis" / f"{city}_gap.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plot_comparison(table, city, out)

    click.echo("Pipeline complete.")


if __name__ == "__main__":
    main()
