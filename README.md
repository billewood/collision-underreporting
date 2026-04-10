# Collision Underreporting Analysis

Quantifies bicycle and pedestrian collision underreporting in the East Bay and SF by comparing 911 dispatch incident data against official SWITRS collision records.

**Core hypothesis:** EMS dispatch calls for bike/ped collisions significantly outnumber what appears in official police-reported statistics. The D.C. Policy Center found 30% of pedestrian/cyclist crashes lacked police reports in a comparable 2021 study.

---

## How It Works

Each city has a different source for dispatch incident data, but all sources produce the same normalized incident schema, which is then compared against SWITRS.

```
Dispatch source (varies by city)
        ↓
   Normalize to incident schema
        ↓
   SWITRS (official reported collisions)
        ↓
   Gap ratio analysis
```

### Dispatch data sources by city

| City | Source | Method |
|---|---|---|
| El Cerrito / Albany / Kensington | Broadcastify feed #33365 (CCRFCC Fire/EMS) | Download MP3 archives → transcribe (Whisper) → extract incidents (Claude Haiku) |
| Berkeley | BPD public 10-minute-delay call log | Parse structured call log — no audio needed |
| San Francisco | DataSF CAD API | Pull structured dispatch data via API — no audio needed |

El Cerrito/Albany/Kensington use audio because Fire/EMS dispatch is still unencrypted (El Cerrito PD encrypted Sept 2025 countywide). Berkeley and SF have structured public data sources that don't require audio transcription.

### Official collision baseline

SWITRS (California Statewide Integrated Traffic Records System) via [TIMS](https://tims.berkeley.edu). Only collisions where a police report was filed are included — the known undercount this project aims to quantify.

---

## Project Structure

```
collision-underreporting/
├── config/
│   ├── cities.yaml          # Per-city config (source type, feed IDs, API URLs)
│   └── keywords.yaml        # Collision-relevant dispatch keywords (pre-filter)
├── broadcastify/
│   ├── auth.py              # Broadcastify login + cookie cache
│   ├── download.py          # Archive MP3 download (date range)
│   ├── transcribe.py        # faster-whisper + audio preprocessing
│   └── extract.py           # Keyword filter → Claude Haiku incident extraction
├── switrs/
│   └── pull.py              # Load and normalize TIMS CSV exports
├── analysis/
│   └── compare.py           # Monthly gap ratio table + chart
├── scripts/
│   └── run_pipeline.py      # Top-level orchestrator
└── data/                    # gitignored
    ├── audio/
    ├── transcripts/
    ├── incidents/
    └── switrs/
```

Berkeley (`call_log.py`) and SF (`cad_api.py`) ingestion modules are not yet built — they will live in `berkeley/` and `sf/` and output to the same `data/incidents/` schema.

---

## Setup

### Prerequisites

- Python 3.11+
- [ffmpeg](https://ffmpeg.org) — required for MP3 decoding (`pydub` dependency)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- [Broadcastify Premium](https://www.broadcastify.com) (~$30/yr) — required for archive downloads
- Anthropic API key — used for incident extraction (~$1–5/year with keyword pre-filtering)

### 1. Create a Python environment

**conda (recommended):**
```bash
conda create -n collision-underreporting python=3.11
conda activate collision-underreporting
```

**venv:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up credentials

Run the interactive wizard — it prompts for each credential and writes `.env`:
```bash
python scripts/setup_credentials.py
```

Or create `.env` manually:
```
BROADCASTIFY_USERNAME=your@email.com
BROADCASTIFY_PASSWORD=yourpassword
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Verify your setup

```bash
python scripts/check_env.py
```

This checks your Python version, environment manager, hardware (GPU/VRAM, RAM, Apple Silicon), system dependencies (ffmpeg), installed packages, and credentials. It also recommends the right `--model` and `--device` flags for your machine.

Example output:
```
── Hardware ─────────────────────────────────────────
  OK     Platform                 macOS 15.3 (arm64)
  OK     CPU/SoC                  Apple M3 Pro
  OK     RAM                      36.0 GB (unified memory)
  OK     Whisper rec.             --model large-v3 --device cpu

── Hardware ──────────── (Linux/NVIDIA example) ────
  OK     GPU                      NVIDIA GeForce GTX 970  (3.9 GB VRAM)
  OK     PyTorch CUDA             available (torch 2.2.0)
  OK     Whisper rec.             --model medium.en --device cuda
```

---

## Running the Pipeline

The pipeline auto-displays your hardware summary at startup so you can confirm it's using the right device.

### El Cerrito / Albany / Kensington (Broadcastify audio)

**Recommended first run — 1-week test:**
```bash
python scripts/run_pipeline.py \
  --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 \
  --steps download,transcribe,extract
```
`--model` and `--device` are auto-detected. Override if needed:
```bash
# GTX 970 (Linux)
python scripts/run_pipeline.py --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 \
  --steps download,transcribe,extract \
  --model medium.en --device cuda

# M3 Pro (Mac)
python scripts/run_pipeline.py --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 \
  --steps download,transcribe,extract \
  --model large-v3
```

**Run individual steps:**
```bash
# Download only
python scripts/run_pipeline.py --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 --steps download

# Transcribe previously downloaded audio
python scripts/run_pipeline.py --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 --steps transcribe

# Extract incidents from existing transcripts
python scripts/run_pipeline.py --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 --steps extract

# Re-authenticate (if Broadcastify session expired)
python scripts/run_pipeline.py --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 --steps download --relogin
```

**Re-process existing files:**
```bash
python scripts/run_pipeline.py --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 \
  --steps transcribe,extract --overwrite
```

**Disable audio preprocessing** (to compare raw vs. filtered transcription quality):
```bash
python scripts/run_pipeline.py --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 \
  --steps transcribe --no-preprocess
```

### SWITRS data

SWITRS has no public API — export manually from [tims.berkeley.edu](https://tims.berkeley.edu):
1. Go to **Data → Collisions**
2. Filter by jurisdiction (e.g. "El Cerrito"), date range, and collision type
3. Export as CSV
4. Save to `data/switrs/el_cerrito/`

> SWITRS lags 6–18 months, so use date ranges from at least a year ago for meaningful comparison.

### Analysis

```bash
# Print gap ratio table
python scripts/run_pipeline.py \
  --city el_cerrito \
  --start 2025-01-01 --end 2025-06-30 \
  --steps analyze

# Print table + save time-series chart to data/analysis/el_cerrito_gap.png
python scripts/run_pipeline.py \
  --city el_cerrito \
  --start 2025-01-01 --end 2025-06-30 \
  --steps analyze --chart
```

### All options

```
--city            City key from config/cities.yaml (required)
--start           Start date YYYY-MM-DD (required)
--end             End date YYYY-MM-DD inclusive (required)
--steps           Comma-separated: download,transcribe,extract,analyze
--model           Whisper model: tiny.en / small.en / medium.en / large-v3
--device          cuda / cpu (auto-detected if omitted)
--jobs            Parallel download threads (default: 4)
--relogin         Force Broadcastify re-authentication
--overwrite       Re-process files that already exist
--no-preprocess   Skip bandpass filter and alert tone trimming
--min-confidence  Minimum incident confidence score for analysis (default: 0.5)
--chart           Save comparison chart (analyze step only)
--data-dir        Root data directory (default: data/)
```

---

## Audio Preprocessing (Broadcastify pipeline)

Each MP3 block passes through a preprocessing stage before Whisper:

1. **Decode + mono 16kHz** — Whisper's native format
2. **Bandpass filter (300–3400 Hz)** — Narrow-band radio speech range; removes low-frequency rumble and high-frequency hiss
3. **Alert tone trimming** — Detects and removes Quick Call II / MDC-1200 dispatch tones at block start (low-variance high-energy bursts)
4. **VAD filter** — Built into faster-whisper; skips silence/static (~70–80% of scanner audio is dead air)
5. **Hallucination filter** — Post-transcription; strips known Whisper junk segments and consecutive duplicates

Disable preprocessing with `--no-preprocess` to compare raw vs. filtered output.

---

## Incident Extraction

For Broadcastify cities, extracted incidents are stored as JSONL in `data/incidents/{city}/{YYYYMMDD}.jsonl`:

```json
{
  "incident_type": "bicycle_collision",
  "involves_bicycle": true,
  "involves_pedestrian": false,
  "location": "3500 block San Pablo Ave",
  "injuries_mentioned": true,
  "block_start_utc": "2025-10-03T14:21:00+00:00",
  "city": "el_cerrito",
  "confidence": 0.92
}
```

Only blocks matching collision keywords are sent to Claude Haiku (~5–10% of blocks), keeping API costs low.
