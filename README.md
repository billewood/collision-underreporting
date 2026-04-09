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

**Requirements:**
- Python 3.11+
- [Broadcastify Premium](https://www.broadcastify.com) (~$30/yr) — required for archive downloads
- Anthropic API key — used for incident extraction (~$1–5/year with keyword pre-filtering)

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
BROADCASTIFY_USERNAME=your@email.com
BROADCASTIFY_PASSWORD=yourpassword
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the Pipeline

### El Cerrito / Albany / Kensington (Broadcastify audio)

```bash
# Full pipeline: download → transcribe → extract incidents
python scripts/run_pipeline.py \
  --city el_cerrito \
  --start 2025-10-01 --end 2025-10-07 \
  --steps download,transcribe,extract \
  --model medium.en --device cuda

# Analysis (after also downloading SWITRS data — see below)
python scripts/run_pipeline.py \
  --city el_cerrito \
  --start 2025-10-01 --end 2025-10-31 \
  --steps analyze --chart
```

**Hardware:** GTX 970 (4GB) → `--model medium.en --device cuda` (~45 sec/block). M3 Pro → `--model large-v3` (~30 sec/block).

### SWITRS data

Export from [tims.berkeley.edu](https://tims.berkeley.edu) → Data → Collisions. Filter by jurisdiction, date range. Save CSV to `data/switrs/{city}/`.

SWITRS lags 6–18 months, so use date ranges from at least a year ago for comparison.

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
