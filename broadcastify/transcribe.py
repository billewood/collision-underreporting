"""
Transcribe Broadcastify archive MP3 blocks using faster-whisper.

Key design choices vs. broadcastify-cli defaults:
  - preprocess_audio(): bandpass filter + alert tone trim before Whisper
  - vad_filter=True: skip silence/static (scanner audio is ~70-80% dead air)
  - no_speech_threshold=0.6: tighter than CopCrawler's 2.0 — less hallucination
  - filter_hallucinations(): strip known Whisper junk segments post-transcription
  - EMS-focused initial_prompt: domain adaptation without fine-tuning
  - Auto-detects device: CUDA (Linux/GTX970) → CPU (Apple Silicon M-series)

Preprocessing pipeline (per block):
  MP3 → pydub decode → mono 16kHz → bandpass 300–3400 Hz → trim alert tone
  → temp WAV → faster-whisper → hallucination filter → JSON transcript

Usage:
    python -m broadcastify.transcribe --city el_cerrito --date 2025-10-01
    python -m broadcastify.transcribe --city el_cerrito --date 2025-10-01 --model large-v3
    python -m broadcastify.transcribe --city el_cerrito --date 2025-10-01 --no-preprocess
"""

import json
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import click
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, sosfilt

# Delay import — faster-whisper takes a moment to load
_model_cache: dict = {}

# Whisper's native sample rate
WHISPER_SR = 16_000

# Scanner audio frequency range (standard narrow-band radio bandwidth)
BANDPASS_LOW_HZ = 300
BANDPASS_HIGH_HZ = 3_400

# Alert tone detection: look at this many seconds at the start of each block
TONE_SCAN_SECONDS = 4.0
# Frame size for RMS energy analysis (milliseconds)
TONE_FRAME_MS = 100
# A tone segment has RMS variance below this threshold (speech has higher variance)
TONE_VARIANCE_THRESHOLD = 0.015

# Whisper hallucination patterns — common phrases generated on silence/noise
# Source: openai/whisper#679, RadioTranscriber project
_HALLUCINATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\.?\s*$",                       # empty or just punctuation
        r"^\s*you\.?\s*$",                  # single word "You."
        r"^\s*the\.?\s*$",                  # single word "The."
        r"thank\s+you\s+(for\s+watching)?", # "Thank you [for watching]"
        r"please\s+sub(scribe)?",
        r"transcribed\s+by",
        r"subtitles?\s+by",
        r"www\.",
        r"\.com\b",
        r"like\s+and\s+subscribe",
        r"^\s*\[?(music|laughter|applause)\]?\s*$",
    ]
]


def _bandpass_filter(samples: np.ndarray, sr: int) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter (300–3400 Hz)."""
    nyq = sr / 2.0
    low = BANDPASS_LOW_HZ / nyq
    high = BANDPASS_HIGH_HZ / nyq
    # Clamp to valid range (must be strictly between 0 and 1)
    low = max(1e-4, min(low, 0.999))
    high = max(low + 1e-4, min(high, 0.999))
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, samples)


def _detect_tone_samples(samples: np.ndarray, sr: int) -> int:
    """
    Detect alert/dispatch tones at the start of a block and return the
    sample index where they end (i.e., where to start transcription).

    EMS/fire Quick Call II tones are sustained, high-energy, low-variance
    signals. Speech has much higher frame-to-frame RMS variance.

    Returns 0 if no tone detected (safe default — don't trim).
    """
    frame_size = int(sr * TONE_FRAME_MS / 1000)
    scan_samples = int(sr * TONE_SCAN_SECONDS)
    audio_slice = samples[:scan_samples]

    if len(audio_slice) < frame_size * 3:
        return 0

    # Compute per-frame RMS
    n_frames = len(audio_slice) // frame_size
    rms_frames = np.array([
        np.sqrt(np.mean(audio_slice[i * frame_size:(i + 1) * frame_size] ** 2))
        for i in range(n_frames)
    ])

    if rms_frames.max() == 0:
        return 0

    norm_rms = rms_frames / rms_frames.max()

    # Walk forward; find first frame where local variance spikes
    # (indicating the tone ended and speech/silence started)
    window = 4  # frames (~400 ms)
    tone_end_frame = 0
    for i in range(window, len(norm_rms)):
        local_var = np.var(norm_rms[max(0, i - window):i])
        if local_var > TONE_VARIANCE_THRESHOLD:
            # Variance spike = end of tone; back up one window to be safe
            tone_end_frame = max(0, i - 1)
            break

    # Only trim if a meaningful tone was detected (at least 0.3 seconds)
    if tone_end_frame < 3:
        return 0

    return tone_end_frame * frame_size


def preprocess_audio(mp3_path: Path, tmp_dir: str) -> Path:
    """
    Decode MP3, apply bandpass filter, trim leading alert tones, and
    write a 16kHz mono WAV to tmp_dir. Returns the WAV path.
    """
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio = audio.set_channels(1).set_frame_rate(WHISPER_SR)

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # Normalize to [-1, 1]
    max_val = np.iinfo(audio.array_type).max
    samples /= max_val

    # 1. Bandpass filter
    samples = _bandpass_filter(samples, WHISPER_SR)

    # 2. Trim leading alert tone
    trim_at = _detect_tone_samples(samples, WHISPER_SR)
    if trim_at > 0:
        samples = samples[trim_at:]

    # Write temp WAV
    out_path = Path(tmp_dir) / (mp3_path.stem + "_pre.wav")
    filtered_audio = AudioSegment(
        (samples * max_val).astype(np.int16).tobytes(),
        frame_rate=WHISPER_SR,
        sample_width=2,  # 16-bit
        channels=1,
    )
    filtered_audio.export(str(out_path), format="wav")
    return out_path


def filter_hallucinations(segments: list[dict]) -> list[dict]:
    """
    Remove segments that match known Whisper hallucination patterns
    or are exact duplicates of the immediately preceding segment.
    """
    cleaned = []
    prev_text = None
    for seg in segments:
        text = seg.get("text", "").strip()
        # Skip empty
        if not text:
            continue
        # Skip known hallucination patterns
        if any(p.search(text) for p in _HALLUCINATION_PATTERNS):
            continue
        # Skip exact repeat of previous segment (repetition loop hallucination)
        if text == prev_text:
            continue
        cleaned.append(seg)
        prev_text = text
    return cleaned


def _parse_filename(filename: str) -> tuple[str | None, str | None]:
    """
    Parse UTC timestamp and feed ID from a Broadcastify archive filename.
    Returns (block_start_utc, feed_id) — either may be None if unparseable.

    Two formats observed in the wild:

    Format A (broadcastify-cli convention):
      {YYYYMMDDHHmm}-{archive_id}-{feed_id}.mp3
      e.g. 202410150021-373175-5318.mp3 → timestamp=2024-10-15T00:21:00Z, feed=5318

    Format B (what this downloader produces — archive_id is a Unix timestamp):
      {YYYYMMDD}-{feed_id}-{unix_timestamp}-{feed_id}.mp3
      e.g. 20251001-33365-1759360481-33365.mp3 → timestamp=2025-10-01T23:14:41Z, feed=33365
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None, None

    # Format A: first part is 12 chars (YYYYMMDDHHmm)
    if len(parts[0]) == 12:
        try:
            dt = datetime.strptime(parts[0], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            feed_id = parts[2] if len(parts) >= 3 else None
            return dt.isoformat(), feed_id
        except ValueError:
            pass

    # Format B: first part is 8 chars (YYYYMMDD), third part is Unix timestamp
    if len(parts[0]) == 8 and len(parts) >= 3:
        try:
            ts = int(parts[2])
            # Sanity check: Unix timestamp should be after 2010 and before 2100
            if 1_262_304_000 < ts < 4_102_444_800:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                feed_id = parts[1]
                return dt.isoformat(), feed_id
        except (ValueError, OSError):
            pass

    return None, None


_model_lock = threading.Lock()


def _get_model(model_size: str, device: str, compute_type: str, cpu_threads: int = 0):
    key = (model_size, device, compute_type, cpu_threads)
    with _model_lock:
        if key not in _model_cache:
            from faster_whisper import WhisperModel
            _model_cache[key] = WhisperModel(
                model_size, device=device, compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
    return _model_cache[key]


def _detect_device() -> tuple[str, str]:
    """Return (device, compute_type) based on available hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "int8"
        # CTranslate2 Metal support is experimental; CPU with int8 is faster on M-series
        if torch.backends.mps.is_available():
            return "cpu", "int8"
    except ImportError:
        pass
    return "cpu", "int8"


EMS_PROMPT = (
    "EMS dispatch, fire department, ambulance, traffic collision, bicycle, "
    "pedestrian, hit and run, injury, medical, structure fire, engine, ladder, "
    "respond, copy, on scene, clear"
)


def transcribe_file(
    audio_path: Path,
    city: str,
    model_size: str = "medium.en",
    device: str | None = None,
    compute_type: str | None = None,
    preprocess: bool = True,
    use_vad: bool = True,
    cpu_threads: int = 0,
) -> dict:
    """
    Transcribe a single MP3 block. Returns transcript dict.
    """
    if device is None or compute_type is None:
        device, compute_type = _detect_device()

    model = _get_model(model_size, device, compute_type, cpu_threads)

    with tempfile.TemporaryDirectory() as tmp_dir:
        if preprocess:
            input_path = preprocess_audio(audio_path, tmp_dir)
        else:
            input_path = audio_path

        transcribe_kwargs = dict(
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.8,
            initial_prompt=EMS_PROMPT,
        )
        if use_vad:
            transcribe_kwargs["vad_filter"] = True
            transcribe_kwargs["vad_parameters"] = {
                "min_silence_duration_ms": 300,
                "threshold": 0.3,
            }
        else:
            transcribe_kwargs["vad_filter"] = False

        segments_iter, info = model.transcribe(str(input_path), **transcribe_kwargs)

        raw_segments = [
            {
                "text": seg.text.strip(),
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
            }
            for seg in segments_iter
        ]

    # Filter hallucinations after closing the temp dir
    segment_list = filter_hallucinations(raw_segments)
    full_text = " ".join(s["text"] for s in segment_list)

    block_start_utc, feed_id = _parse_filename(audio_path.name)
    return {
        "file": audio_path.name,
        "city": city,
        "feed_id": feed_id,
        "block_start_utc": block_start_utc,
        "model": model_size,
        "device": device,
        "preprocessed": preprocess,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "text": full_text,
        "segments": segment_list,
        "segments_raw_count": len(raw_segments),
    }


def transcribe_date(
    city: str,
    dt: date,
    model_size: str = "medium.en",
    device: str | None = None,
    compute_type: str | None = None,
    data_dir: Path = Path("data"),
    overwrite: bool = False,
    preprocess: bool = True,
    use_vad: bool = True,
    workers: int = 1,
) -> list[Path]:
    """
    Transcribe all downloaded blocks for a city on a given date.
    Returns list of output transcript paths.

    workers > 1: run that many files concurrently using threads. CTranslate2
    is thread-safe; each worker gets cpu_threads = max(1, total_cores // workers)
    so total CPU usage stays within the available core count.
    """
    audio_dir = data_dir / "audio" / city / dt.strftime("%Y%m%d")
    if not audio_dir.exists():
        click.echo(f"No audio found at {audio_dir}")
        return []

    transcript_dir = data_dir / "transcripts" / city / dt.strftime("%Y%m%d")
    transcript_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(audio_dir.glob("*.mp3"))
    if not mp3_files:
        click.echo(f"No MP3 files in {audio_dir}")
        return []

    if device is None or compute_type is None:
        device, compute_type = _detect_device()

    cpu_threads = max(1, os.cpu_count() // workers) if device == "cpu" else 0

    flags = []
    if not preprocess:
        flags.append("no-preprocess")
    if not use_vad:
        flags.append("no-vad")
    if workers > 1:
        flags.append(f"{workers} workers × {cpu_threads} threads")
    click.echo(
        f"Transcribing {len(mp3_files)} blocks | {city} {dt} | "
        f"{model_size} on {device}/{compute_type}"
        + (f" | {', '.join(flags)}" if flags else "")
    )

    # Pre-warm the model before entering the thread pool (avoid redundant loads)
    _get_model(model_size, device, compute_type, cpu_threads)

    pending = [
        mp3 for mp3 in mp3_files
        if overwrite or not (transcript_dir / (mp3.stem + ".json")).exists()
    ]
    already_done = [
        transcript_dir / (mp3.stem + ".json")
        for mp3 in mp3_files
        if not overwrite and (transcript_dir / (mp3.stem + ".json")).exists()
    ]

    print_lock = threading.Lock()
    completed: list[Path] = list(already_done)

    def _transcribe_one(mp3: Path) -> Path:
        result = transcribe_file(
            mp3, city, model_size, device, compute_type, preprocess, use_vad, cpu_threads
        )
        out_path = transcript_dir / (mp3.stem + ".json")
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        filtered = result["segments_raw_count"] - len(result["segments"])
        with print_lock:
            click.echo(
                f"  {mp3.name} → {len(result['segments'])} segments"
                + (f" ({filtered} hallucinations removed)" if filtered else "")
            )
        return out_path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_transcribe_one, mp3): mp3 for mp3 in pending}
        for future in as_completed(futures):
            completed.append(future.result())

    return completed


@click.command()
@click.option("--city", required=True)
@click.option("--date", "date_str", required=True, help="YYYY-MM-DD")
@click.option("--model", "model_size", default="medium.en", show_default=True)
@click.option("--device", default=None, help="cuda / mps / cpu (auto-detected if omitted)")
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--no-preprocess", "preprocess", is_flag=True, default=True,
              flag_value=False, help="Skip bandpass filter and tone trimming")
@click.option("--no-vad", "use_vad", is_flag=True, default=True,
              flag_value=False, help="Disable Whisper VAD filter (use if getting 0 segments)")
@click.option("--data-dir", default="data", show_default=True)
def main(city, date_str, model_size, device, overwrite, preprocess, use_vad, data_dir):
    """Transcribe downloaded audio blocks for a city on a given date."""
    dt = date.fromisoformat(date_str)
    paths = transcribe_date(
        city, dt,
        model_size=model_size,
        device=device,
        data_dir=Path(data_dir),
        overwrite=overwrite,
        preprocess=preprocess,
        use_vad=use_vad,
    )
    click.echo(f"\nDone. {len(paths)} transcripts written.")


if __name__ == "__main__":
    main()
