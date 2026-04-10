"""
Environment and hardware diagnostic for collision-underreporting.

Run this before your first pipeline run to verify your setup.

Usage:
    python scripts/check_env.py
"""

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def _ok(label: str, value: str) -> None:
    print(f"  {'OK':<6} {label:<28} {value}")


def _warn(label: str, value: str) -> None:
    print(f"  {'WARN':<6} {label:<28} {value}")


def _fail(label: str, value: str) -> None:
    print(f"  {'FAIL':<6} {label:<28} {value}")


# ── Environment manager ───────────────────────────────────────────────────────

def detect_env_manager() -> tuple[str, str]:
    """Returns (manager_name, env_name_or_path)."""
    # Conda
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_env or conda_prefix:
        name = conda_env or Path(conda_prefix).name
        return "conda", name

    # venv / virtualenv
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return "venv", Path(venv).name

    # Pipenv
    if os.environ.get("PIPENV_ACTIVE"):
        return "pipenv", os.environ.get("PIPENV_ACTIVE", "")

    # Poetry
    if os.environ.get("POETRY_ACTIVE"):
        return "poetry", ""

    # pyenv (shim in PATH but no active venv)
    if shutil.which("pyenv"):
        try:
            result = subprocess.run(
                ["pyenv", "version-name"], capture_output=True, text=True
            )
            return "pyenv", result.stdout.strip()
        except Exception:
            pass

    return "none", "(system Python)"


def check_environment() -> None:
    _section("Python Environment")
    manager, env_name = detect_env_manager()

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_path = sys.executable

    if sys.version_info >= (3, 11):
        _ok("Python", f"{py_ver}  ({py_path})")
    else:
        _fail("Python", f"{py_ver} — Python 3.11+ required")

    if manager == "none":
        _warn("Env manager", "none — running in system Python (use conda or venv)")
    else:
        _ok("Env manager", f"{manager}  (env: {env_name})")

    _ok("Working dir", str(Path.cwd()))


# ── Hardware ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


def _bytes_to_gb(b: int) -> str:
    return f"{b / 1024**3:.1f} GB"


def check_hardware() -> None:
    _section("Hardware")
    system = platform.system()
    machine = platform.machine()

    # ── Platform
    if system == "Darwin":
        sw = _run(["sw_vers", "-productVersion"])
        print(f"  Platform:  macOS {sw}  ({machine})")
        if machine == "arm64":
            # Apple Silicon
            chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if not chip:
                chip = "Apple Silicon"
            mem_bytes = int(_run(["sysctl", "-n", "hw.memsize"]) or 0)
            mem = _bytes_to_gb(mem_bytes) if mem_bytes else "unknown"
            print(f"  CPU/SoC:   {chip}")
            print(f"  RAM:       {mem}  (unified memory — shared with GPU)")
            print()
            print("  GPU:       Apple Silicon (Metal via unified memory)")
            # Model recommendation
            if mem_bytes >= 16 * 1024**3:
                _ok("Whisper rec.", "--model large-v3 --device cpu")
            else:
                _ok("Whisper rec.", "--model medium.en --device cpu")
        else:
            cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            mem_bytes = int(_run(["sysctl", "-n", "hw.memsize"]) or 0)
            print(f"  CPU:       {cpu}")
            print(f"  RAM:       {_bytes_to_gb(mem_bytes) if mem_bytes else 'unknown'}")
            _warn("GPU", "No NVIDIA GPU detected (Intel Mac) — transcription will be slow on CPU")
            _ok("Whisper rec.", "--model small.en --device cpu")
    else:
        # Linux / Windows
        cpu = _run(["bash", "-c", "grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2"]).strip()
        mem_kb = 0
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    mem_kb = int(line.split()[1])
                    break
        except Exception:
            pass
        print(f"  Platform:  {system} ({machine})")
        print(f"  CPU:       {cpu or 'unknown'}")
        print(f"  RAM:       {_bytes_to_gb(mem_kb * 1024) if mem_kb else 'unknown'}")
        print()

        # CUDA / NVIDIA GPU
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            # GPU name and VRAM
            gpu_name = _run([
                "nvidia-smi", "--query-gpu=name",
                "--format=csv,noheader"
            ]).split("\n")[0].strip()
            vram_mib = _run([
                "nvidia-smi", "--query-gpu=memory.total",
                "--format=csv,noheader,nounits"
            ]).split("\n")[0].strip()
            try:
                vram_gb = int(vram_mib) / 1024
                vram_str = f"{vram_gb:.1f} GB VRAM"
            except ValueError:
                vram_str = f"{vram_mib} MiB VRAM"

            driver = _run([
                "nvidia-smi", "--query-gpu=driver_version",
                "--format=csv,noheader"
            ]).split("\n")[0].strip()

            print(f"  GPU:       {gpu_name}  ({vram_str})")
            print(f"  Driver:    {driver}")

            # CUDA toolkit
            nvcc = _run(["nvcc", "--version"])
            if nvcc:
                for line in nvcc.splitlines():
                    if "release" in line.lower():
                        print(f"  CUDA:      {line.strip()}")
                        break

            # Torch CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    _ok("PyTorch CUDA", f"available  (torch {torch.__version__})")
                    cc = torch.cuda.get_device_capability()
                    print(f"             Compute capability: {cc[0]}.{cc[1]}")
                else:
                    _warn("PyTorch CUDA", "torch installed but CUDA not available")
            except ImportError:
                _warn("PyTorch", "not installed — needed for CUDA device detection")

            try:
                vram_gb_f = float(vram_mib) / 1024
                if vram_gb_f >= 6:
                    _ok("Whisper rec.", "--model large-v3 --device cuda")
                elif vram_gb_f >= 2:
                    _ok("Whisper rec.", "--model medium.en --device cuda")
                else:
                    _ok("Whisper rec.", "--model small.en --device cuda")
            except ValueError:
                _ok("Whisper rec.", "--model medium.en --device cuda")
        else:
            _warn("GPU", "nvidia-smi not found — no CUDA GPU detected")
            _warn("Whisper rec.", "--model small.en --device cpu  (will be slow)")


# ── System dependencies ───────────────────────────────────────────────────────

def check_system_deps() -> None:
    _section("System Dependencies")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        ver = _run(["ffmpeg", "-version"]).splitlines()[0] if ffmpeg else ""
        _ok("ffmpeg", ver[:60] if ver else ffmpeg)
    else:
        _fail(
            "ffmpeg",
            "NOT FOUND — required for MP3 decoding (pydub)\n"
            "         Install: brew install ffmpeg  OR  apt install ffmpeg",
        )


# ── Python packages ───────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    "faster-whisper",
    "anthropic",
    "requests",
    "pydub",
    "scipy",
    "numpy",
    "pyyaml",
    "pandas",
    "click",
    "python-dotenv",
    "matplotlib",
]


def check_packages() -> None:
    _section("Python Packages")
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            ver = importlib.metadata.version(pkg)
            _ok(pkg, ver)
        except importlib.metadata.PackageNotFoundError:
            _fail(pkg, "NOT INSTALLED — run: pip install -r requirements.txt")
            all_ok = False
    if all_ok:
        print("\n  All packages installed.")


# ── Credentials ───────────────────────────────────────────────────────────────

CREDENTIAL_KEYS = [
    ("BROADCASTIFY_USERNAME", "Broadcastify account email"),
    ("BROADCASTIFY_PASSWORD", "Broadcastify account password"),
    ("ANTHROPIC_API_KEY",     "Anthropic API key (incident extraction)"),
]


def check_credentials() -> None:
    _section("Credentials (.env)")

    env_file = Path(".env")
    if not env_file.exists():
        _warn(".env file", "not found — run: python scripts/setup_credentials.py")
    else:
        _ok(".env file", str(env_file.resolve()))
        # Load without polluting the environment
        env_vals: dict[str, str] = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env_vals[k.strip()] = v.strip()

    print()
    for key, description in CREDENTIAL_KEYS:
        val = os.environ.get(key) or env_vals.get(key, "") if env_file.exists() else os.environ.get(key, "")  # noqa
        if val:
            masked = val[:4] + "****" if len(val) > 4 else "****"
            _ok(key, f"set  ({masked})")
        else:
            _warn(key, f"not set  — {description}")

    if not env_file.exists() or any(
        not (os.environ.get(k) or (env_file.exists() and k in Path(".env").read_text()))
        for k, _ in CREDENTIAL_KEYS
    ):
        print("\n  Run: python scripts/setup_credentials.py")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== collision-underreporting: environment check ===")
    check_environment()
    check_hardware()
    check_system_deps()
    check_packages()
    check_credentials()
    print()
