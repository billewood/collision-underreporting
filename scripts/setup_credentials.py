"""
Interactive wizard to create or update the .env credentials file.

Usage:
    python scripts/setup_credentials.py

Prompts for each credential, shows what's already set, and writes
(or updates) .env in the project root without touching other variables.
"""

import getpass
import os
import sys
from pathlib import Path

ENV_FILE = Path(".env")

FIELDS = [
    {
        "key": "BROADCASTIFY_USERNAME",
        "label": "Broadcastify username (email)",
        "secret": False,
        "help": "Your broadcastify.com account email. Premium subscription required for archive downloads.",
    },
    {
        "key": "BROADCASTIFY_PASSWORD",
        "label": "Broadcastify password",
        "secret": True,
        "help": "Your broadcastify.com account password.",
    },
    {
        "key": "ANTHROPIC_API_KEY",
        "label": "Anthropic API key",
        "secret": True,
        "help": "From console.anthropic.com. Used for incident extraction (~$1-5/year).",
    },
]


def _load_existing() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    vals: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            vals[k.strip()] = v.strip()
    return vals


def _write_env(vals: dict[str, str]) -> None:
    lines = ["# collision-underreporting credentials\n"]
    for k, v in vals.items():
        lines.append(f"{k}={v}\n")
    ENV_FILE.write_text("".join(lines))


def _mask(val: str) -> str:
    if not val:
        return "(not set)"
    if len(val) <= 4:
        return "****"
    return val[:4] + "****"


def main() -> None:
    print("\n=== Credential Setup ===")
    print(f"Writing to: {ENV_FILE.resolve()}\n")

    existing = _load_existing()
    updated = dict(existing)

    for field in FIELDS:
        key = field["key"]
        current = existing.get(key, "")

        print(f"{field['label']}")
        print(f"  {field['help']}")
        if current:
            print(f"  Current value: {_mask(current)}")
            keep = input("  Keep existing value? [Y/n]: ").strip().lower()
            if keep in ("", "y", "yes"):
                print()
                continue

        while True:
            if field["secret"]:
                val = getpass.getpass("  Enter value (hidden): ").strip()
            else:
                val = input("  Enter value: ").strip()
            if val:
                updated[key] = val
                break
            print("  Value cannot be empty. Try again.")
        print()

    _write_env(updated)
    print(f".env written to {ENV_FILE.resolve()}")
    print("Keep this file out of version control — it's already in .gitignore.")

    # Sanity check
    missing = [f["key"] for f in FIELDS if not updated.get(f["key"])]
    if missing:
        print(f"\nWarning: still missing: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("\nAll credentials set. Run 'python scripts/check_env.py' to verify.")


if __name__ == "__main__":
    main()
