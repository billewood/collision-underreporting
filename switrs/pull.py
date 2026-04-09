"""
SWITRS data loading and normalization.

SWITRS (Statewide Integrated Traffic Records System) is California's official
collision database. We access it via TIMS (UC Berkeley SafeTREC):
  https://tims.berkeley.edu

Data export is manual (TIMS has no public API as of 2026):
  1. Go to tims.berkeley.edu → Data → Collisions
  2. Filter by jurisdiction (e.g. "El Cerrito"), date range, and collision type
  3. Export as CSV → save to data/switrs/{city}/

This module loads, filters, and normalizes those CSVs to the same schema
used by the incident extraction output, enabling direct comparison.

Key SWITRS columns used:
  ACCIDENT_YEAR, COLLISION_DATE, COLLISION_TIME
  CITY_NAME (or JURIS for unincorporated)
  PRIMARY_RD, SECONDARY_RD
  COLLISION_SEVERITY (0=PDO, 1=fatal, 2=severe, 3=other visible, 4=complaint of pain)
  PEDESTRIAN_KILLED_COUNT, PEDESTRIAN_INJURED_COUNT
  BICYCLE_KILLED_COUNT, BICYCLE_INJURED_COUNT
  PCF_VIOL_CATEGORY (primary collision factor)
  TYPE_OF_COLLISION
"""

from datetime import date
from pathlib import Path

import click
import pandas as pd
import yaml

CITIES_FILE = Path("config/cities.yaml")

# SWITRS severity codes that represent injury collisions (exclude PDO = 0)
INJURY_SEVERITIES = {1, 2, 3, 4}


def _load_city_config(city: str) -> dict:
    cities = yaml.safe_load(CITIES_FILE.read_text())
    if city not in cities:
        raise ValueError(f"Unknown city '{city}'.")
    return cities[city]


def load_switrs_csv(path: Path) -> pd.DataFrame:
    """Load a raw SWITRS CSV export from TIMS."""
    df = pd.read_csv(path, low_memory=False)
    # Normalize column names — TIMS exports vary slightly
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def filter_bike_ped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to collisions involving bicycles or pedestrians.
    Keeps both injury and property-damage-only (PDO) collisions to see full scope.
    """
    bike = (df.get("BICYCLE_KILLED_COUNT", 0) > 0) | (df.get("BICYCLE_INJURED_COUNT", 0) > 0)
    ped = (df.get("PEDESTRIAN_KILLED_COUNT", 0) > 0) | (df.get("PEDESTRIAN_INJURED_COUNT", 0) > 0)

    # Also catch by collision type description if count columns not present
    if "TYPE_OF_COLLISION" in df.columns:
        type_col = df["TYPE_OF_COLLISION"].fillna("").str.upper()
        bike |= type_col.str.contains("BICYCLE|BIKE|CYCLIST", regex=True, na=False)
        ped |= type_col.str.contains("PEDESTRIAN|PED ", regex=True, na=False)

    return df[bike | ped].copy()


def normalize(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Normalize SWITRS columns to the pipeline's shared incident schema.
    """
    rows = []
    for _, row in df.iterrows():
        # Parse collision date
        col_date = None
        for date_col in ("COLLISION_DATE", "ACCIDENT_DATE"):
            if date_col in row and pd.notna(row[date_col]):
                try:
                    col_date = pd.to_datetime(row[date_col]).date().isoformat()
                except Exception:
                    pass
                break

        # Build location string from road columns
        road1 = str(row.get("PRIMARY_RD", "") or "").strip()
        road2 = str(row.get("SECONDARY_RD", "") or "").strip()
        if road1 and road2:
            location = f"{road1} at {road2}"
        else:
            location = road1 or road2 or "unknown"

        involves_bicycle = bool(
            (row.get("BICYCLE_KILLED_COUNT", 0) or 0) > 0
            or (row.get("BICYCLE_INJURED_COUNT", 0) or 0) > 0
        )
        involves_pedestrian = bool(
            (row.get("PEDESTRIAN_KILLED_COUNT", 0) or 0) > 0
            or (row.get("PEDESTRIAN_INJURED_COUNT", 0) or 0) > 0
        )

        severity = row.get("COLLISION_SEVERITY", None)
        injury = severity in INJURY_SEVERITIES if severity is not None else None

        if involves_bicycle and involves_pedestrian:
            incident_type = "traffic_collision"
        elif involves_bicycle:
            incident_type = "bicycle_collision"
        elif involves_pedestrian:
            incident_type = "pedestrian_collision"
        else:
            incident_type = "traffic_collision"

        rows.append({
            "source": "switrs",
            "city": city,
            "collision_date": col_date,
            "incident_type": incident_type,
            "involves_bicycle": involves_bicycle,
            "involves_pedestrian": involves_pedestrian,
            "injuries_mentioned": injury,
            "location": location,
            "switrs_case_id": row.get("CASE_ID", row.get("CASEID", None)),
            "collision_severity": severity,
        })

    return pd.DataFrame(rows)


def load_city_switrs(
    city: str,
    start: date | None = None,
    end: date | None = None,
    data_dir: Path = Path("data"),
) -> pd.DataFrame:
    """
    Load all SWITRS CSV files for a city, filter to bike/ped, normalize,
    and optionally filter to a date range.
    """
    switrs_dir = data_dir / "switrs" / city
    if not switrs_dir.exists():
        click.echo(f"No SWITRS data at {switrs_dir}")
        click.echo(
            "Export data from tims.berkeley.edu → Data → Collisions, "
            f"filter by jurisdiction '{_load_city_config(city)['switrs_jurisdiction']}', "
            f"save CSV to {switrs_dir}/"
        )
        return pd.DataFrame()

    csv_files = list(switrs_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for csv_path in csv_files:
        raw = load_switrs_csv(csv_path)
        filtered = filter_bike_ped(raw)
        normalized = normalize(filtered, city)
        frames.append(normalized)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df.empty:
        return df

    # Apply date filter
    if start or end:
        df["collision_date"] = pd.to_datetime(df["collision_date"], errors="coerce")
        if start:
            df = df[df["collision_date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["collision_date"] <= pd.Timestamp(end)]

    return df.sort_values("collision_date")


@click.command()
@click.option("--city", required=True)
@click.option("--start", "start_str", default=None, help="YYYY-MM-DD")
@click.option("--end", "end_str", default=None, help="YYYY-MM-DD")
@click.option("--data-dir", default="data", show_default=True)
def main(city, start_str, end_str, data_dir):
    """Load and summarize SWITRS data for a city."""
    start = date.fromisoformat(start_str) if start_str else None
    end = date.fromisoformat(end_str) if end_str else None
    df = load_city_switrs(city, start=start, end=end, data_dir=Path(data_dir))
    if df.empty:
        click.echo("No SWITRS records found.")
        return
    click.echo(f"{len(df)} bike/ped collision records")
    click.echo(df[["collision_date", "incident_type", "location", "injuries_mentioned"]].to_string(index=False))


if __name__ == "__main__":
    main()
