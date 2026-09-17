"""Reading the well pool and cluster coordinates from CSV or Excel tables.

See ``docs/data-format.md`` for the column contract.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from datetime import time as datetime_time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from hyperactive.core import Well

# Column name -> Well field. Column order does not matter; a field is found by
# its header.
COLUMN_SYNONYMS = {
    "name": ("well", "well_name", "name"),
    "cluster": ("cluster", "pad"),
    "field": ("field",),
    "layer": ("layer", "reservoir"),
    "well_type": ("well_type",),
    "oil_rate": ("oil_rate", "oil_rate_tpd"),
    "liq_rate": ("liq_rate", "liq_rate_tpd", "liquid_rate_tpd"),
    "length": ("length", "length_m", "lateral_length_m"),
    "purpose": ("purpose",),
    "init_entry_date": ("init_entry_date", "readiness_date"),
}

REQUIRED_FIELDS = (
    "name", "cluster", "field", "layer", "well_type",
    "oil_rate", "liq_rate", "length",
)
# Must match the field order of Well: rows are built positionally.
FIELD_ORDER = (*REQUIRED_FIELDS, "purpose", "init_entry_date")

_LOOKUP = {
    synonym: canonical
    for canonical, synonyms in COLUMN_SYNONYMS.items()
    for synonym in synonyms
}

# Hour of the day at which a well becomes ready on its readiness date.
READINESS_HOUR = 8


def normalize_header(value) -> str:
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_cluster(value: Any) -> str:
    """Cluster ids may come as numbers; ``25.0`` and ``25`` must be the same cluster."""
    try:
        number_value = float(value)
        if number_value.is_integer():
            return str(int(number_value))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, header=0)
    return pd.read_csv(path)


def _column_map(columns) -> dict:
    mapping = {}
    taken = set()
    for column in columns:
        canonical = _LOOKUP.get(normalize_header(column))
        # The first occurrence wins.
        if canonical and canonical not in taken:
            mapping[column] = canonical
            taken.add(canonical)
    return mapping


def _text(value):
    """A text cell; integer-valued numbers are written without the ``.0`` suffix."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return str(number)
    text = str(value).strip()
    return text or None


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    mapping = _column_map(df.columns)
    missing = set(REQUIRED_FIELDS) - set(mapping.values())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.rename(columns=mapping)
    for optional in ("purpose", "init_entry_date"):
        if optional not in df.columns:
            df[optional] = None
    df = df[list(FIELD_ORDER)].copy()

    for column in ("oil_rate", "liq_rate", "length"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["length"] = df["length"].fillna(df["length"].mean())
    parsed_dates = pd.to_datetime(df["init_entry_date"], errors="coerce")
    df["init_entry_date"] = parsed_dates.astype(object).where(parsed_dates.notna(), None)

    for column in ("name", "cluster", "field", "layer", "well_type", "purpose"):
        df[column] = df[column].map(_text)
    df = df.dropna(subset=list(REQUIRED_FIELDS))
    return df.reset_index(drop=True)


def is_valid_well(well: Well) -> bool:
    if not str(well.name).strip() or str(well.name).strip().lower() in {"nan", "none"}:
        return False
    if not str(well.cluster).strip() or str(well.cluster).strip().lower() in {"nan", "none"}:
        return False
    if pd.isna(well.oil_rate) or pd.isna(well.liq_rate) or pd.isna(well.length):
        return False
    _ = well.tasks
    return True


def readiness_from_date(value: Any, hour: int = READINESS_HOUR) -> Optional[datetime]:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return datetime.combine(parsed.to_pydatetime().date(), datetime_time(hour=hour))


def load_wells(path: str | Path, readiness_hour: Optional[int] = READINESS_HOUR) -> tuple[list[Well], int]:
    """Load valid wells and the number of rejected rows.

    ``init_entry_date`` (or ``readiness_date``) becomes the infrastructure
    readiness of the well: at ``readiness_hour`` o'clock of that day, or at the
    exact timestamp from the table when ``readiness_hour`` is None.
    """
    frame = _prepare(read_table(path))
    wells: list[Well] = []
    rejected = 0
    for row in frame.itertuples(index=False):
        try:
            well = Well(*row)
            well.cluster = normalize_cluster(well.cluster)
            if is_valid_well(well):
                wells.append(well)
            else:
                rejected += 1
        except (TypeError, ValueError, AttributeError):
            rejected += 1
    for well in wells:
        if well.readiness_date is not None:
            continue
        if readiness_hour is None:
            well.readiness_date = well.init_entry_date
        else:
            well.readiness_date = readiness_from_date(well.init_entry_date, readiness_hour)
    return wells, rejected


def ring_coordinates(wells: list[Well], radius_m: float = 2500.0) -> list[dict[str, Any]]:
    """Clusters placed evenly on a circle; used when no coordinates are given."""
    clusters = list(dict.fromkeys(normalize_cluster(well.cluster) for well in wells))
    count = max(len(clusters), 1)
    return [
        {
            "cluster": cluster,
            "x": float(radius_m * math.cos(2.0 * math.pi * index / count)),
            "y": float(radius_m * math.sin(2.0 * math.pi * index / count)),
            "z": 0.0,
        }
        for index, cluster in enumerate(clusters)
    ]


def load_coordinates(path: str | Path | None, wells: list[Well]) -> list[dict[str, Any]]:
    """Cluster coordinates in metres (columns ``cluster, x, y, z``)."""
    if path is None or not Path(path).exists():
        return ring_coordinates(wells)
    frame = read_table(path)
    frame = frame.rename(columns={column: normalize_header(column) for column in frame.columns})
    frame = frame[["cluster", "x", "y"] + (["z"] if "z" in frame.columns else [])].copy()
    if "z" not in frame.columns:
        frame["z"] = 0.0
    for column in ("x", "y", "z"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["cluster", "x", "y"]).copy()
    frame["cluster"] = frame["cluster"].map(normalize_cluster)
    frame["z"] = frame["z"].fillna(0.0)
    return frame.to_dict(orient="records")
