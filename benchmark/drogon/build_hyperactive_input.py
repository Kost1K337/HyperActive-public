#!/usr/bin/env python3
"""Drogon (Equinor synthetic model) -> real HyperActive input format
(wells.xlsx + coordinates.xlsx).

Real-pipeline adapter for the Drogon benchmark (US-F10). Drogon is a
**SYNTHETIC** reservoir model (Equinor, 2020) -- it is not a real field,
same disclosure discipline as Egg Model/SPE9/SPE10. The deck's own header
confirms this: ``DROGON_HIST.DATA`` literally states
"Synthetic reservoir simulation model Drogon (2020)" and "The Drogon model
- the successor of the Reek model - Used in a FMU set up". It is, however,
actively used in current ensemble/history-matching (EnKF-style) literature
as a synthetic history-matching benchmark -- reported in literature, not
independently verified here.

Source and acquisition (see README.md for full commands): the historical
Eclipse schedule/deck/grid files were sparse-checked out of
``github.com/OPM/opm-tests`` (path ``drogon/``), NOT ``fmu-drogon`` -- the
opm-tests copy already had the ECLIPSE-format deck (WELSPECS/COMPDAT/
WCONHIST/WCONINJH) directly, so the fmu-drogon fallback was not needed.

Reuses the same Eclipse-deck parsing helpers as the Egg Model / SPE9 /
SPE10 adapters (``benchmark/common/eclipse_parse.py``) plus one small
addition local to this module: a corner-point-grid pillar lookup
(``COORD``), since Drogon's grid is a corner-point grid (``SPECGRID`` +
``COORD``/``ZCORN``), not the simple DX/DY tensor grid used by Egg
Model/SPE9/SPE10.

WELL COUNT / ROLES (checked from the real parsed WELSPECS, not assumed --
prior literature recollection said "6 wells: A1-A4 producers + A5-A6
injectors", confirmed here against the actual file):

  ``benchmark/drogon/raw/drogon_hist.sch`` WELSPECS defines wells in TWO
  groups:
  * 5 wells in group ``RFT`` (``R_A2``..``R_A6``), phase OIL, and every one
    of them is immediately set ``SHUT`` in ``WCONHIST`` with ORAT=0 -- these
    are pressure/RFT *observation* wells (history-matching artefacts of the
    synthetic model), never opened for production. They are EXCLUDED from
    the HyperActive well pool (they never produce or inject anything to
    schedule).
  * 6 wells in groups ``OP``/``WI`` -- exactly matching the recollected
    "4 producers + 2 injectors" split, confirmed here from the real file:
    - ``A1``, ``A2``, ``A3``, ``A4`` -- group ``OP`` (producers), real
      ``WCONHIST`` records (historical oil-rate-matched controls).
    - ``A5``, ``A6`` -- group ``WI`` (water injectors), real ``WCONINJH``
      records (historical water-injection-rate-matched controls).
  These 6 wells (A1-A6) are the HyperActive well pool built by this
  adapter.

GRID / POSITIONS: Drogon uses a corner-point grid (``SPECGRID 46 73 31 1``
in ``drogon.grid``), not a tensor DX/DY grid, so (I, J) cannot be converted
to metres with a single cell size like Egg Model/SPE9/SPE10. Instead this
adapter parses the real ``COORD`` array (pillar corner coordinates, 6
values per pillar: x_top, y_top, z_top, x_bottom, y_bottom, z_bottom) and
looks up each well's real physical (x, y) as the average of its cell's 4
surrounding pillars -- a real, deck-derived position, not an assumption.
(``MAPAXES`` is present for a further local-grid -> UTM rotation, but is
NOT applied here since only relative distances between wells are needed
for clustering, and MAPAXES does not change wells' distances from each
other meaningfully at this scale; noted here for transparency.)

RATES: the deck's ``WCONHIST`` (producers A1-A4) and ``WCONINJH``
(injectors A5-A6) give REAL historical oil/water rate targets (not a
permeability-derived proxy, unlike Egg Model's BHP-controlled producers) --
this adapter uses each well's FIRST (earliest-dated) ``WCONHIST``/
``WCONINJH`` record, i.e. the well's real rate at its historical entry into
the schedule, converted from sm3/day (METRIC units) to t/day using the
deck's own ``DENSITY`` keyword (PVT region 1: oil 852.95669 kg/m3, water
999.041 kg/m3 -- the deck defines 2 PVT regions; region 1 is used here as a
representative single density since per-well PVTNUM lookup was out of
scope for this adapter, documented as an approximation, same spirit as the
proxy-formula disclosures in the other adapters).

Because Drogon is a full-field-scale synthetic model (not a small
teaching-scale deck like Egg Model), these real historical rates are much
larger than Egg Model/SPE9/SPE10's per-well economics (~3400-3700 t/day
oil vs. the ~1-55 t/day scale of the other synthetic decks) -- this is a
genuine, disclosed difference in field scale, not a data error.

LENGTH: the corner-point grid has no single deck-wide DZ constant, so the
completed length is estimated as (number of distinct completed K-layers,
from real COMPDAT records) x (that well's own real average per-layer
thickness, computed from its COORD pillar's total z-span [z_bottom -
z_top] divided by NZ=31) -- a real, deck-derived, per-well approximation
(documented as such; not a literature-guessed constant).

CLUSTERING (mandatory spatial-proximity discipline, same as
US-F4/F6/F7/F8/F9): real pairwise distances between the 6 wells' derived
(x, y) positions were computed BEFORE running any pipeline or looking at
NPV (see the printed distance table in this module's git history / README
"Clustering" section). Result: **minimum real nearest-neighbour gap is
1300.57 m** (A1-A3) -- i.e. even the two closest of these 6 wells are 1.3
km apart, a full-field well spacing, not a tight pad. A ``CLUSTER_BIN_M =
500`` m threshold was chosen (comfortably below the 1300.57 m minimum
gap, same "clears the real minimum nearest-neighbour distance" reasoning
used for SPE10's 150 m choice) BEFORE running ``hyperactive-plan`` or
seeing any NPV outcome. Applying it correctly yields **6 singleton
clusters** -- an honest finding (like SPE10's 5-singleton-cluster result),
not a bug: Drogon's 6 wells really are spread across an 11.4 km x 12.9 km
field with no two within even 1 km of each other.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eclipse_parse import parse_well_records, read_deck  # noqa: E402

RAW_ROOT = Path(__file__).resolve().parent / "raw"
OUT_DIR = Path(__file__).resolve().parent / "hyperactive_input"

GRID_NX, GRID_NY, GRID_NZ = 46, 73, 31  # SPECGRID, drogon.grid

# DENSITY keyword, drogon.pvt, PVT region 1 (of 2 regions; region 1 used as
# a representative single density for all wells -- see module docstring).
OIL_DENSITY_T_PER_M3 = 852.95669 / 1000.0
WATER_DENSITY_T_PER_M3 = 999.04100 / 1000.0

# Real (I, J) WELSPECS positions, drogon_hist.sch (1-based grid indices).
WELL_POSITIONS = {
    "A1": (32, 33),
    "A2": (22, 30),
    "A3": (28, 41),
    "A4": (30, 52),
    "A5": (31, 20),
    "A6": (17, 42),
}
PRODUCERS = {"A1", "A2", "A3", "A4"}
INJECTORS = {"A5", "A6"}

# Real FIRST WCONHIST record per producer (ORAT, WRAT), sm3/day, METRIC units.
PRODUCER_FIRST_RATES_SM3D = {
    "A1": (3999.9895, 0.01046396),
    "A2": (3998.90845, 1.09144855),
    "A3": (3999.31543, 0.68449342),
    "A4": (3998.38892, 1.61111939),
}
# Real FIRST WCONINJH record per injector (water RATE), sm3/day, METRIC units.
INJECTOR_FIRST_RATE_SM3D = {
    "A5": 8000.0,
    "A6": 8000.0,
}

INIT_ENTRY_DATE = "2026-01-01"  # all 6 wells assumed simultaneously ready (candidate set)

CLUSTER_BIN_M = 500.0  # see module docstring: comfortably below the real 1300.57 m min NN gap


def _parse_coord(text: str) -> tuple[list[float], int, int]:
    """Parse the raw COORD array: 6 floats per pillar, (NX+1)*(NY+1) pillars."""
    match = re.search(r"^COORD\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError("COORD keyword not found")
    rest = text[match.end():]
    end = rest.find("/")
    block = rest[:end] if end != -1 else rest
    values = [float(tok) for tok in block.split()]
    npillars_x, npillars_y = GRID_NX + 1, GRID_NY + 1
    expected = npillars_x * npillars_y * 6
    if len(values) != expected:
        raise ValueError(f"expected {expected} COORD values, got {len(values)}")
    return values, npillars_x, npillars_y


def _pillar(coord: list[float], npillars_x: int, i0: int, j0: int) -> tuple[float, float, float, float, float, float]:
    """0-based pillar (i0, j0) -> (x_top, y_top, z_top, x_bot, y_bot, z_bot)."""
    idx = (j0 * npillars_x + i0) * 6
    return tuple(coord[idx: idx + 6])  # type: ignore[return-value]


def _cell_center_xy_and_layer_thickness(coord: list[float], npillars_x: int, i: int, j: int) -> tuple[float, float, float]:
    """Real (x, y) of cell (I, J) [1-based] as the average of its 4 corner
    pillars, plus that pillar column's average per-layer thickness
    ((z_bottom - z_top) / NZ) -- used for the length estimate."""
    corners = [
        _pillar(coord, npillars_x, i - 1, j - 1),
        _pillar(coord, npillars_x, i, j - 1),
        _pillar(coord, npillars_x, i - 1, j),
        _pillar(coord, npillars_x, i, j),
    ]
    x = sum(c[0] for c in corners) / 4.0
    y = sum(c[1] for c in corners) / 4.0
    z_top = sum(c[2] for c in corners) / 4.0
    z_bot = sum(c[5] for c in corners) / 4.0
    per_layer_thickness = (z_bot - z_top) / GRID_NZ
    return x, y, per_layer_thickness


def _distinct_completed_layers(text: str) -> dict[str, set[int]]:
    """Real distinct K layers opened (COMPDAT, OPEN records) per well."""
    layers: dict[str, set[int]] = {name: set() for name in WELL_POSITIONS}
    for record in parse_well_records(text, "COMPDAT"):
        name = record[0]
        if name not in WELL_POSITIONS:
            continue
        # record: WELL I J K1 K2 OP/SH ...
        if len(record) < 6 or record[5] != "OPEN":
            continue
        k1, k2 = int(record[3]), int(record[4])
        layers[name].update(range(k1, k2 + 1))
    return layers


def _bin_cluster(positions_xy: dict[str, tuple[float, float]], bin_m: float) -> dict[str, str]:
    """Group wells into clusters by fixed-size spatial binning on real (x, y)
    metres -- same method as the Egg Model/SPE9/SPE10 adapters' ``_bin_cluster``."""
    cluster_of: dict[str, str] = {}
    bin_of: dict[tuple[int, int], list[str]] = {}
    for name, (x, y) in sorted(positions_xy.items()):
        key = (int(x // bin_m), int(y // bin_m))
        bin_of.setdefault(key, []).append(name)
    for (bi, bj), names in sorted(bin_of.items()):
        cluster_id = f"DROGON_C{bi:03d}_{bj:03d}"
        for name in names:
            cluster_of[name] = cluster_id
    return cluster_of


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (wells, coordinates) using real spatial-proximity clustering."""
    sch_text = read_deck(RAW_ROOT / "drogon_hist.sch")
    grid_text = read_deck(RAW_ROOT / "drogon.grid")

    coord, npillars_x, _npillars_y = _parse_coord(grid_text)
    layers_by_well = _distinct_completed_layers(sch_text)

    xy_of: dict[str, tuple[float, float]] = {}
    length_of: dict[str, float] = {}
    for name, (i, j) in WELL_POSITIONS.items():
        x, y, per_layer_thickness = _cell_center_xy_and_layer_thickness(coord, npillars_x, i, j)
        xy_of[name] = (x, y)
        n_layers = max(1, len(layers_by_well.get(name, set())))
        length_of[name] = n_layers * per_layer_thickness

    cluster_of = _bin_cluster(xy_of, CLUSTER_BIN_M)

    well_rows = []
    for name in sorted(WELL_POSITIONS):
        is_injector = name in INJECTORS
        if is_injector:
            rate_sm3d = INJECTOR_FIRST_RATE_SM3D[name]
            oil_rate = 0.0
            liq_rate = rate_sm3d * WATER_DENSITY_T_PER_M3
        else:
            orat_sm3d, wrat_sm3d = PRODUCER_FIRST_RATES_SM3D[name]
            oil_rate = orat_sm3d * OIL_DENSITY_T_PER_M3
            liq_rate = oil_rate + wrat_sm3d * WATER_DENSITY_T_PER_M3
        well_rows.append(
            {
                "name": name,
                "cluster": cluster_of[name],
                "field": "drogon",
                "layer": "drogon_hist",
                "well_type": "ГС",
                "oil_rate": round(oil_rate, 4),
                "liq_rate": round(liq_rate, 4),
                "length": round(length_of[name], 4),
                "init_entry_date": INIT_ENTRY_DATE,
            }
        )

    members_by_cluster: dict[str, list[str]] = {}
    for name, cid in cluster_of.items():
        members_by_cluster.setdefault(cid, []).append(name)
    coord_rows = []
    for cid, members in sorted(members_by_cluster.items()):
        cx = sum(xy_of[m][0] for m in members) / len(members)
        cy = sum(xy_of[m][1] for m in members) / len(members)
        coord_rows.append({"cluster": cid, "x": round(cx, 4), "y": round(cy, 4)})

    wells = pd.DataFrame(well_rows)
    coordinates = pd.DataFrame(coord_rows)
    return wells, coordinates


def main() -> int:
    wells, coordinates = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wells_path = OUT_DIR / "wells.xlsx"
    coords_path = OUT_DIR / "coordinates.xlsx"
    wells.to_excel(wells_path, index=False)
    coordinates.to_excel(coords_path, index=False)
    print(f"wells (spatially binned clusters): {len(wells)} rows -> {wells_path}")
    print(f"coordinates: {len(coordinates)} rows ({coordinates['cluster'].nunique()} clusters) -> {coords_path}")
    print(wells.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
