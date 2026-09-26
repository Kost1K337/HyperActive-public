#!/usr/bin/env python3
"""SPE10 Model 2 -> real HyperActive input format (wells.xlsx + coordinates.xlsx).

This is the real-pipeline adapter for the SPE10 dataset (US-F6). Unlike the
Egg Model / SPE9 adapters, there is no earlier proxy-objective sibling for
SPE10 in this repo -- this is the first and only adapter for this benchmark,
written directly against the real HyperActive schema documented in
``docs/data-format.md`` so it can be run through the actual planner (greedy
``PlanBuilder`` and the deployed ``models/bc39`` MaskedCDQN checkpoint).

Reuses the same Eclipse-deck parsing helpers as the Egg Model / SPE9
adapters (``benchmark/common/eclipse_parse.py``).

WELL COUNT (checked from the real parsed WELSPECS, not assumed): SPE10
Model 2's grid is huge (60 x 220 x 85 = 1,122,000 cells), but ``SCHEDULE``
only defines **5 wells**: 1 injector (``INJ``, centre of the field, (I,J) =
(30, 110)) and 4 producers (``P1``..``P4``, at the four grid corners: (1,1),
(60,1), (60,220), (1,220)). This is the well-known SPE10 "quarter five-spot
at each corner + one injector in the middle" waterflood pattern -- SPE10 is
a permeability-heterogeneity benchmark, not a well-count benchmark, so a
handful of wells sampling a very fine, very heterogeneous permeability
field is exactly what the deck defines. (COMPDAT confirms this: each well
opens the full K=1..85 column, and WELLDIMS declares a hard cap of
``5`` wells, matching the 5 found.)

MAPPING CHOICES (documented honestly, same disclosure discipline as the
Egg Model / SPE9 adapters):

* ``cluster``: real spatial-proximity clustering from the start (US-F6
  requirement -- do not repeat the US-F2 one-cluster-per-well mistake).
  Real (I, J) positions converted to metres with the deck's real cell size
  (``DX = 20 ft``, ``DY = 10 ft``, FIELD units, ``SPE10_MODEL2.DATA``) give:

  | well | (I, J)     | x (m)  | y (m)  | local PERMX (mD, mean K=1..85) |
  |------|-----------|-------:|-------:|--------------------------------:|
  | INJ  | (30, 110) | 179.83 | 333.76 | 424.70 |
  | P1   | (1, 1)    |   3.05 |   1.52 | 169.24 |
  | P2   | (60, 1)   | 362.71 |   1.52 | 122.35 |
  | P3   | (60, 220) | 362.71 | 669.04 | 271.18 |
  | P4   | (1, 220)  |   3.05 | 669.04 | 139.15 |

  Pairwise distances range from **359.66 m** (P1-P2 and P3-P4, the two
  well pairs sharing a field edge) up to 758.24 m (the two diagonal
  producer pairs); INJ sits 376-382 m from every producer. This means the
  *closest* two real wells in this deck are still 360 m apart -- these 5
  wells are genuinely, physically isolated across a 366 m x 671 m field,
  not a tightly-packed pad layout like Egg Model or SPE9's diagonal chain.
  ``CLUSTER_BIN_M = 150`` m was chosen (same ``_bin_cluster`` fixed-size
  spatial-binning method as the Egg Model / SPE9 adapters) specifically
  because it is comfortably below the real 359.66 m minimum nearest-
  neighbour gap -- guaranteeing no two genuinely distant wells are ever
  merged into one bin, while remaining a plausible single-cluster radius
  at this field's own scale (150 m is under half the field's shorter side,
  366 m). Applying this threshold to real positions naturally yields
  **5 singleton clusters (one well per cluster)** -- this is NOT a repeat
  of the US-F2 mistake: there, one-cluster-per-well was a mapping shortcut
  applied to wells that were genuinely close together (Egg Model/SPE9),
  artificially triggering the 90-day inter-cluster rig-move floor on every
  move. Here, one-cluster-per-well is what the *real* spatial-binning
  method above correctly produces for wells that really are >= 360 m
  apart -- the threshold was picked from real nearest-neighbour distances
  before running the pipeline or looking at any NPV outcome, and it was
  not revisited afterwards. No one-cluster-per-well "history" file is
  produced for SPE10 since there is no distinct alternative mapping to
  compare against here (spatial clustering IS the one-cluster-per-well
  result for this well layout).
* ``field`` = ``"spe10"``, ``layer`` = ``"spe10_model2"`` (free text).
* ``well_type`` = ``"ГС"`` for every well (same reasoning as Egg Model/SPE9:
  no fracturing/stimulation keyword in the deck, so the closest honest fit
  among HyperActive's task-chain vocabulary is drilling-only).
* ``x``/``y``: real (I, J) grid position from WELSPECS converted to metres
  with the deck's real, vendored cell size (``DX = 20 ft``, ``DY = 10 ft``,
  ``SPE10_MODEL2.DATA``: ``DX``/``DY`` = ``1122000*20`` / ``1122000*10``),
  cell-centre convention: ``x = (I - 0.5) * 20 * 0.3048``,
  ``y = (J - 0.5) * 10 * 0.3048``.
* ``length``: the real, deck-derived completion interval. COMPDAT opens
  K=1..85 (the full column) for every well, and ``DZ = 2 ft`` per layer
  (``SPE10_MODEL2.DATA``: ``DZ`` = ``1122000*2``), so
  ``length = 85 * 2 ft = 170 ft = 51.816 m`` for all 5 wells.
* ``oil_rate`` / ``liq_rate`` (t/day):
  - Injector (``INJ``): the deck's real ``WCONINJE`` target IS a rate
    (``'RESV' 1* 5000 10000`` -- reservoir-volume-rate control, target
    5000 rb/day, item 7 is the BHP-limit fallback), so this is a REAL
    target, not a proxy: ``liq_rate = 5000 rb/day`` converted to t/day
    with the deck's real water density (``DENSITY`` keyword FIELD units:
    64 lb/ft3 = 1.025181 t/m3) and 1 rb = 0.1589873 m3 (same constant used
    by the Egg Model/SPE9 adapters for STB): ``liq_rate = 5000 *
    0.1589873 * 1.025181 = 815.14`` t/day. ``oil_rate = 0`` (no oil
    production; same injector/producer economic-model mismatch disclosure
    as the Egg Model/SPE9 adapters -- see below).
  - Producers (``P1``..``P4``): the deck's ``WCONPROD`` is BHP-controlled
    (``'OPEN' 'BHP' 5* 4000``) with no rate target at all -- same situation
    as the Egg Model's producers. As there, a local-permeability-derived
    PROXY oil rate is used, with the SAME methodology and ceiling as the
    Egg Model adapter (``PRODUCER_OIL_RATE_CEILING_TPD = 55`` t/day -- kept
    identical across adapters for consistency, not re-tuned per benchmark):
    ``oil_rate = 55 * clamp(local_permeability / 300, 0, 1)`` t/day (mean
    real ``PERMX`` over the well's full K=1..85 completed column, from the
    deck's own vendored ``SPE10MODEL2_PERM.INC``). The 300 mD normaliser is
    chosen (following the Egg Model adapter's reasoning) because the 4
    producers' real local permeabilities span 122.35-271.18 mD in this
    deck, so 300 keeps every producer's ratio in the linear 0.41-0.90
    range instead of saturating at 1.0 -- chosen from the real permeability
    values above, before running the pipeline. ``liq_rate = oil_rate *
    1.05`` (an assumed 5% water cut -- NOT measured, same assumption as
    the Egg Model/SPE9 adapters; SPE10 Model 2 has no offline simulator
    run available to measure a real water cut).

PROXY OBJECTIVE / SCHEMA-FIT DISCLOSURE: SPE10's injector rate is a real
deck target (unlike Egg Model's BHP-controlled producers, this one target
is directly usable); the 4 producers' oil rates are a permeability-derived
proxy (no rate target exists in the deck for them at all -- they are pure
BHP-controlled). The single injector's liquid volume is a pure lift cost
with no oil revenue in HyperActive's economics, so it will always show
negative well NPV in this evaluator -- same disclosure as the Egg
Model/SPE9 adapters: this reflects HyperActive's producer-only economic
model, not a judgement on injection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eclipse_parse import parse_numeric_array, parse_well_records, read_deck  # noqa: E402

RAW_ROOT = Path(__file__).resolve().parent / "raw"
OUT_DIR = Path(__file__).resolve().parent / "hyperactive_input"

GRID_NX, GRID_NY, GRID_NZ = 60, 220, 85
FT_TO_M = 0.3048
RB_TO_M3 = 0.1589873  # 1 barrel = 0.1589873 m3 (same constant as the STB conversion elsewhere)

CELL_SIZE_X_FT = 20.0  # DX, SPE10_MODEL2.DATA
CELL_SIZE_Y_FT = 10.0  # DY, SPE10_MODEL2.DATA
LAYER_THICKNESS_FT = 2.0  # DZ, SPE10_MODEL2.DATA
COMPLETED_LAYERS = 85  # COMPDAT: K1=1, K2=85 for every well
WELL_LENGTH_M = COMPLETED_LAYERS * LAYER_THICKNESS_FT * FT_TO_M  # 51.816 m

OIL_DENSITY_T_PER_M3 = 53.0 * 0.0160185  # DENSITY keyword: 53 lb/ft3 oil (FIELD units)
WATER_DENSITY_T_PER_M3 = 64.0 * 0.0160185  # DENSITY keyword: 64 lb/ft3 water

INJECTOR_RATE_RB_PER_DAY = 5000.0  # WCONINJE 'RESV' target, well INJ
PRODUCER_OIL_RATE_CEILING_TPD = 55.0  # same proxy ceiling as the Egg Model adapter
PRODUCER_PERM_NORMALIZER_MD = 300.0  # chosen from this deck's real 122-271 mD producer range
PRODUCER_WATER_CUT_ASSUMPTION = 0.05  # assumed, not measured

INIT_ENTRY_DATE = "2026-01-01"  # all 5 wells assumed simultaneously ready (candidate set)

CLUSTER_BIN_M = 150.0  # spatial-binning threshold, see module docstring (US-F6)


def _bin_cluster(positions: dict[str, tuple[int, int]], cell_size_x_m: float, cell_size_y_m: float, bin_m: float) -> dict[str, str]:
    """Group wells into clusters by fixed-size spatial binning on (I, J).

    Same method as the Egg Model/SPE9 adapters' ``_bin_cluster`` -- see
    module docstring for why binning was used and how ``bin_m`` was chosen.
    Unlike those adapters, SPE10's cell is not square (DX=20ft, DY=10ft),
    so the bin is sized independently per axis.
    """
    n_cells_x = max(1, round(bin_m / cell_size_x_m))
    n_cells_y = max(1, round(bin_m / cell_size_y_m))
    bin_of: dict[tuple[int, int], list[str]] = {}
    for name, (i, j) in sorted(positions.items()):
        key = ((i - 1) // n_cells_x, (j - 1) // n_cells_y)
        bin_of.setdefault(key, []).append(name)
    cluster_of: dict[str, str] = {}
    for (bi, bj), names in sorted(bin_of.items()):
        cluster_id = f"SPE10_C{bi:02d}_{bj:02d}"
        for name in names:
            cluster_of[name] = cluster_id
    return cluster_of


def _perm_array() -> list[float]:
    text = read_deck(RAW_ROOT / "SPE10MODEL2_PERM.INC")
    values = parse_numeric_array(text, "PERMX")
    expected = GRID_NX * GRID_NY * GRID_NZ
    if len(values) != expected:
        raise ValueError(f"expected {expected} PERMX values, got {len(values)}")
    return values


def _local_permeability(perm: list[float], i: int, j: int, k1: int, k2: int) -> float:
    total = 0.0
    for k in range(k1, k2 + 1):
        idx = (k - 1) * GRID_NX * GRID_NY + (j - 1) * GRID_NX + (i - 1)
        total += perm[idx]
    return total / (k2 - k1 + 1)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (wells, coordinates) using real spatial-proximity clustering."""
    text = read_deck(RAW_ROOT / "SPE10_MODEL2.DATA")

    positions: dict[str, tuple[int, int]] = {}
    for record in parse_well_records(text, "WELSPECS"):
        name, _group, i_str, j_str = record[0], record[1], record[2], record[3]
        positions[name] = (int(i_str), int(j_str))

    completions: dict[str, tuple[int, int]] = {}
    for record in parse_well_records(text, "COMPDAT"):
        name, _i, _j, k1_str, k2_str = record[0], record[1], record[2], record[3], record[4]
        completions[name] = (int(k1_str), int(k2_str))

    perm = _perm_array()
    cluster_of = _bin_cluster(positions, CELL_SIZE_X_FT * FT_TO_M, CELL_SIZE_Y_FT * FT_TO_M, CLUSTER_BIN_M)

    injector_liq_tpd = INJECTOR_RATE_RB_PER_DAY * RB_TO_M3 * WATER_DENSITY_T_PER_M3

    well_rows = []
    xy_of: dict[str, tuple[float, float]] = {}
    for name, (i, j) in sorted(positions.items()):
        is_injector = name == "INJ"
        k1, k2 = completions.get(name, (1, 1))
        x = round((i - 0.5) * CELL_SIZE_X_FT * FT_TO_M, 4)
        y = round((j - 0.5) * CELL_SIZE_Y_FT * FT_TO_M, 4)
        xy_of[name] = (x, y)
        if is_injector:
            oil_rate = 0.0
            liq_rate = injector_liq_tpd
        else:
            local_perm = _local_permeability(perm, i, j, k1, k2)
            perm_ratio = min(1.0, max(0.0, local_perm / PRODUCER_PERM_NORMALIZER_MD))
            oil_rate = PRODUCER_OIL_RATE_CEILING_TPD * perm_ratio
            liq_rate = oil_rate * (1.0 + PRODUCER_WATER_CUT_ASSUMPTION)
        well_rows.append(
            {
                "name": name,
                "cluster": cluster_of[name],
                "field": "spe10",
                "layer": "spe10_model2",
                "well_type": "ГС",
                "oil_rate": round(oil_rate, 4),
                "liq_rate": round(liq_rate, 4),
                "length": round(WELL_LENGTH_M, 4),
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
