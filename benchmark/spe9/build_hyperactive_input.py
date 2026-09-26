#!/usr/bin/env python3
"""SPE9 -> real HyperActive input format (wells.xlsx + coordinates.xlsx).

This is the SECOND, real-pipeline adapter for the SPE9 dataset (US-F2). It is
a sibling of, not a replacement for, ``adapter.py``: ``adapter.py`` feeds the
proxy-objective ``benchmark/common/sequencing_env.py`` evaluator used by the
earlier AttentionDuelingQuantileQ work (kept for history, see README.md
Section A). This script instead produces the real ``wells.xlsx`` /
``coordinates.xlsx`` schema documented in ``docs/data-format.md`` so the
dataset can be run through the actual HyperActive planner (greedy
``PlanBuilder`` and the deployed ``models/bc39`` MaskedCDQN checkpoint) --
see README.md Section B.

Reuses the same Eclipse-deck parsing helpers as ``adapter.py``
(``benchmark/common/eclipse_parse.py``); only the OUTPUT schema and the
value/duration modelling differ.

MAPPING CHOICES (documented honestly, same disclosure discipline as
``adapter.py``):

* ``cluster`` (US-F4, current default): SPE9's 26 wells have no vendored pad
  grouping, so clusters are derived from the wells' real (I, J) grid
  positions using the same fixed-size spatial binning as the Egg Model
  adapter -- each well's (I, J) is quantized into a square bin of
  ``CLUSTER_BIN_M = 700`` m (``n = round(700 / 91.44) = 8`` grid cells per
  side, since ``DX = DY = 300 ft = 91.44`` m), and wells sharing a bin
  become one cluster. Binning (fixed-size grid quantization), not
  nearest-neighbour agglomerative/single-linkage clustering, was used
  deliberately: SPE9's 25 producers sit on a single connected diagonal
  chain across the field (each well's nearest real neighbour is only
  1.4-4.1 grid cells away, with no natural gap to cut at), so any
  single-linkage threshold above ~4.1 cells chains the *entire* producer
  set into one degenerate cluster -- exactly the "all-in-one" failure mode
  this task explicitly asks to avoid. Fixed-size binning has no chaining
  effect because bins are bounded squares, not a walk along nearest-neighbour
  edges. 700 m was chosen after trying smaller bins that were still close to
  one-per-well (300 m: 22 clusters, mostly singletons, mean 1.18
  wells/cluster; 500 m: 16 clusters, mean 1.62) -- 700 m gives 7 clusters of
  size [2, 2, 2, 4, 4, 6, 6] (mean 3.7 wells/cluster, no singleton cluster,
  no cluster holding more than 6/26 wells), which best matches the "roughly
  2-5 wells per cluster on average, not 1, not all-in-one" target among the
  thresholds tried. See ``_bin_cluster`` (shared logic with the Egg Model
  adapter, duplicated here rather than factored into ``common/`` to keep
  each adapter self-contained per the existing pattern in this file).
  Cluster coordinates for the movement model use each cluster's centroid
  (mean x/y of its member wells). The prior one-cluster-per-well mapping is
  still produced, for comparison, in ``wells_one_cluster_per_well.xlsx`` /
  ``coordinates_one_cluster_per_well.xlsx`` (see ``main()``).
* ``field`` = ``"spe9"``, ``layer`` = ``"spe9"`` (free text).
* ``well_type`` = ``"ГС"`` for every well (same reasoning as the Egg Model
  adapter: no fracturing/stimulation program in the deck, HyperActive has no
  vertical-well-specific code with an independent cost entry that is not
  also implicitly a horizontal code, so ``"ГС"`` is the closest honest fit
  for a drilling-only well type).
* ``x``/``y``: real (I, J) grid position from WELSPECS converted to metres
  using the deck's real, vendored cell size ``DX = DY = 300 ft``
  (``SPE9.DATA``: ``DX``/``DY`` = ``9000*300``), cell-centre convention:
  ``x = (I - 0.5) * 300 * 0.3048``, ``y = (J - 0.5) * 300 * 0.3048``.
* ``length``: the real, deck-derived completion interval thickness (NOT the
  true total measured wellbore depth -- see the same disclosure as the Egg
  Model adapter). ``DZ = 20 ft`` per layer for every layer (``SPE9.DATA``:
  ``600*20``); COMPDAT gives each well's real K1:K2 completion range:
  producers are completed K=2..4 (3 layers = 60 ft = 18.29 m), the single
  injector INJE1 is completed K=11..15 (5 layers = 100 ft = 30.48 m).
* ``oil_rate`` / ``liq_rate`` (t/day): SPE9's schedule genuinely re-issues
  WCONPROD three times (a real, dated rate schedule, not a placement-only
  deck, per Killough's paper as documented in the deck's own comments):
  day 0-300 target 1500 STB/d oil per producer, day 300-360 target 100 STB/d,
  day 360-900 target 1500 STB/d again. The real HyperActive schema has a
  single ``oil_rate`` (an *initial* rate fed into
  ``ArpsDeclineProductionProfile``, docs/data-format.md), so this adapter
  uses the deck's real INITIAL target (day-0 phase, 1500 STB/d) as
  ``oil_rate`` -- the later rate-cut/recovery phases are a genuine part of
  the source deck but are NOT representable in HyperActive's schema (its
  Arps decline models one starting rate that declines monotonically; it has
  no mechanism for a scheduled rate cut and recovery). This loses real
  information from the deck; it is not a proxy but a real documented
  schema-fit limitation.
  - Unit conversion uses the deck's own real fluid densities (``DENSITY``
    keyword: oil 44.9856 lb/ft3 = 0.7206 g/cc = 720.6 kg/m3; water 63.0210
    lb/ft3 = 1.0095 g/cc = 1009.5 kg/m3) and 1 STB = 0.1589873 m3:
    ``t/day = STB/d * 0.1589873 * density_t_per_m3``.
  - Producers: ``oil_rate = 1500 STB/d -> t/day`` (real target, converted
    units). ``liq_rate``: the deck has no separate liquid-rate target for
    producers (oil-rate-controlled with a min BHP constraint), so an assumed
    5% water cut is applied on top of the real oil rate (PROXY, not
    measured, same discipline as the Egg Model adapter):
    ``liq_rate = oil_rate + 0.05 * oil_rate * (water_density / oil_density)``
    (converting the assumed extra water volume through water density).
  - Injector (INJE1): ``liq_rate = 5000 STBW/d -> t/day`` (real WCONINJE
    target, converted with water density), ``oil_rate = 0`` (no oil
    production; same injector/producer economic-model mismatch disclosure as
    the Egg Model adapter applies here -- see below).

PROXY OBJECTIVE / SCHEMA-FIT DISCLOSURE: unlike the Egg Model, SPE9's
producer rates ARE the deck's real target rates (not a permeability-derived
guess) -- only the day-300 rate cut/recovery phase and the producer water
cut are not representable/measured. The single injector's liquid volume is a
pure lift cost with no oil revenue in HyperActive's economics, so it will
always show negative NPV in this evaluator; as with the Egg Model, this
reflects HyperActive's producer-only economic model, not a judgement on
injection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eclipse_parse import parse_well_records, read_deck  # noqa: E402

RAW_ROOT = Path(__file__).resolve().parent / "raw"
OUT_DIR = Path(__file__).resolve().parent / "hyperactive_input"

FT_TO_M = 0.3048
STB_TO_M3 = 0.1589873
CELL_SIZE_FT = 300.0  # DX = DY = 300 ft, SPE9.DATA
LAYER_THICKNESS_FT = 20.0  # DZ = 20 ft per layer, SPE9.DATA

OIL_DENSITY_T_PER_M3 = 0.7206  # DENSITY keyword: 44.9856 lb/ft3 oil
WATER_DENSITY_T_PER_M3 = 1.0095  # DENSITY keyword: 63.0210 lb/ft3 water

PRODUCER_INITIAL_OIL_RATE_STB = 1500.0  # WCONPROD day-0 phase, all producers
INJECTOR_RATE_STB = 5000.0  # WCONINJE, INJE1
PRODUCER_WATER_CUT_ASSUMPTION = 0.05  # assumed, not measured

INIT_ENTRY_DATE = "2026-01-01"  # all 26 wells assumed simultaneously ready (candidate set)

CLUSTER_BIN_M = 700.0  # spatial-binning threshold, see module docstring (US-F4)


def _stb_to_tpd(stb_per_day: float, density_t_per_m3: float) -> float:
    return stb_per_day * STB_TO_M3 * density_t_per_m3


def _bin_cluster(positions: dict[str, tuple[int, int]], cell_size_m: float, bin_m: float) -> dict[str, str]:
    """Group wells into clusters by fixed-size spatial binning on (I, J).

    Same method as the Egg Model adapter's ``_bin_cluster`` -- see module
    docstring for why binning (not nearest-neighbour agglomerative
    clustering) was used here, and how ``bin_m`` was chosen.
    """
    n_cells = max(1, round(bin_m / cell_size_m))
    bin_of: dict[tuple[int, int], list[str]] = {}
    for name, (i, j) in sorted(positions.items()):
        key = ((i - 1) // n_cells, (j - 1) // n_cells)
        bin_of.setdefault(key, []).append(name)
    cluster_of: dict[str, str] = {}
    for (bi, bj), names in sorted(bin_of.items()):
        cluster_id = f"SPE9_C{bi:02d}_{bj:02d}"
        for name in names:
            cluster_of[name] = cluster_id
    return cluster_of


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (wells, coordinates, wells_one_cluster_per_well, coordinates_one_cluster_per_well).

    The first pair uses the new spatial-binning cluster assignment (US-F4,
    current default); the second pair reproduces the original
    one-cluster-per-well mapping (US-F2) for comparison.
    """
    text = read_deck(RAW_ROOT / "SPE9.DATA")

    positions: dict[str, tuple[int, int]] = {}
    for record in parse_well_records(text, "WELSPECS"):
        name, _group, i_str, j_str = record[0], record[1], record[2], record[3]
        positions[name] = (int(i_str), int(j_str))

    completions: dict[str, tuple[int, int]] = {}
    for record in parse_well_records(text, "COMPDAT"):
        name, _i, _j, k1_str, k2_str = record[0], record[1], record[2], record[3], record[4]
        completions[name] = (int(k1_str), int(k2_str))

    producer_oil_tpd = _stb_to_tpd(PRODUCER_INITIAL_OIL_RATE_STB, OIL_DENSITY_T_PER_M3)
    producer_water_tpd = PRODUCER_WATER_CUT_ASSUMPTION * _stb_to_tpd(
        PRODUCER_INITIAL_OIL_RATE_STB, WATER_DENSITY_T_PER_M3
    )
    injector_liq_tpd = _stb_to_tpd(INJECTOR_RATE_STB, WATER_DENSITY_T_PER_M3)

    cluster_of = _bin_cluster(positions, CELL_SIZE_FT * FT_TO_M, CLUSTER_BIN_M)

    well_rows = []
    well_rows_flat = []
    xy_of: dict[str, tuple[float, float]] = {}
    for name, (i, j) in sorted(positions.items()):
        is_injector = name.startswith("INJE")
        k1, k2 = completions.get(name, (1, 1))
        length_m = (k2 - k1 + 1) * LAYER_THICKNESS_FT * FT_TO_M
        if is_injector:
            oil_rate = 0.0
            liq_rate = injector_liq_tpd
        else:
            oil_rate = producer_oil_tpd
            liq_rate = producer_oil_tpd + producer_water_tpd
        x = round((i - 0.5) * CELL_SIZE_FT * FT_TO_M, 4)
        y = round((j - 0.5) * CELL_SIZE_FT * FT_TO_M, 4)
        xy_of[name] = (x, y)
        base_row = {
            "name": name,
            "field": "spe9",
            "layer": "spe9",
            "well_type": "ГС",
            "oil_rate": round(oil_rate, 4),
            "liq_rate": round(liq_rate, 4),
            "length": round(length_m, 4),
            "init_entry_date": INIT_ENTRY_DATE,
        }
        well_rows.append({**base_row, "cluster": cluster_of[name]})
        well_rows_flat.append({**base_row, "cluster": name})

    members_by_cluster: dict[str, list[str]] = {}
    for name, cid in cluster_of.items():
        members_by_cluster.setdefault(cid, []).append(name)
    coord_rows = []
    for cid, members in sorted(members_by_cluster.items()):
        cx = sum(xy_of[m][0] for m in members) / len(members)
        cy = sum(xy_of[m][1] for m in members) / len(members)
        coord_rows.append({"cluster": cid, "x": round(cx, 4), "y": round(cy, 4)})

    coord_rows_flat = [
        {"cluster": name, "x": x, "y": y} for name, (x, y) in sorted(xy_of.items())
    ]

    wells = pd.DataFrame(well_rows)
    coordinates = pd.DataFrame(coord_rows)
    wells_flat = pd.DataFrame(well_rows_flat)
    coordinates_flat = pd.DataFrame(coord_rows_flat)
    return wells, coordinates, wells_flat, coordinates_flat


def main() -> int:
    wells, coordinates, wells_flat, coordinates_flat = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wells_path = OUT_DIR / "wells.xlsx"
    coords_path = OUT_DIR / "coordinates.xlsx"
    wells.to_excel(wells_path, index=False)
    coordinates.to_excel(coords_path, index=False)
    print(f"wells (spatially binned clusters, default): {len(wells)} rows -> {wells_path}")
    print(f"coordinates: {len(coordinates)} rows ({coordinates['cluster'].nunique()} clusters) -> {coords_path}")
    print(wells.to_string(index=False))

    flat_wells_path = OUT_DIR / "wells_one_cluster_per_well.xlsx"
    flat_coords_path = OUT_DIR / "coordinates_one_cluster_per_well.xlsx"
    wells_flat.to_excel(flat_wells_path, index=False)
    coordinates_flat.to_excel(flat_coords_path, index=False)
    print(f"wells (one cluster per well, US-F2 history): {len(wells_flat)} rows -> {flat_wells_path}")
    print(f"coordinates (one cluster per well, US-F2 history): {len(coordinates_flat)} rows -> {flat_coords_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
