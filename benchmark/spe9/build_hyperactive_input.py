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

* ``cluster``: one cluster per well (``cluster = well name``), same
  reasoning as the Egg Model adapter -- SPE9's 26 wells have no vendored pad
  grouping, so each is its own single-well cluster and crew move time is a
  pure function of real (I, J) distance.
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


def _stb_to_tpd(stb_per_day: float, density_t_per_m3: float) -> float:
    return stb_per_day * STB_TO_M3 * density_t_per_m3


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
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

    well_rows = []
    coord_rows = []
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
        well_rows.append({
            "name": name,
            "cluster": name,
            "field": "spe9",
            "layer": "spe9",
            "well_type": "ГС",
            "oil_rate": round(oil_rate, 4),
            "liq_rate": round(liq_rate, 4),
            "length": round(length_m, 4),
            "init_entry_date": INIT_ENTRY_DATE,
        })
        coord_rows.append({
            "cluster": name,
            "x": round((i - 0.5) * CELL_SIZE_FT * FT_TO_M, 4),
            "y": round((j - 0.5) * CELL_SIZE_FT * FT_TO_M, 4),
        })

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
    print(f"wells: {len(wells)} rows -> {wells_path}")
    print(f"coordinates: {len(coordinates)} rows -> {coords_path}")
    print(wells.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
