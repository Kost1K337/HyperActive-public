#!/usr/bin/env python3
"""Egg Model -> real HyperActive input format (wells.xlsx + coordinates.xlsx).

This is the SECOND, real-pipeline adapter for the Egg Model dataset (US-F2).
It is a sibling of, not a replacement for, ``adapter.py``: ``adapter.py``
feeds the proxy-objective ``benchmark/common/sequencing_env.py`` evaluator
used by the earlier AttentionDuelingQuantileQ work (kept for history, see
``README.md`` Section A). This script instead produces the real
``wells.xlsx`` / ``coordinates.xlsx`` schema documented in
``docs/data-format.md`` so the dataset can be run through the actual
HyperActive planner (``hyperactive-plan`` / ``benchmark/run_benchmark.py``,
greedy ``PlanBuilder`` and the deployed ``models/bc39`` MaskedCDQN
checkpoint) -- see README.md Section B.

Reuses the same Eclipse-deck parsing helpers as ``adapter.py``
(``benchmark/common/eclipse_parse.py``) since the raw vendored deck is
identical; only the OUTPUT schema and the value/duration modelling differ,
because the real HyperActive schema and economics are different from the
proxy evaluator's.

MAPPING CHOICES (documented honestly, same disclosure discipline as
``adapter.py``):

* ``cluster`` (US-F4, current default): the Egg Model archive has no
  vendored pad/cluster grouping, so clusters are derived from the wells'
  real (I, J) grid positions using fixed-size spatial binning -- each well's
  (I, J) is quantized into a square bin of ``CLUSTER_BIN_M = 250`` m
  (``n = round(250 / 8) = 31`` grid cells per side, since ``DX = DY = 8``
  m), and wells sharing a bin become one cluster. 250 m was chosen (over a
  few candidates tried: 150 m gave 8 clusters averaging 1.5 wells each,
  still close to one-per-well; 300 m gave 4 clusters but one 5-well and one
  3-cell-radius cluster reaching further than the wells' natural spacing
  would justify) because it is the smallest threshold that clears every
  well's real nearest-neighbour gap in this field (all 12 wells' nearest
  real neighbour is >= 109 m / 13.6 cells apart -- this is a wide-spaced
  5-spot-style waterflood pattern, not a padded field) while producing 4
  clusters of size [2, 3, 3, 4] (mean 3.0 wells/cluster), matching the
  "roughly 2-5 wells per cluster, not 1, not all-in-one" target. Binning
  (fixed-size grid quantization) was used instead of nearest-neighbour
  agglomerative clustering because Egg Model's producers/injectors are laid
  out with fairly uniform pairwise spacing (13.6-18.4 grid cells to nearest
  neighbour) with no natural single-linkage gap to cut at, so a chosen
  threshold is a deliberate spatial-radius choice either way; binning keeps
  cluster shapes bounded (a square of side 250 m) rather than a possible
  single-linkage chain across the whole field. See ``_bin_cluster`` below.
  Cluster coordinates for the movement model use each cluster's centroid
  (mean x/y of its member wells). The prior one-cluster-per-well mapping is
  still produced, for comparison, in ``wells_one_cluster_per_well.xlsx`` /
  ``coordinates_one_cluster_per_well.xlsx`` (see ``main()``).
* ``field`` = ``"egg_model"``, ``layer`` = ``"egg"`` (free text, not used by
  planner logic per docs/data-format.md).
* ``well_type``: HyperActive's task-chain vocabulary
  (``hyperactive.core.Task``) only has DRILLING codes (``ГС``/``ННС``/``МЗС``/
  ``БУРЕНИЕ``) and a GTM code (``ГРП``, hydraulic fracturing). The Egg Model
  deck has no fracturing/stimulation program for any well (COMPDAT opens all
  7 layers with no WELTRAJ/frac keywords), so every well -- injector or
  producer -- is mapped to ``"ГС"`` (horizontal well, drilling only). This is
  a simplification: Egg Model's wells are Cartesian vertical stacks (I, J,
  K=1..7), not literally horizontal, but ``ГС`` is HyperActive's only
  DRILLING code with a build-cost-per-metre entry that does not also imply a
  fracturing follow-up, so it is the closest honest fit.
* ``x``/``y``: real (I, J) grid position from WELSPECS, converted to metres
  using the deck's real, vendored cell size ``DX = DY = 8 m``
  (``Egg_Model_ECL.DATA``, ``DX``/``DY`` keywords: ``25200*8``) --
  ``x = (I - 0.5) * 8``, ``y = (J - 0.5) * 8`` (cell-centre convention).
* ``length``: the real, deck-derived vertical completion interval, not an
  assumed constant. COMPDAT opens K=1..7 for every well (full column) and
  ``DZ = 4 m`` per layer (``Egg_Model_ECL.DATA``: ``DZ`` = ``25200*4``), so
  ``length = 7 * 4 = 28 m``. DISCLOSURE: this is the perforated interval
  thickness, not the true total measured wellbore depth/length (the deck has
  no record of the vertical drilling distance from surface down to the
  reservoir, which a real length-based drilling cost would also include) --
  it is a real, sourced number, but a partial one.
* ``oil_rate`` / ``liq_rate`` (t/day):
  - Injectors: real WCONINJE water RATE target (``79.5`` for every
    ``INJECT*`` well, in Sm3/day per the deck's metric units) converted to
    tonnes/day with the deck's own real water density (``DENSITY`` keyword:
    1000 kg/m3): ``liq_rate = 79.5 * 1.0 = 79.5`` t/day, ``oil_rate = 0``
    (an injector has no oil production; HyperActive's economics have no
    injection-support concept, only per-well oil revenue and liquid/water
    lift cost -- see PROXY OBJECTIVE DISCLOSURE below for what this implies).
  - Producers: Egg Model producers are BHP-controlled (WCONPROD ``'OPEN'
    'BHP'``), so there is no rate target in the deck at all. As in
    ``adapter.py``, a local-permeability-derived PROXY oil rate is used:
    ``oil_rate = 55 * clamp(local_permeability / 5000, 0, 1)`` t/day (mean
    PERMX over the well's full K=1..7 column, from one of the vendored
    ``Permeability_Realizations_subset10`` fields; the 5000 mD normaliser is
    chosen because the 12 wells' real local permeabilities span ~330-4430 mD
    in this realization, so 5000 keeps all of them in the ratio's linear
    range instead of saturating at 1.0), capped at a plausible ~55 t/day
    ceiling for this reservoir's scale. ``liq_rate = oil_rate *
    1.05`` (an assumed 5% early-life water cut -- NOT measured; Egg Model's
    real water cut evolves with the simulation and is not available without
    running the simulator). THIS IS A PROXY, NOT A SIMULATED RATE.

PROXY OBJECTIVE DISCLOSURE (carried over from ``adapter.py``, restated for
the real pipeline): the vendored Egg Model archive has no simulator run, so
there is no ground-truth production/NPV signal offline. Producer oil rates
above are an analytic proxy from static permeability, not simulated
production; injector rates are the deck's real target but, run through
HyperActive's real economics, always show negative NPV (their liquid volume
is a pure lift cost with no matching oil revenue in this model) since
HyperActive was built to sequence oil-producing wells, not waterflood
support wells. Do not read greedy/RL NPV rankings against injectors as an
endorsement or rejection of injecting -- it reflects an economic model that
does not credit injection support, not a judgement on injector value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eclipse_parse import parse_numeric_array, parse_well_records, read_deck  # noqa: E402

RAW_ROOT = Path(__file__).resolve().parent / "raw"
ECLIPSE_ROOT = RAW_ROOT / "Eclipse"
OUT_DIR = Path(__file__).resolve().parent / "hyperactive_input"

GRID_NX, GRID_NY, GRID_NZ = 60, 60, 7
CELL_SIZE_M = 8.0  # DX = DY = 8 m, Egg_Model_ECL.DATA
LAYER_THICKNESS_M = 4.0  # DZ = 4 m per layer
COMPLETED_LAYERS = 7  # COMPDAT: K1=1, K2=7 for every well
WELL_LENGTH_M = LAYER_THICKNESS_M * COMPLETED_LAYERS  # 28 m, see module docstring

OIL_DENSITY_T_PER_M3 = 0.900  # DENSITY keyword: 900 kg/m3 oil (Egg_Model_ECL.DATA)
WATER_DENSITY_T_PER_M3 = 1.000  # DENSITY keyword: 1000 kg/m3 water
INJECTOR_RATE_M3_PER_DAY = 79.5  # WCONINJE RATE, every INJECT* well
PRODUCER_OIL_RATE_CEILING_TPD = 55.0  # proxy ceiling, see module docstring
PRODUCER_WATER_CUT_ASSUMPTION = 0.05  # assumed, not measured

INIT_ENTRY_DATE = "2026-01-01"  # all 12 wells assumed simultaneously ready (candidate set)

CLUSTER_BIN_M = 250.0  # spatial-binning threshold, see module docstring (US-F4)


def _bin_cluster(positions: dict[str, tuple[int, int]], cell_size_m: float, bin_m: float) -> dict[str, str]:
    """Group wells into clusters by fixed-size spatial binning on (I, J).

    Each well's grid position is quantized into a square bin of side
    ``bin_m`` metres (converted to a whole number of grid cells); wells
    landing in the same bin become one cluster. See module docstring for why
    binning (not nearest-neighbour agglomerative clustering) was used and
    how ``bin_m`` was chosen.
    """
    n_cells = max(1, round(bin_m / cell_size_m))
    bin_of: dict[tuple[int, int], list[str]] = {}
    for name, (i, j) in sorted(positions.items()):
        key = ((i - 1) // n_cells, (j - 1) // n_cells)
        bin_of.setdefault(key, []).append(name)
    cluster_of: dict[str, str] = {}
    for (bi, bj), names in sorted(bin_of.items()):
        cluster_id = f"EGG_C{bi:02d}_{bj:02d}"
        for name in names:
            cluster_of[name] = cluster_id
    return cluster_of


def _perm_array(realization: str = "PERM1_ECL.INC") -> list[float]:
    text = read_deck(RAW_ROOT / "Permeability_Realizations_subset10" / realization)
    values = parse_numeric_array(text, "PERMX")
    expected = GRID_NX * GRID_NY * GRID_NZ
    if len(values) != expected:
        raise ValueError(f"expected {expected} PERMX values, got {len(values)}")
    return values


def _local_permeability(perm: list[float], i: int, j: int) -> float:
    total = 0.0
    for k in range(1, GRID_NZ + 1):
        idx = (k - 1) * GRID_NX * GRID_NY + (j - 1) * GRID_NX + (i - 1)
        total += perm[idx]
    return total / GRID_NZ


def _well_positions() -> dict[str, tuple[int, int]]:
    text = read_deck(ECLIPSE_ROOT / "Egg_Model_ECL.DATA")
    positions: dict[str, tuple[int, int]] = {}
    for record in parse_well_records(text, "WELSPECS"):
        name, _group, i_str, j_str = record[0], record[1], record[2], record[3]
        positions[name] = (int(i_str), int(j_str))
    return positions


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (wells, coordinates, wells_one_cluster_per_well, coordinates_one_cluster_per_well).

    The first pair uses the new spatial-binning cluster assignment (US-F4,
    current default); the second pair reproduces the original
    one-cluster-per-well mapping (US-F2) for comparison.
    """
    positions = _well_positions()
    perm = _perm_array()
    cluster_of = _bin_cluster(positions, CELL_SIZE_M, CLUSTER_BIN_M)

    well_rows = []
    well_rows_flat = []
    xy_of: dict[str, tuple[float, float]] = {}
    for name, (i, j) in sorted(positions.items()):
        is_injector = name.startswith("INJECT")
        local_perm = _local_permeability(perm, i, j)
        if is_injector:
            oil_rate = 0.0
            liq_rate = INJECTOR_RATE_M3_PER_DAY * WATER_DENSITY_T_PER_M3
        else:
            perm_ratio = min(1.0, max(0.0, local_perm / 5000.0))
            oil_rate = PRODUCER_OIL_RATE_CEILING_TPD * perm_ratio
            liq_rate = oil_rate * (1.0 + PRODUCER_WATER_CUT_ASSUMPTION)
        x, y = (i - 0.5) * CELL_SIZE_M, (j - 0.5) * CELL_SIZE_M
        xy_of[name] = (x, y)
        base_row = {
            "name": name,
            "field": "egg_model",
            "layer": "egg",
            "well_type": "ГС",
            "oil_rate": round(oil_rate, 4),
            "liq_rate": round(liq_rate, 4),
            "length": WELL_LENGTH_M,
            "init_entry_date": INIT_ENTRY_DATE,
        }
        well_rows.append({**base_row, "cluster": cluster_of[name]})
        well_rows_flat.append({**base_row, "cluster": name})

    # New: cluster coordinates = centroid of member wells' (x, y).
    members_by_cluster: dict[str, list[str]] = {}
    for name, cid in cluster_of.items():
        members_by_cluster.setdefault(cid, []).append(name)
    coord_rows = []
    for cid, members in sorted(members_by_cluster.items()):
        cx = sum(xy_of[m][0] for m in members) / len(members)
        cy = sum(xy_of[m][1] for m in members) / len(members)
        coord_rows.append({"cluster": cid, "x": cx, "y": cy})

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
