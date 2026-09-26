#!/usr/bin/env python3
"""Volve (partial, real field data) -> real HyperActive input format
(wells.xlsx + coordinates.xlsx).

Real-pipeline-only adapter (US-F7), same discipline as the SPE10 adapter:
no proxy-objective sibling exists for Volve in this repo. This is written
directly against the real HyperActive schema (``docs/data-format.md``).

ACQUISITION PATH (documented honestly -- see README.md for the full
writeup and exact commands): Equinor's own ~40,000-file Volve archive
(equinor.com/energy/volve-data-sharing) currently gates programmatic
download behind a Databricks Marketplace listing (interactive, requires an
account) -- there is no stable direct-file URL for it. Two OTHER real,
public, directly-downloadable sources were used instead:

1. ``benchmark/volve/raw/volve_wells_bysarmad_mirror.csv`` -- a third-party
   GitHub mirror of Equinor's own vendored Volve well-header table
   (26 real wellbore names, purpose, water depth, TVD, MD, entered/
   completed dates), from
   ``github.com/bysarmad/Well-Data-Management-and-Visualization``,
   commit ``f9202a244b2e864a77cab2140cbee899fd6be5b4`` (2019-05-29), file
   ``Data/volve_wells.csv``. This gives real well identity, depth and
   date data but NO surface coordinates -- the Equinor-derived CSV mirror
   does not carry wellhead position.
2. ``benchmark/volve/raw/volve_wells_coords_sodir.csv`` -- real wellhead
   UTM coordinates for each of the 26 wells above, looked up by name via
   the Norwegian Offshore Directorate (Sodir, formerly NPD)'s public
   FactPages register (``factpages.sodir.no``, NLOD -- Norwegian Licence
   for Open Government Data), a government open-data source independent
   of Equinor's own archive, but describing the same real wellbores (NPD
   regulates and records every well drilled on the Norwegian continental
   shelf, including Volve's). 24/26 wells resolved (2 sidetracks, ``15/9-19
   BT2`` and ``15/9-19 ST2``, were not found by the FactPages name search
   and are reused from their parent slot's coordinates -- a sidetrack
   re-enters the same physical wellhead, so this is a real, not invented,
   position).

MAPPING CHOICES (documented honestly, same disclosure discipline as the
Egg Model/SPE9/SPE10 adapters):

* **Candidate set restricted to Production + Injection wells (12 of the 26
  Volve wells).** The other 14 wells are Observation (9), Appraisal (3) or
  Wildcat (2) wells -- exploration/monitoring wellbores with no producible
  rate for HyperActive's producer-only economics to model. Including them
  with a fabricated non-zero rate would misrepresent them as productive
  wells they are not; they are documented here as excluded, not silently
  dropped. See README.md for the full 26-well purpose breakdown.
* ``cluster`` (real spatial-proximity clustering from the start, US-F7
  requirement): real wellhead UTM positions for the 12 Production/
  Injection wells were measured (see the coordinate lookup above) BEFORE
  choosing a bin size or running the pipeline. Result: all 12 wells'
  pairwise distances range from 0 m (co-located sidetracks sharing one
  physical slot, e.g. ``15/9-F-9``/``15/9-F-9 A``) up to **9.48 m**
  (``15/9-F-5`` to ``15/9-F-11 B``) -- i.e. every real Production/
  Injection well in this dataset is drilled from ONE physical platform
  template, all slots within 9.48 m of each other (unlike the 15/9-19
  exploration wells 1.5-2.5 km away, which are excluded from the candidate
  set above anyway). ``CLUSTER_BIN_M = 50`` m was chosen specifically
  because it is comfortably larger than this real 9.48 m maximum spread
  (so it does not arbitrarily split a genuinely single physical location)
  while remaining two orders of magnitude smaller than any realistic
  inter-field distance -- picked from the real distance table above,
  before running `hyperactive-plan` or looking at any NPV outcome.
  Applying it correctly produces **1 cluster of all 12 wells** -- an
  honest, real finding (Volve's producing wells are genuinely
  platform-based, unlike Egg Model/SPE9/SPE10's scattered layouts), not a
  repeat of the earlier US-F2 one-cluster-per-well mistake (which grouped
  wells that were NOT actually this close together).
* ``field`` = ``"volve"``, ``layer`` = ``"volve"`` (free text).
* ``well_type`` = ``"ГС"`` for every well (same closest-honest-fit
  reasoning as the other adapters -- no fracturing/stimulation keyword is
  available in this quick-acquisition pass's source data, so the drilling-
  only code is used; note Volve's real MD >> TVD for several wells, e.g.
  ``15/9-F-15 D`` TVD 3212 m / MD 4684 m, indicates real, significant
  directional deviation, which ``ННС`` would describe more precisely, but
  ``"ГС"`` is kept for consistency with every other adapter in this repo).
* ``x``/``y``: real wellhead UTM31N easting/northing (metres) from the
  Sodir FactPages lookup, used directly (already in metres, no unit
  conversion needed, unlike the Eclipse-deck adapters).
* ``length``: the REAL measured depth (``MD``, metres) from the
  ``bysarmad`` mirror CSV -- unlike the Egg Model/SPE9/SPE10 adapters
  (which only had a completion-interval proxy), Volve's source table gives
  each well's real total wellbore length directly.
* ``init_entry_date``: the REAL ``Completed date`` from the mirror CSV
  (when each well was finished and could start producing/injecting) --
  unlike the other three adapters (which all used one assumed simultaneous
  readiness date), Volve's real per-well completion dates are used,
  parsed from ``DD.MM.YYYY`` format.
* ``oil_rate`` / ``liq_rate`` (t/day): **NO measured production/injection
  RATE data was available within this pass's time-boxed acquisition** (the
  accessible sources above give identity, position, depth and dates, not
  volumes -- Equinor's own production-volume tables live inside the
  ~40,000-file archive gated behind the Databricks Marketplace login, not
  reached in this pass). Unlike the Egg Model/SPE9/SPE10 adapters (whose
  proxy rates are at least anchored to a real, deck-sourced static
  quantity -- local permeability or a real target-rate keyword), there is
  no such anchor available here. A uniform PLACEHOLDER rate is used
  instead, clearly weaker than the other adapters' proxies and disclosed
  as such: ``PRODUCER_OIL_RATE_TPD = 40.0`` t/day for all 9 producers
  (``liq_rate = oil_rate * 1.05``, the same 5% water-cut assumption used
  elsewhere), ``INJECTOR_LIQ_RATE_TPD = 150.0`` t/day for all 3 injectors
  (``oil_rate = 0``). These two numbers are NOT derived from any measured
  or static Volve quantity; they exist only to exercise the pipeline
  mechanics honestly-labeled as a placeholder, not a measurement.

PROXY OBJECTIVE / SCHEMA-FIT DISCLOSURE: position, depth (length) and
readiness dates are all real, well-specific, sourced data for the 12
included wells. Rates are an undifferentiated placeholder (see above) --
this means every producer in this benchmark has IDENTICAL modelled
economics, so any greedy/RL choice among producers is driven entirely by
readiness-date/crew-scheduling structure, not by any real well-to-well
value difference; this is a materially weaker benchmark than Egg Model/
SPE9/SPE10 on that specific axis, disclosed rather than dressed up. The
injectors' liquid volume is, as with every other adapter here, a pure lift
cost with no oil revenue in HyperActive's producer-only economics.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW_ROOT = Path(__file__).resolve().parent / "raw"
OUT_DIR = Path(__file__).resolve().parent / "hyperactive_input"

WELLS_CSV = RAW_ROOT / "volve_wells_bysarmad_mirror.csv"
COORDS_CSV = RAW_ROOT / "volve_wells_coords_sodir.csv"

PRODUCER_OIL_RATE_TPD = 40.0  # PLACEHOLDER, not measured -- see module docstring
PRODUCER_WATER_CUT_ASSUMPTION = 0.05  # assumed, not measured (same as other adapters)
INJECTOR_LIQ_RATE_TPD = 150.0  # PLACEHOLDER, not measured -- see module docstring

CLUSTER_BIN_M = 50.0  # spatial-binning threshold, see module docstring (US-F7)

INCLUDED_PURPOSES = {"Production", "Injection"}


def _slug(name: str) -> str:
    return name.strip()


def _parse_date(value: str) -> str:
    return datetime.strptime(value.strip(), "%d.%m.%Y").strftime("%Y-%m-%d")


def _bin_cluster(positions: dict[str, tuple[float, float]], bin_m: float) -> dict[str, str]:
    """Group wells into clusters by fixed-size spatial binning on real x/y (m).

    Same method as the Egg Model/SPE9/SPE10 adapters' ``_bin_cluster``, but
    operating directly on real metre coordinates (Volve's source already
    gives UTM metres, no grid-cell conversion needed). Unlike those
    adapters' (I, J) grid indices (which start near 1 and so have a
    natural bin origin), raw UTM easting/northing have no such anchor, so
    the bin grid is anchored at this well pool's own minimum x/y first --
    otherwise an arbitrary absolute-coordinate bin boundary could split
    two genuinely nearby wells purely because they straddle a multiple of
    ``bin_m`` in UTM space, which is not a real spatial-proximity signal.
    """
    min_x = min(x for x, _y in positions.values())
    min_y = min(y for _x, y in positions.values())
    bin_of: dict[tuple[int, int], list[str]] = {}
    for name, (x, y) in sorted(positions.items()):
        key = (int((x - min_x) // bin_m), int((y - min_y) // bin_m))
        bin_of.setdefault(key, []).append(name)
    cluster_of: dict[str, str] = {}
    for (bi, bj), names in sorted(bin_of.items()):
        cluster_id = f"VOLVE_C{bi:02d}_{bj:02d}"
        for name in names:
            cluster_of[name] = cluster_id
    return cluster_of


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    with WELLS_CSV.open(encoding="utf-8") as f:
        well_rows_raw = list(csv.DictReader(f, delimiter=";"))

    with COORDS_CSV.open(encoding="utf-8") as f:
        coord_rows_raw = list(csv.DictReader(f))

    coords_by_name: dict[str, tuple[str | None, str | None]] = {
        _slug(r["Wellbore"]): (r["ns_utm"], r["ew_utm"]) for r in coord_rows_raw
    }

    # Sidetracks not resolved by the FactPages name search reuse their
    # parent slot's real coordinates (a sidetrack re-enters the same
    # physical wellhead) -- see module docstring.
    SIDETRACK_PARENT = {
        "15/9-19 BT2": "15/9-19 B",
        "15/9-19 ST2": "15/9-19 S",
        "15/9-F-11 T2": "15/9-F-11",
    }
    for child, parent in SIDETRACK_PARENT.items():
        if child in coords_by_name and coords_by_name[child][0] is None and parent in coords_by_name:
            coords_by_name[child] = coords_by_name[parent]

    included = [r for r in well_rows_raw if r["Purpose"].strip() in INCLUDED_PURPOSES]

    positions: dict[str, tuple[float, float]] = {}
    for row in included:
        name = _slug(row["Wellbore"])
        ns, ew = coords_by_name.get(name, (None, None))
        if ns is None or ew is None:
            raise ValueError(f"no coordinates resolved for included well {name!r}")
        positions[name] = (float(ew), float(ns))  # (x=EW easting, y=NS northing)

    cluster_of = _bin_cluster(positions, CLUSTER_BIN_M)

    well_rows = []
    for row in included:
        name = _slug(row["Wellbore"])
        purpose = row["Purpose"].strip()
        is_injector = purpose == "Injection"
        md_m = float(row["MD (m)"])
        entry_date = _parse_date(row["Completed date"])
        if is_injector:
            oil_rate = 0.0
            liq_rate = INJECTOR_LIQ_RATE_TPD
        else:
            oil_rate = PRODUCER_OIL_RATE_TPD
            liq_rate = oil_rate * (1.0 + PRODUCER_WATER_CUT_ASSUMPTION)
        well_rows.append(
            {
                "name": name,
                "cluster": cluster_of[name],
                "field": "volve",
                "layer": "volve",
                "well_type": "ГС",
                "oil_rate": round(oil_rate, 4),
                "liq_rate": round(liq_rate, 4),
                "length": round(md_m, 4),
                "init_entry_date": entry_date,
            }
        )

    members_by_cluster: dict[str, list[str]] = {}
    for name, cid in cluster_of.items():
        members_by_cluster.setdefault(cid, []).append(name)
    coord_rows = []
    for cid, members in sorted(members_by_cluster.items()):
        cx = sum(positions[m][0] for m in members) / len(members)
        cy = sum(positions[m][1] for m in members) / len(members)
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
    print(f"wells (Production+Injection only, spatially binned clusters): {len(wells)} rows -> {wells_path}")
    print(f"coordinates: {len(coordinates)} rows ({coordinates['cluster'].nunique()} clusters) -> {coords_path}")
    print(wells.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
