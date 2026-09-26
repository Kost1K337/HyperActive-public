#!/usr/bin/env python3
"""SPE9 -> candidate-set well-sequencing adapter.

Parses the vendored ``raw/SPE9.DATA`` deck (plus its ``PERMVALUES.DATA``
include) to build a single ``BenchmarkCase`` with all 26 wells (1 injector +
25 producers).

Unlike the Egg Model, SPE9's schedule genuinely changes well controls over
time: WCONPROD is re-issued three times (a real, dated schedule, not a
placement-only deck):

    day    0: all producers target  1500 STB/d oil
    day  300: all producers target   100 STB/d oil
    day  360: all producers target  1500 STB/d oil  (900-day total run)

This adapter keeps that real per-phase rate structure in each producer's
feature vector (three separate rate features, one per schedule phase)
instead of collapsing it to one number, since it is a genuine part of the
source deck (see README.md "Proxy objective disclosure" for the caveat
about what is NOT simulated).

PROXY OBJECTIVE DISCLOSURE: as with the Egg Model, no flow-simulator run is
available offline, so ``value_index`` is an ANALYTIC PROXY, not a
reservoir-simulated NPV:

    injector value_index  = 0.3 * injection_rate_target * local_permeability
    producer value_index  = local_permeability * mean(phase_rate_targets) / 100

where ``local_permeability`` is the mean PERMX (mDarcy) over the well's
completed layers (from COMPDAT's K1:K2 range) at its (I, J) location, and
the rate targets come directly from WCONINJE / the three WCONPROD records.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eclipse_parse import parse_numeric_array, parse_well_records, read_deck
from common.sequencing_env import BenchmarkCase, Well

RAW_ROOT = Path(__file__).resolve().parent / "raw"
GRID_NX, GRID_NY, GRID_NZ = 24, 25, 15
PHASE_RATES = (1500.0, 100.0, 1500.0)  # STB/d oil target, per WCONPROD reissue
INJECTOR_RATE = 5000.0  # STBW/d, from WCONINJE (constant across the schedule)


def _perm_array() -> list[float]:
    text = read_deck(RAW_ROOT / "PERMVALUES.DATA")
    values = parse_numeric_array(text, "PERMX")
    expected = GRID_NX * GRID_NY * GRID_NZ
    if len(values) != expected:
        raise ValueError(f"expected {expected} PERMX values, got {len(values)}")
    return values


def _local_permeability(perm: list[float], i: int, j: int, k1: int, k2: int) -> float:
    total = 0.0
    count = 0
    for k in range(k1, k2 + 1):
        idx = (k - 1) * GRID_NX * GRID_NY + (j - 1) * GRID_NX + (i - 1)
        total += perm[idx]
        count += 1
    return total / max(1, count)


def load_case(case_id: str = "spe9_base") -> BenchmarkCase:
    main_text = read_deck(RAW_ROOT / "SPE9.DATA")
    perm = _perm_array()

    positions: dict[str, tuple[int, int]] = {}
    for record in parse_well_records(main_text, "WELSPECS"):
        name, _group, i_str, j_str = record[0], record[1], record[2], record[3]
        positions[name] = (int(i_str), int(j_str))

    completions: dict[str, tuple[int, int]] = {}
    for record in parse_well_records(main_text, "COMPDAT"):
        # 'INJE1' 24 25 11 15 'OPEN' 1* 1* 1
        name, k1_str, k2_str = record[0], record[3], record[4]
        completions[name] = (int(k1_str), int(k2_str))

    wells: list[Well] = []
    for name, (i, j) in positions.items():
        is_injector = name.startswith("INJE")
        k1, k2 = completions.get(name, (1, GRID_NZ))
        local_perm = _local_permeability(perm, i, j, k1, k2)
        if is_injector:
            value_index = 0.3 * INJECTOR_RATE * local_perm / 1000.0
            phase_rates = (INJECTOR_RATE, INJECTOR_RATE, INJECTOR_RATE)
        else:
            mean_rate = sum(PHASE_RATES) / len(PHASE_RATES)
            value_index = local_perm * mean_rate / 100.0 / 1000.0
            phase_rates = PHASE_RATES
        features = (
            i / GRID_NX,
            j / GRID_NY,
            1.0 if is_injector else 0.0,
            0.0 if is_injector else 1.0,
            local_perm / 500.0,
            60.0 / 100.0,  # drainage radius (ft), constant across all wells per WELSPECS
            phase_rates[0] / 5000.0,
            phase_rates[1] / 5000.0,
            phase_rates[2] / 5000.0,
            1.0,  # bias
        )
        wells.append(
            Well(
                well_id=name,
                is_injector=is_injector,
                value_index=value_index,
                duration_days=10.0,  # matches the deck's initial TSTEP granularity
                features=features,
            )
        )
    wells.sort(key=lambda w: w.well_id)
    return BenchmarkCase(case_id=case_id, wells=wells, discount_rate=0.10)


if __name__ == "__main__":
    import json

    case = load_case()
    print(json.dumps({"well_count": len(case.wells)}, indent=2))
    for well in case.wells:
        print(well.well_id, well.is_injector, round(well.value_index, 3), well.features)
