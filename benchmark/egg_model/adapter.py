#!/usr/bin/env python3
"""Egg Model -> candidate-set well-sequencing adapter.

Parses the vendored Eclipse deck (``raw/Eclipse/Egg_Model_ECL.DATA`` plus its
``COMPDAT.INC`` / ``SCHEDULE_NEW.INC`` / ``mDARCY.INC`` includes) and one of
the vendored permeability realizations to build a ``BenchmarkCase``: 12
candidate wells (8 injectors + 4 producers), each with a feature vector and
a proxy "bring this well online" value.

PROXY OBJECTIVE DISCLOSURE (see README.md for the full statement): the Egg
Model archive vendored here does not include a flow-simulator run, so there
is no ground-truth production/NPV signal available offline. The value used
here is an ANALYTIC PROXY, not a reservoir-simulated NPV:

    injector value_index  = 0.3 * injection_rate_target * local_permeability
    producer value_index  = local_permeability * (reference_bhp / 100)

where ``local_permeability`` is the mean PERMX (mDarcy) over the well's
completed layers (K=1..7, i.e. the full column) at its (I, J) location, and
``injection_rate_target`` / ``reference_bhp`` come directly from the deck's
WCONINJE / WCONPROD keywords. This rewards bringing high-permeability,
high-rate wells online earlier -- a reasonable, transparent stand-in for
value, but it is NOT a substitute for simulated cumulative oil/water
production. Do not read these numbers as physical NPV.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eclipse_parse import parse_numeric_array, parse_well_records, read_deck
from common.sequencing_env import BenchmarkCase, Well

RAW_ROOT = Path(__file__).resolve().parent / "raw"
ECLIPSE_ROOT = RAW_ROOT / "Eclipse"
GRID_NX, GRID_NY, GRID_NZ = 60, 60, 7


def _perm_array() -> list[float]:
    text = read_deck(ECLIPSE_ROOT / "mDARCY.INC")
    values = parse_numeric_array(text, "PERMX")
    expected = GRID_NX * GRID_NY * GRID_NZ
    if len(values) != expected:
        raise ValueError(f"expected {expected} PERMX values, got {len(values)}")
    return values


def _local_permeability(perm: list[float], i: int, j: int) -> float:
    """Mean PERMX over the full vertical column at 1-indexed (I, J)."""
    total = 0.0
    for k in range(1, GRID_NZ + 1):
        idx = (k - 1) * GRID_NX * GRID_NY + (j - 1) * GRID_NX + (i - 1)
        total += perm[idx]
    return total / GRID_NZ


def _well_controls() -> tuple[dict[str, tuple[int, int]], dict[str, float], dict[str, float]]:
    main_text = read_deck(ECLIPSE_ROOT / "Egg_Model_ECL.DATA")
    schedule_text = read_deck(ECLIPSE_ROOT / "SCHEDULE_NEW.INC")

    positions: dict[str, tuple[int, int]] = {}
    for record in parse_well_records(main_text, "WELSPECS"):
        name, _group, i_str, j_str = record[0], record[1], record[2], record[3]
        positions[name] = (int(i_str), int(j_str))

    injection_rate: dict[str, float] = {}
    for record in parse_well_records(schedule_text, "WCONINJE"):
        # 'INJECT1' 'WATER' 'OPEN' 'RATE' 79.5 1* 420
        try:
            injection_rate[record[0]] = float(record[4])
        except ValueError:
            injection_rate[record[0]] = 0.0

    producer_bhp: dict[str, float] = {}
    for record in parse_well_records(main_text, "WCONPROD"):
        # 'PROD1' 'OPEN' 'BHP' 1* 1* 1* 1* 1* 395
        numeric = [tok for tok in record[3:] if tok not in ("*",) and not tok.endswith("*")]
        if numeric:
            try:
                producer_bhp[record[0]] = float(numeric[-1])
            except ValueError:
                pass
    return positions, injection_rate, producer_bhp


def _build_case(case_id: str, perm: list[float]) -> BenchmarkCase:
    positions, injection_rate, producer_bhp = _well_controls()
    wells: list[Well] = []
    for name, (i, j) in positions.items():
        is_injector = name.startswith("INJECT")
        local_perm = _local_permeability(perm, i, j)
        if is_injector:
            rate = injection_rate.get(name, 0.0)
            value_index = 0.3 * rate * local_perm / 1000.0
            control_feature = rate / 100.0
        else:
            bhp = producer_bhp.get(name, 0.0)
            value_index = local_perm * (bhp / 100.0) / 1000.0
            control_feature = bhp / 1000.0
        features = (
            i / GRID_NX,
            j / GRID_NY,
            1.0 if is_injector else 0.0,
            0.0 if is_injector else 1.0,
            local_perm / 2000.0,
            0.2,  # wellbore radius (ft), constant across all wells per COMPDAT
            control_feature,
            1.0,  # bias
        )
        wells.append(
            Well(
                well_id=name,
                is_injector=is_injector,
                value_index=value_index,
                duration_days=30.0,  # one TSTEP block; the deck reports the schedule in 30-day steps
                features=features,
            )
        )
    wells.sort(key=lambda w: w.well_id)
    return BenchmarkCase(case_id=case_id, wells=wells, discount_rate=0.10)


def load_case(case_id: str = "egg_model_base") -> BenchmarkCase:
    return _build_case(case_id, _perm_array())


def load_realization_cases() -> list[BenchmarkCase]:
    """One case per vendored permeability realization (base + subset10)."""
    cases = [load_case("egg_model_base")]
    realization_dir = RAW_ROOT / "Permeability_Realizations_subset10"
    if not realization_dir.exists():
        return cases
    for perm_file in sorted(realization_dir.glob("PERM*_ECL.INC")):
        realization_id = perm_file.stem.replace("_ECL", "")
        text = read_deck(perm_file)
        values = parse_numeric_array(text, "PERMX")
        if len(values) != GRID_NX * GRID_NY * GRID_NZ:
            continue
        cases.append(_build_case(f"egg_model_{realization_id.lower()}", values))
    return cases


if __name__ == "__main__":
    import json

    cases = load_realization_cases()
    print(json.dumps({"case_count": len(cases), "wells_per_case": len(cases[0].wells)}, indent=2))
    for well in cases[0].wells:
        print(well.well_id, well.is_injector, round(well.value_index, 3), well.features)
