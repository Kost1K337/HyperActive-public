"""Generic, standalone well-sequencing evaluator + baselines.

This reimplements (for the public repo, stdlib-only) the same *pattern*
used internally for the Oklahoma OCC drilling benchmark: ``resource_count``
parallel "bring-online" resource slots each process a sub-sequence of a
permutation ("order") of wells, using earliest-available-resource
assignment (the same rule used by the private repo's Oklahoma OCC
benchmark, ``scripts/research/oklahoma_benchmark.py``) plus an optional
grid-distance travel/setup term between consecutive wells assigned to the
same resource. Each well's value is realized (and time-discounted) at the
moment it comes online, and a "complete/feasible" plan is one that uses
every candidate well exactly once. Baselines (random-multistart,
greedy/value-density, local-swap search) and the AttentionDuelingQuantileQ
network (see ``benchmark/common/attention_dueling_quantile_q.py``) are all
evaluated under this same protocol so their objective values are
comparable.

NOTE ON WHY THIS MATTERS: with a single shared resource and uniform well
duration (an earlier version of this evaluator), the discounted objective
reduces to a rearrangement-inequality problem -- sorting by value density
descending is provably optimal, so no search or learned policy could ever
beat greedy on it. Using multiple resources (so *which* resource a well is
assigned to, not just its relative order, matters) plus variable per-well
duration (so the discount schedule itself depends on which wells land in
which position) removes that degeneracy -- see each benchmark's README.md
for confirmation that greedy is no longer uniformly optimal under this
objective.

IMPORTANT HONESTY NOTE: this is a *proxy* objective, not a reservoir
simulation. See each benchmark's README.md ("Proxy objective disclosure")
for exactly how ``value_index``, ``duration_days``, and ``location`` were
derived for that benchmark's wells. Nothing here calls Eclipse/OPM Flow or
any other simulator.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Well:
    well_id: str
    is_injector: bool
    value_index: float  # proxy economic value realized once the well is online
    duration_days: float  # proxy time this well occupies its assigned bring-online resource
    features: tuple[float, ...]  # fixed-length feature vector for the candidate-set scorer
    location: tuple[float, float] = (0.0, 0.0)  # grid (I, J) position, for the travel-time term


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    wells: list[Well]
    discount_rate: float = 0.10
    resource_count: int = 2  # parallel bring-online resource slots (K); K=1 recovers the old, provably-trivial objective
    travel_speed: float = 0.0  # grid units/day a resource can move between consecutive wells; 0 disables the travel term


@dataclass(frozen=True)
class EvaluationResult:
    method: str
    case_id: str
    seed: int
    complete_plan: bool
    feasible: bool
    objective_value: float
    runtime_seconds: float
    order: list[str]
    violations: list[str]
    metadata: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_value"] = (
            f"{self.objective_value:.6f}" if math.isfinite(self.objective_value) else "nan"
        )
        payload["runtime_seconds"] = f"{self.runtime_seconds:.6f}"
        payload["order"] = " ".join(self.order)
        payload["violations"] = json.dumps(self.violations, ensure_ascii=False)
        payload["metadata"] = json.dumps(self.metadata, ensure_ascii=False, sort_keys=True)
        return payload


def validate_order(case: BenchmarkCase, order: Sequence[str]) -> list[str]:
    required = {well.well_id for well in case.wells}
    order_set = set(order)
    violations: list[str] = []
    if len(order) != len(required):
        violations.append(f"wrong_length:{len(order)}!={len(required)}")
    missing = sorted(required - order_set)
    extra = sorted(order_set - required)
    if missing:
        violations.append("missing:" + ",".join(missing[:10]))
    if extra:
        violations.append("extra:" + ",".join(extra[:10]))
    if len(order_set) != len(order):
        violations.append("duplicates")
    return violations


def evaluate_order(
    case: BenchmarkCase,
    order: Sequence[str],
    *,
    method: str = "unknown",
    seed: int = 0,
    started_at: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """``case.resource_count`` parallel resource slots bring wells online;
    each well in ``order`` is assigned to whichever resource slot becomes
    available earliest (earliest-available-resource assignment, the same
    rule used by the private repo's Oklahoma OCC benchmark), optionally
    paying a grid-distance travel/setup delay if that resource's previous
    well is far away. Each well's proxy value is discounted by the day it
    actually comes online (its resource's start time), not by its position
    in ``order``."""
    started_at = time.perf_counter() if started_at is None else started_at
    violations = validate_order(case, order)
    if violations:
        return EvaluationResult(
            method=method,
            case_id=case.case_id,
            seed=seed,
            complete_plan=False,
            feasible=False,
            objective_value=float("nan"),
            runtime_seconds=time.perf_counter() - started_at,
            order=list(order),
            violations=violations,
            metadata=metadata or {},
        )

    wells = {well.well_id: well for well in case.wells}
    resource_count = max(1, int(case.resource_count))
    resource_free_day = [0.0] * resource_count
    resource_last_loc: list[tuple[float, float] | None] = [None] * resource_count
    total = 0.0
    for well_id in order:
        well = wells[well_id]
        best_idx = 0
        best_start_day: float | None = None
        for idx in range(resource_count):
            travel_days = 0.0
            last_loc = resource_last_loc[idx]
            if last_loc is not None and case.travel_speed > 0:
                distance = math.hypot(well.location[0] - last_loc[0], well.location[1] - last_loc[1])
                travel_days = distance / case.travel_speed
            start_day = resource_free_day[idx] + travel_days
            if best_start_day is None or start_day < best_start_day:
                best_start_day = start_day
                best_idx = idx
        start_day = best_start_day or 0.0
        resource_free_day[best_idx] = start_day + well.duration_days
        resource_last_loc[best_idx] = well.location

        discounted = well.value_index / ((1.0 + case.discount_rate) ** (start_day / 365.0))
        total += discounted

    return EvaluationResult(
        method=method,
        case_id=case.case_id,
        seed=seed,
        complete_plan=True,
        feasible=True,
        objective_value=total,
        runtime_seconds=time.perf_counter() - started_at,
        order=list(order),
        violations=[],
        metadata=metadata or {},
    )


def greedy_value_density_order(case: BenchmarkCase) -> list[str]:
    """Order wells by value / duration, descending (ties broken by id).

    NOTE: with ``resource_count`` > 1 and/or a nonzero ``travel_speed``,
    this heuristic is NOT provably optimal any more (see module docstring):
    it ignores which resource a well lands on and any travel-time coupling
    between consecutive wells on the same resource, so local-swap search
    and random multistart can and do beat it on some cases/seeds -- see
    each benchmark's README.md for the actual win/loss counts."""
    ranked = sorted(
        case.wells,
        key=lambda w: (-(w.value_index / max(1e-6, w.duration_days)), w.well_id),
    )
    return [w.well_id for w in ranked]


def random_best_order(case: BenchmarkCase, *, seed: int, samples: int) -> tuple[list[str], float]:
    rng = random.Random(seed)
    ids = [w.well_id for w in case.wells]
    best_order = ids[:]
    best_value = -float("inf")
    for _ in range(samples):
        order = ids[:]
        rng.shuffle(order)
        value = evaluate_order(case, order).objective_value
        if value > best_value:
            best_value = value
            best_order = order
    return best_order, best_value


def local_swap_search_order(
    case: BenchmarkCase,
    *,
    seed: int,
    max_evaluations: int,
    initial_order: Sequence[str] | None = None,
) -> tuple[list[str], float]:
    """Simulated-annealing-flavored local-swap search (VNS-style neighborhood
    of single pairwise swaps), reimplemented standalone from the same
    algorithmic pattern used for the Oklahoma OCC benchmark's local search."""
    rng = random.Random(seed)
    current = list(initial_order) if initial_order is not None else greedy_value_density_order(case)
    current_value = evaluate_order(case, current).objective_value
    best_order = list(current)
    best_value = current_value
    if len(current) < 2:
        return best_order, best_value
    temperature = max(1.0, abs(current_value) * 0.001)
    for _ in range(max_evaluations):
        candidate = list(current)
        i, j = rng.sample(range(len(candidate)), 2)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        candidate_value = evaluate_order(case, candidate).objective_value
        delta = candidate_value - current_value
        accept = delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-9))
        if accept:
            current = candidate
            current_value = candidate_value
        if candidate_value > best_value:
            best_order = candidate
            best_value = candidate_value
        temperature *= 0.995
    return best_order, best_value


def write_results_csv(results: Sequence[EvaluationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "benchmark_case", "metric_value", "valid", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            valid = result.complete_plan and result.feasible
            notes = {
                "seed": result.seed,
                "runtime_seconds": round(result.runtime_seconds, 6),
                "violations": result.violations,
                **result.metadata,
            }
            writer.writerow(
                {
                    "method": result.method,
                    "benchmark_case": result.case_id,
                    "metric_value": (
                        f"{result.objective_value:.6f}"
                        if math.isfinite(result.objective_value)
                        else "nan"
                    ),
                    "valid": "true" if valid else "false",
                    "notes": json.dumps(notes, ensure_ascii=False, sort_keys=True),
                }
            )
