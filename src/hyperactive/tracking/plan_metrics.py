"""Metrics of a built plan, for tracking.

Why. A plan summary carries three numbers - NPV, well count, production.
Everything else one needs to explain why RL beat (or lost to) the greedy
planner has to be recomputed by hand: unfold the schedule, add up crew
occupancy, count the clusters touched. That analysis was redone by hand many
times in this project.

This module computes those quantities once, at the moment the plan exists, and
they travel into the MLflow run as metrics - available afterwards by query,
without the plan at hand.

Nothing here may break an experiment: every quantity has a case where there is
nothing to compute it from (an empty plan, a missing schedule), and then it is
simply absent from the result.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable, Optional

from hyperactive.core import Plan, Task

logger = logging.getLogger(__name__)

DAY_SECONDS = 86400.0


def _overlap_days(start: datetime, end: datetime,
                  window: Optional[tuple[datetime, datetime]]) -> float:
    """Length of a segment inside the window, in days."""
    if window is not None:
        start = max(start, window[0])
        end = min(end, window[1])
    if end <= start:
        return 0.0
    return (end - start).total_seconds() / DAY_SECONDS


def _entries(plan: Plan) -> Iterable[Any]:
    for context in plan.well_plans:
        for entry in context.entries:
            yield entry


def _team_metrics(plan: Plan, task: Task, declared_teams: Any,
                  window: Optional[tuple[datetime, datetime]]) -> dict[str, float]:
    """Occupancy and utilisation of one crew type."""
    busy = 0.0
    travel = 0.0
    teams: set[str] = set()
    for entry in _entries(plan):
        if entry.task is not task:
            continue
        busy += _overlap_days(entry.start, entry.end, window)
        travel += entry.travel_time.total_seconds() / DAY_SECONDS
        teams.add(str(entry.team.id))
    if not teams and busy <= 0.0:
        return {}

    try:
        count = int(declared_teams)
    except (TypeError, ValueError):
        count = 0
    # The crew count from the scenario is the right denominator: a crew the
    # planner found no work for never appears in the schedule, yet its idle
    # time is exactly what has to be counted.
    count = count or len(teams)
    prefix = "drill" if task is Task.DRILLING else "gtm"
    out = {f"{prefix}_busy_days": round(busy, 2),
           f"{prefix}_teams_used": float(len(teams)),
           f"{prefix}_teams_declared": float(count),
           f"{prefix}_travel_days": round(travel, 2)}
    if window is not None and count > 0:
        capacity = count * _overlap_days(window[0], window[1], None)
        if capacity > 0:
            out[f"{prefix}_utilization"] = round(busy / capacity, 4)
    return out


def plan_metrics(plan: Optional[Plan],
                 window: Optional[tuple[datetime, datetime]] = None,
                 drilling_teams: Any = None,
                 gtm_teams: Any = None) -> dict[str, float]:
    """Composition, coverage, crew utilisation and span of a plan.

    ``window`` is the work window the utilisation is measured against
    (normally ``start`` .. ``end_jobs`` of the scenario). Utilisation without an
    explicit denominator is meaningless: the same occupancy over a 12-month and
    a 60-month window gives different shares.

    Returns only what could be computed; an empty dict is a valid answer for an
    empty plan.
    """
    if plan is None or not plan.well_plans:
        return {}
    try:
        contexts = plan.well_plans
        out: dict[str, float] = {
            "total_profit": float(plan.total_profit()),
            "total_wells": float(len(contexts)),
            "total_oil": float(plan.total_oil()),
        }
        pads = {str(c.well.cluster) for c in contexts}
        fields = {str(c.well.field) for c in contexts}
        out["pads_touched"] = float(len(pads - {"None", ""}))
        out["fields_touched"] = float(len(fields - {"None", ""}))

        starts = [entry.start for entry in _entries(plan)]
        ends = [entry.end for entry in _entries(plan)]
        if starts and ends:
            out["plan_span_days"] = round(_overlap_days(min(starts), max(ends), None), 2)

        out.update(_team_metrics(plan, Task.DRILLING, drilling_teams, window))
        out.update(_team_metrics(plan, Task.GTM, gtm_teams, window))

        # Per-well NPV: the sum matches the plan total, while the spread shows
        # whether the plan was filled with large wells or with small ones.
        values = [float(c.cost) for c in contexts if c.cost is not None]
        if values:
            out["npv_per_well_mean"] = round(sum(values) / len(values), 2)
            out["npv_per_well_min"] = round(min(values), 2)
            out["npv_per_well_max"] = round(max(values), 2)
        return out
    except Exception as exc:  # metrics have no right to break an experiment
        logger.warning("Plan metrics were not computed: %s", exc)
        return {}
