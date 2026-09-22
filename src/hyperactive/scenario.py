"""Project-wide scenario settings: economics, calendar conventions and crews.

These are the settings the benchmark is computed with, and every entry point -
``hyperactive-plan``, training, evaluation and the benchmark - takes them from
here, so a plan built anywhere in the project is priced and scheduled the same
way. A run that needs different economics passes a settings file
(:func:`load_settings`) rather than editing a copy.

Monetary values are in arbitrary currency units; volumes in tonnes.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from hyperactive.core import TeamPool
from hyperactive.data.loader import READINESS_HOUR
from hyperactive.planning import (
    NPV,
    ArpsDeclineProductionProfile,
    BaseCapex,
    BaseOpex,
    OilConstraint,
)

PROJECT_START = datetime(2025, 1, 1)

ECONOMICS: dict[str, Any] = {
    "equipment_cost": 2500000.0,
    "oil_cost_per_tone": 109.9,
    "water_cost_per_tone": 420.0,
    "repair_per_year": 3093900.0,
    "maintain_per_year": 2336200.0,
    "oil_price_per_tone": 13896.0,
    "discount_rate": 0.125,
    # Drilling cost per metre by well type (see hyperactive.core.Task for the codes).
    "build_cost_per_meter": {
        "ГС+ГРП": 25300.0, "ННС+ГРП": 12900.0, "МЗС": 27300.0,
        "МЗС+ГРП": 28300.0, "ГС": 23300.0, "ННС": 10900.0,
    },
}

# The economic horizon counts calendar years; the work window counts months of
# a fixed 30.4 days. The two differ on purpose: both are the benchmark's own
# conventions, and changing either moves its cell boundaries by a day.
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.4

DRILLING_CODES = ("ГС", "ННС", "МЗС")
GTM_CODES = ("ГРП",)


@dataclass
class Settings:
    """Everything that prices and schedules a plan, beyond the crews and the dates."""

    economics: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(ECONOMICS))
    readiness_hour: Optional[int] = READINESS_HOUR
    days_per_year: float = DAYS_PER_YEAR

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: Optional[str | Path] = None) -> Settings:
    """The project settings, with the overrides of a JSON file applied on top.

    The file may set ``economics`` (merged key by key into :data:`ECONOMICS`; a
    ``build_cost_per_meter`` given there replaces the whole table),
    ``readiness_hour`` and ``days_per_year``. Unknown keys are rejected rather
    than ignored: a typo would otherwise silently train or benchmark on the
    defaults.
    """
    settings = Settings()
    if path is None:
        return settings
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(overrides) - {"economics", "readiness_hour", "days_per_year"}
    if unknown:
        raise ValueError(f"{path}: unknown settings {sorted(unknown)}")
    settings.economics.update(overrides.get("economics", {}))
    if "readiness_hour" in overrides:
        settings.readiness_hour = overrides["readiness_hour"]
    if "days_per_year" in overrides:
        settings.days_per_year = float(overrides["days_per_year"])
    return settings


def make_npv(project_start: datetime = PROJECT_START,
             economics: Optional[dict[str, Any]] = None) -> NPV:
    economics = ECONOMICS if economics is None else economics
    return NPV(
        oil_price_per_tone=economics["oil_price_per_tone"],
        project_start_date=project_start,
        capex_cost=BaseCapex(
            build_cost_per_metr=dict(economics["build_cost_per_meter"]),
            equipment_cost=economics["equipment_cost"],
        ),
        opex_cost=BaseOpex(
            oil_cost_per_tone=economics["oil_cost_per_tone"],
            water_cost_per_tone=economics["water_cost_per_tone"],
            repair_per_year=economics["repair_per_year"],
            maintain_per_year=economics["maintain_per_year"],
        ),
        discount_rate=economics["discount_rate"],
    )


def default_profile() -> ArpsDeclineProductionProfile:
    return ArpsDeclineProductionProfile()


def planning_start(wells: list[Any], default: datetime = PROJECT_START) -> datetime:
    """Midnight of the earliest readiness date in the pool.

    The benchmark starts every fund here, so its work window opens with the
    first ready well instead of idle months; a fixed calendar date would make
    a short work window infeasible for any pool that becomes ready later.
    """
    dates = [well.readiness_date for well in wells if well.readiness_date is not None]
    if not dates:
        return default
    earliest = min(dates)
    return datetime(earliest.year, earliest.month, earliest.day)


def make_team_pool(drilling_crews: int, gtm_crews: int) -> TeamPool:
    pool = TeamPool()
    pool.add_teams(DRILLING_CODES, num_teams=drilling_crews)
    pool.add_teams(GTM_CODES, num_teams=gtm_crews)
    return pool


def horizon_end(start: datetime, horizon_years: float,
                days_per_year: float = DAYS_PER_YEAR) -> datetime:
    """End of the economic horizon (NPV is accumulated up to this date)."""
    return start + timedelta(days=int(round(days_per_year * horizon_years)))


def work_window_end(start: datetime, end: datetime, drilling_months: Optional[int]) -> datetime:
    """End of the work window: every task must finish before it. Never beyond ``end``."""
    if not drilling_months:
        return end
    return min(start + timedelta(days=int(round(DAYS_PER_MONTH * int(drilling_months)))), end)


def oil_constraints(bound: Optional[float]) -> list[Any]:
    """An annual oil production cap that applies to every year; ``None`` means no cap."""
    if bound is None:
        return []
    return [OilConstraint(bounds=[{"value": float(bound), "date": None}])]
