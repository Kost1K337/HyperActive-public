"""Default economics and helpers shared by training, evaluation and inference.

Monetary values are in arbitrary currency units; volumes in tonnes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from hyperactive.core import TeamPool
from hyperactive.planning import (
    NPV,
    ArpsDeclineProductionProfile,
    BaseCapex,
    BaseOpex,
    OilConstraint,
)

PROJECT_START = datetime(2025, 1, 1)

# Drilling cost per metre by well type (see hyperactive.core.Task for the codes).
BUILD_COST_PER_METRE = {
    "ГС+ГРП": 25300,
    "ННС+ГРП": 12900,
    "МЗС": 27300,
    "МЗС+ГРП": 28300,
    "ГС": 23300,
}
EQUIPMENT_COST = 2_500_000
OIL_COST_PER_TONNE = 109.9
WATER_COST_PER_TONNE = 48.6
REPAIR_PER_YEAR = 3_093_900
MAINTAIN_PER_YEAR = 2_336_200
OIL_PRICE_PER_TONNE = 13_896
DISCOUNT_RATE = 0.125

# Average month length used to convert a drilling horizon in months to days.
DAYS_PER_MONTH = 30.4

DRILLING_CODES = ("ГС", "ННС", "МЗС")
GTM_CODES = ("ГРП",)


def default_npv(project_start: datetime = PROJECT_START) -> NPV:
    capex = BaseCapex(
        build_cost_per_metr=dict(BUILD_COST_PER_METRE),
        equipment_cost=EQUIPMENT_COST,
    )
    opex = BaseOpex(
        oil_cost_per_tone=OIL_COST_PER_TONNE,
        water_cost_per_tone=WATER_COST_PER_TONNE,
        repair_per_year=REPAIR_PER_YEAR,
        maintain_per_year=MAINTAIN_PER_YEAR,
    )
    return NPV(
        oil_price_per_tone=OIL_PRICE_PER_TONNE,
        project_start_date=project_start,
        capex_cost=capex,
        opex_cost=opex,
        discount_rate=DISCOUNT_RATE,
    )


def default_profile() -> ArpsDeclineProductionProfile:
    return ArpsDeclineProductionProfile()


def make_team_pool(drilling_crews: int, gtm_crews: int) -> TeamPool:
    pool = TeamPool()
    pool.add_teams(DRILLING_CODES, num_teams=drilling_crews)
    pool.add_teams(GTM_CODES, num_teams=gtm_crews)
    return pool


def horizon_end(start: datetime, horizon_years: int) -> datetime:
    """End of the economic horizon (NPV is accumulated up to this date)."""
    return start + timedelta(days=365 * horizon_years)


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
