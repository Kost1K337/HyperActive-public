"""Shared fixtures: a small, hand-built scenario used across the test suite.

Deliberately not the bundled synthetic pool: tests need wells whose NPV
ordering, cluster structure and readiness dates are known by construction, so
that "the greedy planner picks the best candidate" can be checked against a
value computed independently of the code under test.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from hyperactive.core import TeamPool, Well
from hyperactive.planning import (
    NPV,
    ArpsDeclineProductionProfile,
    BaseCapex,
    BaseOpex,
    DistanceTeamMovement,
    LinearProductionProfile,
    SimpleTeamMovement,
)

START = datetime(2025, 1, 1)

BUILD_COST_PER_METRE = {"DRILLING+GTM": 20_000.0, "DRILLING": 15_000.0}
EQUIPMENT_COST = 1_000_000.0
OIL_PRICE_PER_TONNE = 10_000.0
OIL_COST_PER_TONNE = 100.0
WATER_COST_PER_TONNE = 50.0
REPAIR_PER_YEAR = 100_000.0
MAINTAIN_PER_YEAR = 100_000.0
DISCOUNT_RATE = 0.1


@pytest.fixture
def project_start() -> datetime:
    return START


@pytest.fixture
def npv() -> NPV:
    """A fixed, simple NPV function; every test scenario shares one economy."""
    return NPV(
        oil_price_per_tone=OIL_PRICE_PER_TONNE,
        project_start_date=START,
        capex_cost=BaseCapex(build_cost_per_metr=dict(BUILD_COST_PER_METRE),
                             equipment_cost=EQUIPMENT_COST),
        opex_cost=BaseOpex(oil_cost_per_tone=OIL_COST_PER_TONNE,
                           water_cost_per_tone=WATER_COST_PER_TONNE,
                           repair_per_year=REPAIR_PER_YEAR,
                           maintain_per_year=MAINTAIN_PER_YEAR),
        discount_rate=DISCOUNT_RATE,
    )


@pytest.fixture
def linear_profile() -> LinearProductionProfile:
    return LinearProductionProfile()


@pytest.fixture
def arps_profile() -> ArpsDeclineProductionProfile:
    return ArpsDeclineProductionProfile()


@pytest.fixture
def movement() -> SimpleTeamMovement:
    """One day within a cluster, fourteen between clusters - no coordinates needed."""
    return SimpleTeamMovement()


@pytest.fixture
def distance_movement() -> DistanceTeamMovement:
    """Three clusters on a line, 1 km apart, for tests that need real geometry."""
    return DistanceTeamMovement.from_dicts([
        {"cluster": "C1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"cluster": "C2", "x": 1000.0, "y": 0.0, "z": 0.0},
        {"cluster": "C3", "x": 2000.0, "y": 0.0, "z": 0.0},
    ])


def _build_well(
    name: str,
    cluster: str = "C1",
    oil_rate: float = 30.0,
    liq_rate: float = 100.0,
    length: float = 1000.0,
    well_type: str = "DRILLING+GTM",
    ready_after_days: int | None = None,
) -> Well:
    """A well ready ``ready_after_days`` after :data:`START`, or immediately if None."""
    readiness = START + timedelta(days=ready_after_days) if ready_after_days is not None else None
    return Well(
        name=name, cluster=cluster, field="F1", layer="L1", well_type=well_type,
        oil_rate=oil_rate, liq_rate=liq_rate, length=length,
        readiness_date=readiness,
    )


@pytest.fixture
def make_well():
    """A factory: ``make_well("w1", cluster="C1", oil_rate=30.0, ready_after_days=5)``."""
    return _build_well


@pytest.fixture
def make_team_pool():
    """A factory: ``make_team_pool(drilling=2, gtm=1)`` -> a ready TeamPool.

    ``"DRILLING"`` / ``"GTM"`` are the task codes' English spellings
    (``Task.DRILLING.name`` / ``Task.GTM.name`` - see
    ``hyperactive.core.Task``), accepted by ``Task.from_code`` exactly like the
    project's native Cyrillic aliases (``ГС``, ``ГРП``, ...).
    """
    def factory(drilling: int = 1, gtm: int = 1) -> TeamPool:
        pool = TeamPool()
        pool.add_teams(["DRILLING"], num_teams=drilling)
        pool.add_teams(["GTM"], num_teams=gtm)
        return pool
    return factory


@pytest.fixture
def small_pool() -> list[Well]:
    """Four wells on two clusters with a clear, hand-computable NPV ordering.

    Oil rates increase from w1 to w4, everything else equal, so the greedy
    NPV ranking is exactly w4 > w3 > w2 > w1 - useful whenever a test needs a
    known-in-advance "best candidate".
    """
    return [
        _build_well("w1", cluster="C1", oil_rate=20.0),
        _build_well("w2", cluster="C1", oil_rate=30.0),
        _build_well("w3", cluster="C2", oil_rate=40.0),
        _build_well("w4", cluster="C2", oil_rate=50.0),
    ]
