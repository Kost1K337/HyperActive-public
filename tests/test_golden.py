"""Golden regression test: fixed inputs, hard-coded expected outputs.

Guards against silent numerical drift in the parts of the pipeline that unit
tests exercise individually but never end-to-end: NPV discounting, the Arps
decline profile, crew scheduling, greedy selection, and (for the RL half) the
released ``arrive39`` weights together with the feature encoding they were
trained against.

The scenario mirrors a real cell of `benchmark/run_benchmark.py`'s grid -
the full bundled pool (``data/synthetic/``, which is committed and does not
change; it is the 64-well fund also shipped at ``benchmark/case_10/64``),
its ``64/2x2/12 months`` crew and work-window combination, the benchmark's
economics and production profile, and readiness at 00:00 rather than 08:00 -
so the reference values below are also a cross-check against
``docs/reproducing.md``'s benchmark numbers, not just against this test's own
history. They were computed once, directly from this repository's code; a
failure here means *something* in that path changed the numbers, not
necessarily that it is wrong - if a deliberate change explains the new
values, recompute and update the constants below rather than loosen the
tolerances.

The greedy half involves no ``torch`` and reproduces exactly (``rel=1e-9``).
The RL half runs the trained network through PyTorch, whose matrix-multiply
backend can differ by platform/BLAS build enough to move the last few digits
of an NPV (observed empirically: ~3e-4 relative on one benchmark cell across
machines) without changing which wells get picked - hence the looser
``rel=1e-3`` there, while the *order* (a discrete, not continuous, quantity)
is still checked exactly.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from hyperactive.data import load_coordinates, load_wells
from hyperactive.env import PlanEnv
from hyperactive.greedy import PlanBuilder
from hyperactive.inference import load_policy, run_episode
from hyperactive.planning import (
    NPV,
    ArpsDeclineProductionProfile,
    BaseCapex,
    BaseOpex,
    ClusterRandomRiskStrategy,
    DistanceTeamMovement,
    TeamManager,
)
from hyperactive.scenario import make_team_pool, work_window_end

ROOT = Path(__file__).resolve().parents[1]
DRILLING_CREWS, GTM_CREWS = 2, 2
WORK_WINDOW_MONTHS = 12
PLANNING_MONTHS = 240  # economic horizon, matches run_benchmark.py's --planning-months default
DAYS_PER_MONTH = 365.25 / 12.0

# Matches BENCHMARK_ECONOMICS in benchmark/run_benchmark.py.
ECONOMICS = {
    "equipment_cost": 2500000.0,
    "oil_cost_per_tone": 109.9,
    "water_cost_per_tone": 420.0,
    "repair_per_year": 3093900.0,
    "maintain_per_year": 2336200.0,
    "oil_price_per_tone": 13896.0,
    "discount_rate": 0.125,
    "build_cost_per_meter": {
        "ГС+ГРП": 25300.0, "ННС+ГРП": 12900.0, "МЗС": 27300.0,
        "МЗС+ГРП": 28300.0, "ГС": 23300.0, "ННС": 10900.0,
    },
}

EXPECTED_GREEDY_NPV = 12517878488.87347
EXPECTED_GREEDY_ORDER = [
    "WELL_051", "WELL_064", "WELL_038", "WELL_025",
    "WELL_049", "WELL_062", "WELL_050", "WELL_024",
]

EXPECTED_RL_NPV = 12943443449.423958
EXPECTED_RL_ORDER = [
    "WELL_050", "WELL_064", "WELL_051", "WELL_038", "WELL_062",
    "WELL_025", "WELL_049", "WELL_024", "WELL_001",
]


def _npv(start) -> NPV:
    return NPV(
        oil_price_per_tone=ECONOMICS["oil_price_per_tone"],
        project_start_date=start,
        capex_cost=BaseCapex(
            build_cost_per_metr=dict(ECONOMICS["build_cost_per_meter"]),
            equipment_cost=ECONOMICS["equipment_cost"],
        ),
        opex_cost=BaseOpex(
            oil_cost_per_tone=ECONOMICS["oil_cost_per_tone"],
            water_cost_per_tone=ECONOMICS["water_cost_per_tone"],
            repair_per_year=ECONOMICS["repair_per_year"],
            maintain_per_year=ECONOMICS["maintain_per_year"],
        ),
        discount_rate=ECONOMICS["discount_rate"],
    )


@pytest.fixture(scope="module")
def fund():
    wells, rejected = load_wells(ROOT / "data" / "synthetic" / "wells.csv", readiness_hour=None)
    assert rejected == 0, "the bundled synthetic pool must not have changed"
    assert len(wells) == 64, "the bundled synthetic pool must not have changed"
    coordinates = load_coordinates(ROOT / "data" / "synthetic" / "clusters.csv", wells)
    movement = DistanceTeamMovement.from_dicts(coordinates)
    # Per-fund planning start = earliest well readiness, as in run_benchmark.py.
    start = min(well.readiness_date for well in wells)
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    plan_end = start + timedelta(days=int(round(PLANNING_MONTHS * DAYS_PER_MONTH)))
    jobs_end = work_window_end(start, plan_end, WORK_WINDOW_MONTHS)
    return wells, movement, start, plan_end, jobs_end


class TestGreedyGolden:
    def test_matches_the_recorded_reference(self, fund):
        wells, movement, start, plan_end, jobs_end = fund
        plan = PlanBuilder(start=start, end=plan_end, end_jobs=jobs_end, cost_function=_npv(start),
                           production_profile=ArpsDeclineProductionProfile()).compile(
            wells=deepcopy(wells),
            manager=TeamManager(team_pool=make_team_pool(DRILLING_CREWS, GTM_CREWS),
                                movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        assert [c.well.name for c in plan.well_plans] == EXPECTED_GREEDY_ORDER
        assert plan.total_profit() == pytest.approx(EXPECTED_GREEDY_NPV, rel=1e-9)


class TestReleasedModelGolden:
    def test_deterministic_rollout_matches_the_recorded_reference(self, fund):
        wells, movement, start, plan_end, jobs_end = fund
        bundle = load_policy(ROOT / "models" / "arrive39", device="cpu")
        env = PlanEnv(
            wells=deepcopy(wells), team_pool=make_team_pool(DRILLING_CREWS, GTM_CREWS),
            movement=movement, cost_function=_npv(start), n_actions=bundle.n_actions,
            start=start, end=plan_end, end_jobs=jobs_end,
            production_profile=ArpsDeclineProductionProfile(),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        plan, _ = run_episode(env, bundle, exploration=0.0, seed=0)
        assert [c.well.name for c in plan.well_plans] == EXPECTED_RL_ORDER
        assert plan.total_profit() == pytest.approx(EXPECTED_RL_NPV, rel=1e-3)

    def test_rl_beats_greedy_on_this_reference_scenario(self, fund):
        """Not a coincidence to preserve for its own sake: this is the paper's
        basic claim, on one fixed, reproducible scenario."""
        assert EXPECTED_RL_NPV > EXPECTED_GREEDY_NPV
