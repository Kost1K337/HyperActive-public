"""Domain objects: task codes and plan aggregation."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from hyperactive.core import Task, TeamPool


class TestTaskFromCode:
    @pytest.mark.parametrize("code, expected", [
        ("DRILLING", Task.DRILLING),
        ("drilling", Task.DRILLING),  # case-insensitive
        ("  GTM  ", Task.GTM),        # surrounding whitespace stripped
        ("ГС", Task.DRILLING),        # native drilling aliases
        ("ННС", Task.DRILLING),
        ("МЗС", Task.DRILLING),
        ("БУРЕНИЕ", Task.DRILLING),
        ("ГРП", Task.GTM),            # native GTM alias
    ])
    def test_recognised_codes(self, code, expected):
        assert Task.from_code(code) is expected

    def test_unknown_code_raises(self):
        with pytest.raises(ValueError):
            Task.from_code("NOT_A_CODE")


class TestWellTasks:
    def test_single_task(self, make_well):
        well = make_well("w1", well_type="DRILLING")
        assert well.tasks == (Task.DRILLING,)

    def test_task_chain(self, make_well):
        well = make_well("w1", well_type="DRILLING+GTM")
        assert well.tasks == (Task.DRILLING, Task.GTM)

    def test_native_codes_equivalent_to_english(self, make_well):
        assert make_well("w1", well_type="ГС+ГРП").tasks == (Task.DRILLING, Task.GTM)


class TestTeamPool:
    def test_add_teams_registers_under_every_task_in_the_chain(self):
        pool = TeamPool()
        pool.add_teams(["DRILLING", "GTM"], num_teams=1)
        team = pool.get_teams_for_task(Task.DRILLING)[0]
        assert team in pool.get_teams_for_task(Task.GTM)
        assert pool.supported_tasks == {Task.DRILLING, Task.GTM}

    def test_add_teams_creates_independent_crews(self):
        pool = TeamPool()
        pool.add_teams(["DRILLING"], num_teams=3)
        assert len(pool.get_teams_for_task(Task.DRILLING)) == 3
        assert len(pool.teams) == 3


class TestPlanAggregation:
    def test_total_profit_sums_well_costs(self, small_pool, npv, movement, make_team_pool):
        from hyperactive.greedy import PlanBuilder
        from hyperactive.planning import ClusterRandomRiskStrategy, TeamManager

        start = datetime(2025, 1, 1)
        builder = PlanBuilder(start=start, end=start + timedelta(days=3650), cost_function=npv)
        plan = builder.compile(
            wells=list(small_pool),
            manager=TeamManager(team_pool=make_team_pool(2, 2), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        assert plan.total_profit() == pytest.approx(
            sum(c.cost for c in plan.well_plans if c.cost is not None)
        )

    def test_oil_production_per_year_matches_manual_sum(self, small_pool, npv, movement,
                                                         make_team_pool):
        from hyperactive.greedy import PlanBuilder
        from hyperactive.planning import ArpsDeclineProductionProfile, ClusterRandomRiskStrategy, TeamManager

        start = datetime(2025, 1, 1)
        builder = PlanBuilder(start=start, end=start + timedelta(days=3650), cost_function=npv,
                              production_profile=ArpsDeclineProductionProfile())
        plan = builder.compile(
            wells=list(small_pool),
            manager=TeamManager(team_pool=make_team_pool(2, 2), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        by_year = plan.get_oil_production_per_year()
        manual_total = sum(sum(c.oil_prod_profile) for c in plan.well_plans)
        assert sum(by_year.values()) == pytest.approx(manual_total)
