"""The greedy planner picks the candidate with the largest NPV, step by step.

Every test here re-derives the expected choice independently (by calling
``NPV.compute`` directly on candidates, exactly as the assignment code in
``hyperactive.planning.team_manager`` would build them) and checks the
planner's choice against that, rather than against a hard-coded well name -
so a test failure points at the selection rule, not at a stale fixture.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

from hyperactive.core import WellPlanContext
from hyperactive.greedy import PlanBuilder
from hyperactive.planning import ClusterRandomRiskStrategy, TeamManager

START = datetime(2025, 1, 1)
END = START + timedelta(days=3650)


def _candidate_cost(well, npv, team_pool, movement, profile):
    """NPV of scheduling ``well`` alone - the same computation the planner runs
    internally to rank candidates, replicated here to get an independent number."""
    manager = TeamManager(team_pool=team_pool, movement=movement)
    context = WellPlanContext(well=deepcopy(well), start=START, end=END)
    manager.get_assignments(context)
    profile.compute(context)
    npv.compute(context)
    return context.cost


class TestGreedySelection:
    def test_picks_the_highest_npv_candidate_first(self, small_pool, npv, movement,
                                                    make_team_pool, linear_profile):
        team_pool = make_team_pool(2, 2)
        expected_first = max(
            small_pool,
            key=lambda w: _candidate_cost(w, npv, team_pool, movement, linear_profile),
        )

        builder = PlanBuilder(start=START, end=END, cost_function=npv,
                              production_profile=linear_profile)
        plan = builder.compile(
            wells=list(small_pool),
            manager=TeamManager(team_pool=make_team_pool(2, 2), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        assert plan.well_plans[0].well.name == expected_first.name

    def test_full_order_is_npv_descending_when_wells_are_independent(
        self, small_pool, npv, movement, make_team_pool, linear_profile
    ):
        # With two drilling crews and four independent wells (two per cluster),
        # nothing forces an order other than NPV: every well is schedulable at
        # the same start time regardless of what was picked before it.
        team_pool = make_team_pool(2, 2)
        standalone_costs = {
            w.name: _candidate_cost(w, npv, team_pool, movement, linear_profile)
            for w in small_pool
        }

        builder = PlanBuilder(start=START, end=END, cost_function=npv,
                              production_profile=linear_profile)
        plan = builder.compile(
            wells=list(small_pool),
            manager=TeamManager(team_pool=make_team_pool(2, 2), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        order = [c.well.name for c in plan.well_plans]
        assert order == sorted(standalone_costs, key=standalone_costs.get, reverse=True)

    def test_not_simply_cheapest_capex(self, npv, movement, make_team_pool, linear_profile,
                                       make_well):
        """Regression guard for a real misconception about this planner.

        A short, cheap well and a long, expensive one that produces far more
        oil: the greedy rule must prefer the higher-NPV (expensive) well, not
        the lower-CAPEX one - it maximises value added, not minimises spend.
        """
        cheap_low_value = make_well("cheap", oil_rate=1.0, length=100.0)
        expensive_high_value = make_well("expensive", oil_rate=200.0, length=2000.0)

        team_pool = make_team_pool(1, 1)
        cheap_cost = _candidate_cost(cheap_low_value, npv, team_pool, movement, linear_profile)
        expensive_cost = _candidate_cost(expensive_high_value, npv, team_pool, movement,
                                         linear_profile)
        assert expensive_cost > cheap_cost, "fixture must actually be higher-NPV to test anything"

        builder = PlanBuilder(start=START, end=END, cost_function=npv,
                              production_profile=linear_profile)
        plan = builder.compile(
            wells=[cheap_low_value, expensive_high_value],
            manager=TeamManager(team_pool=make_team_pool(1, 1), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        assert plan.well_plans[0].well.name == "expensive"

    def test_keep_order_follows_init_entry_date_instead_of_npv(
        self, npv, movement, make_team_pool, linear_profile, make_well
    ):
        """``keep_order=True`` is a different selection rule entirely (used to
        evaluate a fixed well priority order), not a variant of the NPV rule."""
        low_value_first = make_well("low", oil_rate=1.0)
        low_value_first.init_entry_date = START
        high_value_second = make_well("high", oil_rate=100.0)
        high_value_second.init_entry_date = START + timedelta(days=1)

        builder = PlanBuilder(start=START, end=END, cost_function=npv,
                              production_profile=linear_profile)
        plan = builder.compile(
            wells=[high_value_second, low_value_first],
            manager=TeamManager(team_pool=make_team_pool(1, 1), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
            keep_order=True,
        )
        assert [c.well.name for c in plan.well_plans] == ["low", "high"]

    def test_infeasible_wells_are_dropped_not_erroring(self, npv, movement, make_team_pool,
                                                        linear_profile, make_well):
        """A well ready after the work window ends can never be scheduled and
        must simply be absent from the plan."""
        never_ready = make_well("never", ready_after_days=100_000)
        builder = PlanBuilder(start=START, end=START + timedelta(days=30),
                              end_jobs=START + timedelta(days=30),
                              cost_function=npv, production_profile=linear_profile)
        plan = builder.compile(
            wells=[never_ready],
            manager=TeamManager(team_pool=make_team_pool(1, 1), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        assert plan.well_plans == []

    def test_respects_cluster_dependency(self, npv, movement, make_team_pool, linear_profile,
                                         make_well):
        """A well that depends on another cluster cannot be scheduled until every
        remaining well of that cluster is placed."""
        blocker = make_well("blocker", cluster="C1", oil_rate=10.0)
        dependent = make_well("dependent", cluster="C2", oil_rate=1000.0)
        dependent.depend_from_cluster = "C1"

        builder = PlanBuilder(start=START, end=END, cost_function=npv,
                              production_profile=linear_profile)
        plan = builder.compile(
            wells=[dependent, blocker],
            manager=TeamManager(team_pool=make_team_pool(1, 1), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        # Despite dependent's far higher NPV, blocker (which unblocks it) must
        # be placed first.
        assert [c.well.name for c in plan.well_plans] == ["blocker", "dependent"]


class TestGreedySearchBudget:
    """``compile_best_of`` gives the greedy baseline the policy's search budget:
    best of N passes, each taking a random candidate with probability p."""

    @staticmethod
    def _run(wells, npv, movement, make_team_pool, linear_profile, **kwargs):
        builder = PlanBuilder(start=START, end=END, cost_function=npv,
                              production_profile=linear_profile)
        return builder.compile_best_of(
            wells=wells,
            make_manager=lambda: TeamManager(team_pool=make_team_pool(1, 1), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
            **kwargs,
        )

    @staticmethod
    def _wells(make_well):
        return [make_well(f"w{i}", cluster=f"C{i % 3}", oil_rate=10.0 + 7.0 * i)
                for i in range(6)]

    def test_zero_budget_is_the_deterministic_pass(self, npv, movement, make_team_pool,
                                                   linear_profile, make_well):
        wells = self._wells(make_well)
        plan, stats = self._run(wells, npv, movement, make_team_pool, linear_profile)
        reference = PlanBuilder(start=START, end=END, cost_function=npv,
                                production_profile=linear_profile).compile(
            wells=deepcopy(wells),
            manager=TeamManager(team_pool=make_team_pool(1, 1), movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        assert [c.well.name for c in plan.well_plans] == [c.well.name for c in reference.well_plans]
        assert stats["episodes"] == 1

    def test_full_exploration_with_window_one_is_still_greedy(self, npv, movement,
                                                              make_team_pool, linear_profile,
                                                              make_well):
        """A random pick among the single best-ranked candidate is the best one."""
        wells = self._wells(make_well)
        det, _ = self._run(wells, npv, movement, make_team_pool, linear_profile)
        rnd, _ = self._run(wells, npv, movement, make_team_pool, linear_profile,
                           episodes=3, exploration=1.0, window=1)
        assert [c.well.name for c in rnd.well_plans] == [c.well.name for c in det.well_plans]

    def test_exploration_is_seeded_and_best_npv_is_kept(self, npv, movement, make_team_pool,
                                                        linear_profile, make_well):
        wells = self._wells(make_well)
        kwargs = dict(episodes=5, exploration=0.5, window=4)
        plan_a, stats_a = self._run(wells, npv, movement, make_team_pool, linear_profile, **kwargs)
        plan_b, stats_b = self._run(wells, npv, movement, make_team_pool, linear_profile, **kwargs)
        assert stats_a["npvs"] == stats_b["npvs"]
        assert [c.well.name for c in plan_a.well_plans] == [c.well.name for c in plan_b.well_plans]
        assert stats_a["episodes"] == 5
        assert plan_a.total_profit() == max(stats_a["npvs"])
        # Exploration must actually change at least one pass's order.
        orders = set()
        for seed in range(5):
            builder = PlanBuilder(start=START, end=END, cost_function=npv,
                                  production_profile=linear_profile)
            p = builder.compile(wells=deepcopy(wells),
                                manager=TeamManager(team_pool=make_team_pool(1, 1),
                                                    movement=movement),
                                exploration=0.5, seed=seed, window=4)
            orders.add(tuple(c.well.name for c in p.well_plans))
        assert len(orders) > 1
