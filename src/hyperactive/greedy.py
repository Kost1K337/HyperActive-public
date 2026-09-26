"""Greedy baseline planner: repeatedly add the candidate with the highest NPV."""

from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

from hyperactive.core import Plan, Well, WellPlanContext
from hyperactive.planning.constraint import Constraint, ConstraintManager
from hyperactive.planning.cost import CostFunction
from hyperactive.planning.infrastructure import Infrastructure, SimpleInfrastructure
from hyperactive.planning.production import LinearProductionProfile, ProductionProfile
from hyperactive.planning.risk_strategy import RiskStrategy
from hyperactive.planning.team_manager import BaseTeamManager


class PlanBuilder:
    """Greedy plan construction.

    At each step every remaining well is scheduled tentatively on the crew
    calendars, its production profile and NPV are computed, candidates that
    violate constraints or do not finish before ``end_jobs`` are dropped, and
    the candidate with the largest ``NPV - drill_team_penalty`` is committed.
    The RL environment (:class:`hyperactive.env.PlanEnv`) builds candidates with
    exactly the same procedure; only the selection rule differs.

    ``end`` is the end of the economic horizon (how long NPV is accumulated);
    ``end_jobs`` is the end of the work window (all tasks must finish before it).
    """

    def __init__(
        self,
        start: datetime,
        end: datetime,
        cost_function: CostFunction,
        infrastructure: Infrastructure = SimpleInfrastructure(),
        production_profile: ProductionProfile = LinearProductionProfile(),
        constraints: Optional[list[Constraint]] = None,
        use_drill_team_penalty: bool = True,
        end_jobs: Optional[datetime] = None,
        max_commissioning_days: int = 0,
    ) -> None:
        self.start = start
        self.end = end
        self.end_jobs = end_jobs or end
        self.infra = infrastructure
        self.profiler = production_profile
        self.cost_function = cost_function
        self.remaining_wells: list[Well]
        self._constraints = ConstraintManager(constraints if constraints is not None else [])
        self.use_drill_team_penalty = use_drill_team_penalty
        # Upper bound on the total commissioning time of one well: drilling,
        # waiting for a GTM crew and the GTM itself. 0 disables the bound.
        self.max_commissioning_days = max_commissioning_days

    def compile(
        self,
        wells: list[Well],
        manager: BaseTeamManager,
        risk_strategy: Optional[RiskStrategy] = None,
        keep_order: bool = False,
        exploration: float = 0.0,
        seed: int = 0,
        window: Optional[int] = None,
    ) -> Plan:
        """Build a plan.

        With ``keep_order=True`` the candidate with the earliest
        ``init_entry_date`` is taken instead of the best one; this evaluates a
        fixed well order under the same scheduling rules.

        With ``exploration=p`` a uniformly random candidate among the ``window``
        best-ranked ones (all of them if ``window`` is None) is taken with
        probability ``p`` instead of the best one, seeded by ``seed``. This is
        the exploration rule of :func:`hyperactive.inference.run_episode`
        applied to the greedy ranking, so both sides can get the same search
        budget. ``exploration=0`` is the deterministic pass.
        """
        plan = Plan()
        self.remaining_wells = wells.copy()
        self._reset_profile_state()
        rng = np.random.default_rng(seed)

        current_start = self.start
        while self.remaining_wells and current_start < self.end_jobs:
            candidates = self._build_contexts(manager, current_start)
            if not candidates:
                break

            candidates = self._filter_candidates(candidates, plan, risk_strategy)

            if not candidates:
                current_start = self._constraints.get_period_end(current_start) or self.end
                continue

            best_candidate = None
            if exploration > 0.0 and not keep_order and float(rng.random()) < exploration:
                ranked = sorted((c for c in candidates if c.cost is not None),
                                key=self._ranking_key, reverse=True)[:window]
                if ranked:
                    best_candidate = ranked[int(rng.integers(len(ranked)))]
            if best_candidate is None:
                best_candidate = self._select_best_candidate(candidates, keep_order=keep_order)
            manager.assign(best_candidate)

            self.remaining_wells.remove(best_candidate.well)

            if risk_strategy:
                risk_strategy.define_risk(best_candidate)
                self.cost_function.compute(best_candidate)

            plan.add_context(best_candidate)
            self._commit_profile_state(best_candidate)

        return plan

    def _build_contexts(
        self,
        manager: BaseTeamManager,
        start: datetime,
    ) -> list[WellPlanContext]:
        return [
            context
            for well in self.remaining_wells
            if (context := self._build_context(well, manager, start)) is not None
        ]

    def _build_context(
        self,
        well: Well,
        manager: BaseTeamManager,
        start: datetime,
    ) -> Optional[WellPlanContext]:
        if not self._is_cluster_finished(well.depend_from_cluster):
            return None

        context = WellPlanContext(
            well,
            start=max(self.infra.get_ready_date(well=well), start),
            end=self.end,
        )
        manager.get_assignments(context)

        # Drop wells that do not fit into the work window.
        if context.get_next_available_date() > self.end_jobs or not context.entries:
            return None
        if self.max_commissioning_days and self._commissioning_days(context) > float(
            self.max_commissioning_days
        ):
            return None
        self.profiler.compute(context)

        return context

    def _commissioning_days(self, context: WellPlanContext) -> float:
        first = min(entry.start for entry in context.entries)
        last = max(entry.end for entry in context.entries)
        return (last - first).total_seconds() / 86400.0

    def _select_best_candidate(
        self, candidates: list[WellPlanContext], keep_order: bool = False
    ) -> WellPlanContext:
        if keep_order:
            return min(
                (c for c in candidates),
                key=lambda x: (
                    x.well.init_entry_date is None,
                    x.well.init_entry_date or datetime.max,
                ),
            )

        return max((c for c in candidates if c.cost is not None), key=self._ranking_key)

    def _ranking_key(self, context: WellPlanContext) -> float:
        return context.cost - (
            context.metadata.get("drill_team_penalty", 0) if self.use_drill_team_penalty else 0
        )

    def compile_best_of(
        self,
        wells: list[Well],
        make_manager: Callable[[], BaseTeamManager],
        risk_strategy: Optional[RiskStrategy] = None,
        episodes: int = 1,
        exploration: float = 0.0,
        window: Optional[int] = None,
    ) -> tuple[Plan, dict[str, Any]]:
        """Best of ``episodes`` plans by NPV; episode ``k`` is seeded with ``k``.

        The greedy counterpart of :func:`hyperactive.inference.plan_with_policy`
        (same seeds, same selection by total NPV, ties to the larger plan).
        Every episode gets fresh copies of the wells, a fresh crew manager and
        a fresh copy of ``risk_strategy``.
        """
        results = []
        for episode in range(max(1, int(episodes))):
            plan = self.compile(
                wells=deepcopy(wells),
                manager=make_manager(),
                risk_strategy=deepcopy(risk_strategy),
                exploration=exploration,
                seed=episode,
                window=window,
            )
            results.append((plan.total_profit(), len(plan.well_plans), episode, plan))
        npv, size, episode, plan = max(results, key=lambda item: (item[0], item[1]))
        stats = {
            "episodes": len(results),
            "exploration": exploration,
            "selected_episode": episode,
            "npvs": [item[0] for item in results],
        }
        return plan, stats

    def _is_cluster_finished(self, cluster: Optional[str]) -> bool:
        if cluster is None:
            return True
        return not any(well.cluster == cluster for well in self.remaining_wells)

    def _filter_candidates(
        self,
        candidates: list[WellPlanContext],
        plan: Plan,
        risk_strategy: Optional[RiskStrategy] = None,
    ) -> list[WellPlanContext]:
        if risk_strategy:
            candidates = [risk_strategy.apply_risk(c) for c in candidates]

        candidates = [self.cost_function.compute(c) for c in candidates]

        return [c for c in candidates if not self._constraints.is_violated(plan, c)]

    def _reset_profile_state(self) -> None:
        reset_fn = getattr(self.profiler, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def _commit_profile_state(self, context: WellPlanContext) -> None:
        commit_fn = getattr(self.profiler, "commit", None)
        if callable(commit_fn):
            commit_fn(context)
