"""Gymnasium environment: sequential construction of a drilling plan."""

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hyperactive.core import Plan, TeamPool, Well, WellPlanContext
from hyperactive.env.features import EMBEDDING_WIDTH, CandidateFeatures
from hyperactive.planning.constraint import Constraint, ConstraintManager
from hyperactive.planning.cost import CostFunction
from hyperactive.planning.infrastructure import Infrastructure, SimpleInfrastructure
from hyperactive.planning.production import LinearProductionProfile, ProductionProfile
from hyperactive.planning.risk_strategy import RiskStrategy
from hyperactive.planning.team_manager import BaseMovement, BaseTeamManager, TeamManager

INVALID_ACTION_REWARD = -10000


class PlanEnv(CandidateFeatures, gym.Env):
    """Build a plan well by well; the action chooses which candidate goes next.

    **State.** At every step all remaining wells are scheduled tentatively on
    the crew calendars (the same procedure as in the greedy planner), their
    production profiles and NPVs are computed, and candidates that violate
    constraints or do not finish before ``end_jobs`` are dropped. The remaining
    candidates are sorted by NPV and truncated to the ``n_actions`` best.

    **Observation.** A ``float32`` matrix ``[n_actions, 39]``, one feature row
    per candidate (see :mod:`hyperactive.env.features`), zero-padded.

    **Action.** An index into the candidate list. The valid actions are the
    first ``len(candidates)`` indices; the mask is returned in
    ``info["action_mask"]`` and by :meth:`get_action_mask`, because after
    ``VecNormalize`` a padding row is no longer zero.

    **Reward.** The increase of plan NPV caused by the step, so the undiscounted
    episode return equals the NPV of the final plan. Choosing a padding row ends
    the episode with reward ``INVALID_ACTION_REWARD``.

    **Termination.** When no candidate is left (all wells are placed, the work
    window is exhausted or every remaining well violates a constraint).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        wells: list[Well],
        team_pool: TeamPool,
        movement: BaseMovement,
        cost_function: CostFunction,
        n_actions: int = 8,
        start: Optional[datetime] = None,
        infrastructure: Infrastructure = SimpleInfrastructure(),
        end: Optional[datetime] = None,
        end_jobs: Optional[datetime] = None,
        production_profile: ProductionProfile = LinearProductionProfile(),
        risk_strategy: Optional[RiskStrategy] = None,
        constraints: Optional[list[Constraint]] = None,
        max_commissioning_days: int = 0,
    ):
        super().__init__()
        start = start if start is not None else datetime.now()
        end = end if end is not None else start + timedelta(days=365 * 25)

        self.embedding_width = EMBEDDING_WIDTH
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_actions, self.embedding_width), dtype=np.float32
        )
        self.wells = wells
        self.start = start
        self.team_pool = team_pool
        self.remaining_wells = None
        self.infra = infrastructure
        self.end = end
        self.end_jobs = end_jobs or end
        self.movement = movement
        self.profiler = production_profile
        self.cost_function = cost_function
        self.risk_strategy = risk_strategy
        self._constraints = ConstraintManager(constraints if constraints is not None else [])
        # Upper bound on the total commissioning time of one well: drilling,
        # waiting for a GTM crew and the GTM itself. 0 disables the bound.
        self.max_commissioning_days = max_commissioning_days

    def step(self, action):
        action = int(action)
        info = {}

        if (self.observation[action] == np.array([0.0] * self.embedding_width)).all():
            self.reward = INVALID_ACTION_REWARD
            self.terminated = True
            info["final_plan"] = self.plan
            info["final_npv"] = self.plan.total_profit()
            info["action_mask"] = self.get_action_mask()
            return self.observation, self.reward, self.terminated, False, info
        elif len(self.remaining_wells) == 0:
            self.terminated = True
            info["final_plan"] = self.plan
            info["final_npv"] = self.plan.total_profit()
            info["action_mask"] = self.get_action_mask()
            return self.observation, self.reward, self.terminated, False, info

        best_cand = self.candidates[action]
        self.manager.assign(best_cand)
        self.remaining_wells.remove(best_cand.well)

        if self.risk_strategy:
            self.risk_strategy.define_risk(best_cand)
            self.cost_function.compute(best_cand)

        self.plan.add_context(best_cand)
        self._last_cluster = str(best_cand.well.cluster)
        self._commit_profile_state(best_cand)

        self.candidates = self._build_contexts(self.manager, self.current_start)
        self.candidates = self._filter_candidates(self.candidates, self.plan, self.risk_strategy)
        self.candidates = self._limit_candidates(self.candidates)

        self.curr_reward = self.plan.total_profit()
        self.reward = self.curr_reward - self.prev_reward
        self.prev_reward = self.curr_reward

        if not self.candidates:
            self.terminated = True
            info["final_plan"] = self.plan
            info["final_npv"] = self.plan.total_profit()
            info["action_mask"] = self.get_action_mask()
            return self.observation, self.reward, self.terminated, False, info

        self.observation = self._build_observation(self.candidates)
        info["action_mask"] = self.get_action_mask()

        return self.observation, self.reward, self.terminated, False, info

    def reset(self, seed=None, options=None):
        info = {}
        self.plan = Plan()
        self.prev_reward = 0
        self.curr_reward = 0
        self.terminated = False

        team_pool = deepcopy(self.team_pool)

        self.manager = TeamManager(
            team_pool=team_pool,
            movement=self.movement,
        )
        self.remaining_wells = self.wells.copy()
        self._reset_profile_state()
        # Cache key of the per-step feature tables: within an episode the
        # remainder shrinks by one well per step, but between episodes its
        # length returns to the initial value, so the length alone is not enough.
        self._episode_serial = getattr(self, "_episode_serial", 0) + 1
        self.current_start = self.start
        self._initial_well_count = max(1, len(self.wells))
        self._last_cluster = None
        self.candidates = self._build_contexts(self.manager, self.current_start)
        self.candidates = self._filter_candidates(self.candidates, self.plan, self.risk_strategy)
        self.candidates = self._limit_candidates(self.candidates)
        self.observation = self._build_observation(self.candidates)
        info["action_mask"] = self.get_action_mask()

        return self.observation, info

    def get_action_mask(self) -> np.ndarray:
        """Boolean mask of valid actions, of length ``action_space.n``.

        The observation is padded with zero rows to a fixed size, but after
        VecNormalize a zero row is no longer zero, so the mask cannot be
        recovered from the observation and is exposed explicitly. Candidates
        are already sorted and truncated to ``action_space.n``, so exactly the
        first ``len(candidates)`` actions are valid.
        """

        mask = np.zeros(self.action_space.n, dtype=bool)
        if getattr(self, "terminated", False):
            return mask
        candidates = getattr(self, "candidates", None) or []
        valid_count = min(len(candidates), self.action_space.n)
        if valid_count > 0:
            mask[:valid_count] = True
        return mask

    def get_candidate_well_names(self) -> list[str]:
        """Names of the wells behind actions ``0..len(candidates)-1``.

        Used by greedy demonstrations to translate the well order of a greedy
        plan into action indices of the current step.
        """

        candidates = getattr(self, "candidates", None) or []
        return [str(context.well.name) for context in candidates[: self.action_space.n]]

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Candidate construction (shared with the greedy PlanBuilder)
    # ------------------------------------------------------------------

    def horizon_days(self) -> float:
        """Length of the work window in days."""
        return max(1.0, (self.end_jobs - self.start).total_seconds() / 86400.0)

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

    def _limit_candidates(self, candidates: list[WellPlanContext]) -> list[WellPlanContext]:
        """The ``n_actions`` candidates with the highest NPV, best first."""
        return sorted(
            candidates,
            key=lambda context: (
                self._to_float(context.cost)
                if context.cost is not None
                else float("-inf")
            ),
            reverse=True,
        )[: self.action_space.n]

    def _build_observation(self, candidates: list[WellPlanContext]) -> np.ndarray:
        # A single origin for all rows; otherwise candidates would be expressed
        # in different frames and become incomparable.
        origin = self._plan_now(candidates)
        emb = [self.get_embedding(candidate, origin=origin) for candidate in candidates]
        if not emb:
            return np.zeros(
                (self.action_space.n, self.observation_space.shape[1]),
                dtype=np.float32,
            )

        observation = np.asarray(emb, dtype=np.float32)
        diff = self.action_space.n - observation.shape[0]
        if diff > 0:
            observation = np.pad(
                observation,
                pad_width=((0, diff), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        return observation

    def _reset_profile_state(self) -> None:
        reset_fn = getattr(self.profiler, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def _commit_profile_state(self, context: WellPlanContext) -> None:
        commit_fn = getattr(self.profiler, "commit", None)
        if callable(commit_fn):
            commit_fn(context)
