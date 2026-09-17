"""Feature extraction: the 39-dimensional encoding of one candidate well.

An observation of :class:`~hyperactive.env.PlanEnv` is a matrix
``[n_actions, 39]``: one row per candidate (sorted by NPV, best first) padded
with zero rows. See ``docs/method.md`` (section 3) for the definition of every feature.

Design principles shared by all blocks:

* **No absolute time.** Moments are measured in days from the planning front
  (the earliest start among current candidates). Absolute timestamps let a
  network memorise individual wells of the training pool instead of learning
  a rule, because the readiness date largely identifies the cluster.
* **No quantities that grow with the size of the well pool.** Training uses
  small subsets of wells while inference may see a much larger pool; features
  whose scale depends on pool size leave the training distribution and hit the
  observation clipping of ``VecNormalize``.
* **Explicit ratios.** The first layer of an MLP is linear and cannot divide, so
  ratios the decision depends on (value per crew-day, remaining capacity) are
  computed here.
* **Markov state.** Besides the candidate itself the row carries the plan state:
  remaining work window, crew occupancy, how close the plan is to the oil cap.
"""

from __future__ import annotations

import bisect
import calendar
from datetime import datetime
from typing import Any, Optional

import numpy as np

from hyperactive.core import Task

FEATURE_SET = "gen39"
"""Identifier of this encoding; stored in model manifests and checked on load."""

DAY_SECONDS = 86400.0

# Candidate physics, economics, relative time and plan state.
BASE_FEATURE_NAMES = (
    # well physics
    "oil_rate", "liq_rate", "length",
    # candidate economics
    "cost_npv", "cash_flow", "capex", "travel_cost",
    # time, days from the planning front
    "ready_in_days", "has_ready_date",
    "job1_start_days", "job1_duration_days", "job1_travel_days",
    "has_job2", "job2_gap_days", "job2_duration_days", "job2_travel_days",
    "finish_in_days",
    # plan state
    "jobs_window_left_days", "placed_share", "same_cluster",
)

# Cluster structure of the remaining wells, normalised by pool size.
PAD_FEATURE_NAMES = (
    "pad_wells_left_current",    # wells left on the cluster of the last placed well
    "pad_wells_left_candidate",  # wells left on the candidate's cluster
    "pad_closes",                # choosing the candidate closes its cluster
)

VALUE_FEATURE_NAMES = (
    "value_density_norm",  # NPV per crew-day, min-max normalised within the window
    "value_rank",          # share of window candidates with a lower density
)

# Recomputed at every step of the plan.
DYNAMIC_FEATURE_NAMES = (
    "drill_capacity_left",  # free drilling-crew capacity in the rest of the work window
    "gtm_capacity_left",    # the same for GTM crews
    "crew_bottleneck",      # which crew type binds: GTM share of the free capacity
    "oil_window_left",      # headroom under the annual oil cap in the binding year
    "oil_window_after",     # headroom left if this candidate is taken
)

# Computed from the monthly profile arrays, so they do not depend on the
# decline model that produced the profile.
PROFILE_FEATURE_NAMES = (
    "horizon_coverage",  # share of the economic horizon the well produces for
    "decline_year",      # ratio of average daily rate after a year to the first full month
)

# "Drill now or wait for a better well": recomputed at every step.
ARRIVAL_FEATURE_NAMES = (
    # Value of the best well that becomes ready later in the work window versus
    # the candidate, squashed into [0, 1) as a / (a + b): 0 means nothing
    # arrives, 0.5 an equivalent replacement, close to 1 a clearly better one.
    "best_arriving_value_ratio",
    # The price of waiting: crew idle time until that arrival divided by the
    # free drilling-crew capacity of the remaining window. Capacity rather than
    # window length: waiting stops one crew while the others keep drilling.
    "wait_idle_share",
    # Share of upcoming arrivals that are more valuable than the candidate.
    # Ordinal, hence independent of the money scale and robust to many wells
    # sharing the same rate.
    "arrivals_better_share",
)

CONFIG_FEATURE_NAMES = (
    "jobs_window_years",    # length of the work window
    "econ_window_ratio",    # economic horizon / work window, capped at 4
    "job_window_share",     # crew-days the candidate occupies / remaining window
    "finish_window_share",  # candidate finish time / remaining window, in [0, 1]
)

FEATURE_NAMES = (
    BASE_FEATURE_NAMES
    + PAD_FEATURE_NAMES
    + VALUE_FEATURE_NAMES
    + DYNAMIC_FEATURE_NAMES
    + PROFILE_FEATURE_NAMES
    + ARRIVAL_FEATURE_NAMES
    + CONFIG_FEATURE_NAMES
)
EMBEDDING_WIDTH = len(FEATURE_NAMES)
assert EMBEDDING_WIDTH == 39

# Month as a fraction of a year: fixed costs in the arrival value estimate are
# monthly, as in the cost function itself.
MONTHS_PER_DAY = 12.0 / 365.25


class CandidateFeatures:
    """Feature extraction mixed into :class:`~hyperactive.env.PlanEnv`.

    Reads the environment state: ``wells``, ``remaining_wells``,
    ``candidates``, ``plan``, ``manager``, ``start``, ``end``, ``end_jobs``,
    ``cost_function``, ``infra``, ``_constraints`` and ``_last_cluster``.
    """

    # ------------------------------------------------------------------
    # Row assembly
    # ------------------------------------------------------------------

    def get_embedding(self, well_context, origin: Optional[datetime] = None) -> list[float]:
        """The feature row of one candidate, in the order of ``FEATURE_NAMES``."""
        row = (self._relative_embedding(well_context, origin)
               + self.fund_pad_features(well_context)
               + self.value_features(well_context))
        row.extend(self._crew_capacity())
        row.extend(self.oil_window_features(well_context))
        row.extend(self.profile_features(well_context))
        row.extend(self.arrival_features(well_context, origin))
        row.extend(self.config_features(well_context))
        return row

    # ------------------------------------------------------------------
    # Time reference
    # ------------------------------------------------------------------

    def _plan_now(self, candidates=None) -> datetime:
        """Time origin: the earliest start among the current candidates.

        ``start`` cannot serve as the origin: it never moves, so "time from
        start" would differ from absolute time only by a constant shift, which
        normalisation removes. The maximum over ends of assigned jobs does not
        work either: with several crews the latest one runs far ahead and the
        waiting time of candidates becomes negative.

        The earliest available start is "now" from the decision's point of
        view: all candidates are compared to it and to each other.
        """

        candidates = candidates if candidates is not None else getattr(self, "candidates", None)
        starts = [
            candidate.entries[0].start
            for candidate in (candidates or [])
            if candidate.entries
        ]
        return min(starts) if starts else self.start

    def _days_from(self, moment, origin) -> float:
        if moment is None:
            return 0.0
        return (moment - origin).total_seconds() / DAY_SECONDS

    # ------------------------------------------------------------------
    # Base block (20)
    # ------------------------------------------------------------------

    def _relative_embedding(self, well_context, origin: Optional[datetime] = None) -> list[float]:
        """Physics, economics, relative timing of both jobs and plan state.

        A well without a second job is encoded by the ``has_job2`` flag and
        zeros rather than by a sentinel on the time scale, which would eat the
        range of the feature.
        """
        now = origin if origin is not None else self._plan_now()
        well = well_context.well
        entries = list(well_context.entries)
        first = entries[0]
        second = entries[1] if len(entries) > 1 else None
        finish = max(entry.end for entry in entries)

        return [
            # well physics
            self._rate_to_float(well.oil_rate),
            self._rate_to_float(well.liq_rate),
            float(well.length),
            # economics
            self._to_float(well_context.cost),
            self._to_float(well_context.metadata.get("cash_flow", 0.0)),
            self._to_float(well_context.metadata.get("capex", 0.0)),
            float(well_context.metadata.get("travel_cost", 0.0)),
            # relative time, days from the planning front
            self._days_from(well.init_entry_date, now),
            1.0 if well.init_entry_date else 0.0,
            self._days_from(first.start, now),
            self._days_from(first.end, first.start),
            first.travel_time.total_seconds() / DAY_SECONDS,
            1.0 if second is not None else 0.0,
            self._days_from(second.start, first.end) if second is not None else 0.0,
            self._days_from(second.end, second.start) if second is not None else 0.0,
            second.travel_time.total_seconds() / DAY_SECONDS if second is not None else 0.0,
            self._days_from(finish, now),
            # plan state
            self._days_from(self.end_jobs, now),
            len(self.plan.well_plans) / self._initial_well_count,
            1.0 if self._last_cluster is not None
            and str(well.cluster) == self._last_cluster else 0.0,
        ]

    # ------------------------------------------------------------------
    # Cluster block (3)
    # ------------------------------------------------------------------

    def fund_pad_features(self, well_context) -> list[float]:
        """Cluster structure of the remaining wells, normalised by pool size.

        A move between clusters costs far more than a move within one, so what
        matters is not the move itself but how many wells it serves. A binary
        "same cluster" flag cannot tell leaving a cluster with five wells left
        from leaving an exhausted one. Normalising by pool size (rather than a
        fixed cluster scale) keeps the largest clusters distinguishable when
        clusters are bigger than in training.
        """

        total = max(1, len(self.wells or []))
        counts: dict[str, int] = {}
        for well in self.remaining_wells or []:
            key = str(well.cluster)
            counts[key] = counts.get(key, 0) + 1
        on_candidate = counts.get(str(well_context.well.cluster), 0)
        on_current = (counts.get(self._last_cluster, 0)
                      if self._last_cluster is not None else 0)
        return [on_current / total, on_candidate / total,
                # The candidate itself is part of the remainder, so "last on
                # its cluster" means exactly one.
                1.0 if on_candidate <= 1 else 0.0]

    # ------------------------------------------------------------------
    # Value block (2)
    # ------------------------------------------------------------------

    def _occupancy_days(self, well_context) -> float:
        """Crew-days the candidate occupies: jobs plus moves plus the gap between jobs."""
        entries = list(well_context.entries)
        if not entries:
            return 0.0
        first = entries[0]
        span = ((first.end - first.start).total_seconds()
                + first.travel_time.total_seconds()) / DAY_SECONDS
        if len(entries) > 1:
            second = entries[1]
            span += ((second.start - first.end).total_seconds()
                     + (second.end - second.start).total_seconds()
                     + second.travel_time.total_seconds()) / DAY_SECONDS
        return span

    def value_features(self, well_context) -> list[float]:
        """Value density of the candidate and its rank within the window.

        When the work window is tight only part of the pool fits into the plan,
        and the question becomes "maximum NPV per crew-day" rather than
        "maximum NPV". Both values are dimensionless and lie in [0, 1]. The
        rank is added explicitly because the network cannot compute it: the
        context is a mean over rows, which discards order, so there is no
        pairwise comparison of candidates in the architecture.
        """

        def density(context) -> float:
            return self._to_float(context.cost) / max(1.0, self._occupancy_days(context))

        own = density(well_context)
        values = [density(c) for c in (self.candidates or [])] or [own]
        low, high = min(values), max(values)
        # Min-max within the window is robust to negative NPVs, unlike division
        # by the maximum.
        norm = (own - low) / (high - low) if high > low else 0.5
        rank = (sum(1 for v in values if v < own) / (len(values) - 1)
                if len(values) > 1 else 1.0)
        return [norm, rank]

    # ------------------------------------------------------------------
    # Dynamic block (5)
    # ------------------------------------------------------------------

    def _crew_capacity(self) -> tuple[float, float, float]:
        """Free crew capacity in the rest of the work window.

        Static crew counts say how many crews exist but not whether they are
        busy. Idle crews are the main mechanism by which a plan loses value, and
        these three values name it directly: capacity left for drilling, for
        GTM, and which of the two binds.
        """

        now = self._plan_now()
        window_left = max(1.0, self._days_from(self.end_jobs, now))
        states = getattr(self.manager, "_states", None) or {}
        free = {"drill": 0.0, "gtm": 0.0}
        total = {"drill": 0.0, "gtm": 0.0}
        for team, state in states.items():
            kind = "drill" if Task.DRILLING in team.supported_tasks else "gtm"
            busy = 0.0
            for interval in getattr(state, "busy_intervals", ()):
                # Only occupancy inside the remaining window counts: work beyond
                # its end does not consume window capacity.
                start = max(interval.start, now)
                end = min(interval.end, self.end_jobs)
                if end > start:
                    busy += (end - start).total_seconds() / DAY_SECONDS
            total[kind] += window_left
            free[kind] += max(0.0, window_left - busy)
        drill_share = free["drill"] / total["drill"] if total["drill"] else 0.0
        gtm_share = free["gtm"] / total["gtm"] if total["gtm"] else 0.0
        both = free["drill"] + free["gtm"]
        # Close to zero: GTM binds; close to one: drilling binds.
        bottleneck = (free["gtm"] / both) if both > 0 else 0.0
        return drill_share, gtm_share, bottleneck

    def _free_drill_days(self, now) -> float:
        """Free drilling-crew capacity in the remaining window, in crew-days.

        This is what the price of waiting is measured in. While the plan waits
        for one well, one crew is idle and the others keep drilling: with five
        crews the same wait is five times cheaper than with one. Busy crews do
        not count, as they cannot absorb idle time.

        Cached per step: the set of busy intervals changes only when the plan
        grows, so the plan length together with the origin is a valid key.
        """

        marker = (getattr(self, "_episode_serial", 0),
                  len(self.plan.well_plans), now)
        cached = getattr(self, "_free_drill_cache", None)
        if cached is not None and cached[0] == marker:
            return cached[1]

        window_left = max(1.0, self._days_from(self.end_jobs, now))
        states = getattr(self.manager, "_states", None) or {}
        free = 0.0
        for team, state in states.items():
            if Task.DRILLING not in team.supported_tasks:
                continue
            busy = 0.0
            for interval in getattr(state, "busy_intervals", ()):
                start = max(interval.start, now)
                end = min(interval.end, self.end_jobs)
                if end > start:
                    busy += (end - start).total_seconds() / DAY_SECONDS
            free += max(0.0, window_left - busy)

        self._free_drill_cache = (marker, free)
        return free

    def _planned_oil_per_year(self) -> dict[int, float]:
        """Oil of the accepted wells by year; recomputed when the plan grows."""
        key = len(self.plan.well_plans)
        if getattr(self, "_oil_cache_key", None) != key:
            self._oil_cache_key = key
            self._oil_cache = self.plan.get_oil_production_per_year()
        return self._oil_cache

    def _oil_constraints(self) -> list[Any]:
        return [c for c in getattr(self._constraints, "constraints", None) or []
                if type(c).__name__ == "OilConstraint"]

    def oil_window_features(self, well_context) -> list[float]:
        """Headroom under the annual oil cap, now and after taking the candidate.

        The cap filters candidates silently: infeasible ones simply disappear
        from the window, and without this block the policy cannot see how close
        the plan is to the ceiling. Both values are the minimum over constraints
        and years of ``1 - used / bound``; without a cap both are 1.
        """

        constraints = self._oil_constraints()
        if not constraints:
            return [1.0, 1.0]

        planned = self._planned_oil_per_year()
        candidate: dict[int, float] = {}
        try:
            for year, oil in self.plan._monthly_to_yearly(
                    well_context.launch_date, well_context.oil_prod_profile):
                candidate[year] = candidate.get(year, 0.0) + oil
        except (AttributeError, TypeError):
            candidate = {}

        left, after = 1.0, 1.0
        for constraint in constraints:
            for year in set(planned) | set(candidate):
                bound = constraint.get_applicable_bound(year)
                if bound is None or bound.value <= 0:
                    continue
                used = planned.get(year, 0.0)
                left = min(left, max(0.0, 1.0 - used / bound.value))
                after = min(after, max(0.0, 1.0 - (used + candidate.get(year, 0.0))
                                       / bound.value))
        return [left, after]

    # ------------------------------------------------------------------
    # Profile block (2)
    # ------------------------------------------------------------------

    def _month_days(self, launch: datetime, index: int) -> int:
        """Number of days in the calendar month with this profile index."""
        month = launch.month - 1 + index
        return calendar.monthrange(launch.year + month // 12, month % 12 + 1)[1]

    def profile_features(self, well_context) -> list[float]:
        """Horizon coverage and annual decline, from the monthly profile.

        Coverage: the profile is truncated at the end of the economic horizon,
        so a well launched later has a shorter profile. The feature says which
        share of the horizon the well produces for, i.e. how much of its value
        is cut off by the evaluation period.

        Decline: computed from average DAILY rates rather than monthly volumes;
        a ratio of monthly volumes would measure the lengths of calendar months
        instead of the physics. Month 0 is partial (the well starts mid-month),
        so the ratio starts from the first full month and is annualised for
        profiles shorter than a year.
        """

        profile = list(getattr(well_context, "oil_prod_profile", None) or [])
        launch = getattr(well_context, "launch_date", None)
        if not profile or launch is None:
            return [0.0, 1.0]

        horizon_months = max(
            1.0, (self.end - self.start).total_seconds() / DAY_SECONDS / 30.4375)
        coverage = min(1.0, sum(1 for value in profile if value > 0) / horizon_months)

        def daily(index: int) -> float:
            days = self._month_days(launch, index)
            return profile[index] / days if days else 0.0

        decline = 1.0
        if len(profile) > 2 and daily(1) > 0:
            last = min(12, len(profile) - 1)
            ratio = daily(last) / daily(1)
            months = max(1, last - 1)
            if ratio > 0:
                decline = float(ratio ** (11.0 / months))
        return [coverage, min(2.0, max(0.0, decline))]

    # ------------------------------------------------------------------
    # Arrival block (3)
    # ------------------------------------------------------------------

    def _well_value(self, well, available) -> float:
        """A cheap economic estimate of a well that becomes available at ``available``.

        Arrivals are not candidates yet, and building a full context with a
        production profile for every remaining well at every step would cost
        more than training itself. The estimate uses the same quantities as the
        cost function (oil price, unit costs, cost per metre, discount rate) but
        a constant rate instead of a profile::

            V = max(0, (margin_per_day - fixed_per_day) * days_to_horizon_end - capex)
                / (1 + r) ** years_from_project_start

        The absolute value is irrelevant: features use ratios and ranks of such
        estimates. What matters is that the order agrees with the real NPV.
        """

        cost = self.cost_function
        capex_cost = getattr(cost, "capex_cost", None)
        opex_cost = getattr(cost, "opex_cost", None)
        if capex_cost is None or opex_cost is None:
            # A cost function of another kind: fall back to the oil rate.
            return max(0.0, self._rate_to_float(well.oil_rate))

        days = self._days_from(self.end, available)
        if days <= 0:
            return 0.0

        oil = self._rate_to_float(well.oil_rate)
        water = max(0.0, self._rate_to_float(well.liq_rate) - oil)
        margin_per_day = (oil * (cost.oil_price - opex_cost.oil_cost)
                          - water * opex_cost.water_cost)
        fixed_per_day = ((opex_cost.repair_monthly + opex_cost.maintain_monthly)
                         * MONTHS_PER_DAY)

        length = self._to_float(np.nan_to_num(well.length, nan=0.0))
        capex = (capex_cost.build_cost_per_metr.get(str(well.well_type), 0.0) * length
                 + capex_cost.equipment)

        value = (margin_per_day - fixed_per_day) * days - capex
        if value <= 0:
            return 0.0
        years = self._days_from(available, cost.start) / 365.0
        return float(cost._discount(value, max(0.0, years)))

    def _arrival_table(self, now):
        """Arrivals of the step: readiness dates, best estimates and all estimates.

        Only wells that become ready after the origin and before the end of the
        work window are included: wells that are ready already are candidates
        and visible directly, and wells arriving after the window take no part
        in the plan.

        Returns the sorted dates; for every date the two best estimates so far,
        each with its well name and date (the candidate may itself be in the
        table and must be excluded from the comparison); and the sorted list of
        all estimates for the ordinal share.

        Cached per step: the origin is shared by all rows and the remainder
        shrinks by one well per step.
        """

        marker = (getattr(self, "_episode_serial", 0),
                  len(self.remaining_wells or []), now)
        cached = getattr(self, "_arrivals_cache", None)
        if cached is not None and cached[0] == marker:
            return cached[1], cached[2], cached[3]

        rows = []
        for well in self.remaining_wells or []:
            ready = self.infra.get_ready_date(well=well)
            if ready is None or ready <= now or ready >= self.end_jobs:
                continue
            rows.append((ready, self._well_value(well, ready), str(well.name)))
        rows.sort(key=lambda item: item[0])

        # The date is stored with each of the two best estimates so that the
        # waiting time is measured to the arrival that is actually compared,
        # also when the best one is the candidate itself.
        dates = [ready for ready, _, _ in rows]
        best: list[tuple[float, str, Any, float, Any]] = []
        top, top_name, top_date = 0.0, "", None
        second, second_date = 0.0, None
        for ready, value, name in rows:
            if value > top:
                top, top_name, top_date, second, second_date = value, name, ready, top, top_date
            elif value > second:
                second, second_date = value, ready
            best.append((top, top_name, top_date, second, second_date))
        ordered = sorted(value for _, value, _ in rows)

        self._arrivals_cache = (marker, dates, best, ordered)
        return dates, best, ordered

    def arrival_features(self, well_context, origin=None) -> list[float]:
        """Economics of "drill now or wait until a better well is ready".

        When wells become ready over the whole work window, crews can be idle
        for a long time waiting for a ready well, and a good plan takes a strong
        well on the day it becomes ready. Observations that describe only
        ready candidates make "occupy the crew now" and "wait a month for a
        better well" look the same. The three values answer three parts of
        the question: how much more valuable the best upcoming well is, what
        the wait costs, and how often something better arrives at all. The
        first two only make sense together: the gain of waiting against its
        price.
        """

        now = origin if origin is not None else self._plan_now()
        entries = well_context.entries or []
        if not entries:
            return [0.0, 0.0, 0.0]

        dates, best, ordered = self._arrival_table(now)
        own = self._well_value(well_context.well, entries[0].end)
        if not dates:
            return [0.0, 0.0, 0.0]

        better = len(ordered) - bisect.bisect_right(ordered, own)
        better_share = better / len(ordered)

        # The best arrival is searched over the whole remaining work window,
        # not only over the duration of the candidate's jobs; the price of
        # waiting is carried by wait_idle_share.
        top, top_name, top_date, second, second_date = best[-1]
        # The candidate itself is excluded: it is being chosen now anyway.
        if top_name == str(well_context.well.name):
            arriving, arriving_date = second, second_date
        else:
            arriving, arriving_date = top, top_date
        if arriving <= 0 or arriving_date is None:
            return [0.0, 0.0, better_share]

        total = arriving + own
        value_ratio = arriving / total if total > 0 else 0.0

        # Idle time until that arrival relative to the capacity that absorbs it.
        # Without free capacity waiting costs the maximum: the denominator is
        # floored at one crew-day and the share saturates at one.
        idle = max(0.0, self._days_from(arriving_date, now))
        absorbing = max(1.0, self._free_drill_days(now))
        idle_share = min(1.0, idle / absorbing)
        return [value_ratio, idle_share, better_share]

    # ------------------------------------------------------------------
    # Configuration block (4)
    # ------------------------------------------------------------------

    def config_features(self, well_context) -> list[float]:
        """Horizons and the candidate's footprint in the remaining window.

        Only quantities that do not depend on the size of the well pool: the
        work window length, the ratio of horizons, and two per-candidate shares
        of the remaining work window.
        """

        now = self._plan_now()
        window_left = max(1.0, self._days_from(self.end_jobs, now))
        window_years = max(
            0.05, (self.end_jobs - self.start).total_seconds() / DAY_SECONDS / 365.0)
        econ_years = max(
            0.05, (self.end - self.start).total_seconds() / DAY_SECONDS / 365.0)

        entries = list(well_context.entries)
        occupied = sum(
            (entry.end - entry.start).total_seconds() / DAY_SECONDS
            + entry.travel_time.total_seconds() / DAY_SECONDS
            for entry in entries
        )
        finish = max(entry.end for entry in entries)
        return [
            window_years,
            min(4.0, econ_years / window_years),
            min(1.0, occupied / window_left),
            max(0.0, min(1.0, self._days_from(finish, now) / window_left)),
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rate_to_float(value) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            if not value:
                return 0.0
            return float(value[0])
        return float(value)

    @staticmethod
    def _to_float(value) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
