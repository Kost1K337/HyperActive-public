"""Crew scheduling: assigning the task chain of a well to drilling and GTM crews."""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Optional, Protocol, Tuple, TypeAlias

from hyperactive.core import ScheduleEntry, Task, Team, TeamPool, Well, WellPlanContext


class BaseMovement(Protocol):
    def get_move_days(
        self,
        from_cluster: str,
        to_cluster: str,
    ) -> float:
        pass


class SimpleTeamMovement:
    """One day within a cluster, fourteen days between clusters."""

    def get_move_days(
        self,
        from_cluster: str,
        to_cluster: str,
    ) -> float:
        if from_cluster == to_cluster:
            return 1
        return 14


@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float
    z: float

    def distance_to(self, other: "Coordinate") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )


@dataclass
class DistanceTeamMovement:
    """Move time between clusters: a fixed rig-down/rig-up time plus distance / speed.

    ``days = min_days_between_clusters + distance_m / (team_speed_kmh * 1000) / 24``.
    Moves within one cluster take ``same_cluster_move_days``; clusters without
    coordinates fall back to ``min_days_between_clusters``.
    """

    cluster_coordinates: dict[str, Coordinate] = field(default_factory=dict)
    min_days_between_clusters: float = 90
    team_speed_kmh: float = 15
    same_cluster_move_days: float = 1

    @classmethod
    def from_dicts(
        cls,
        clusters: list[dict[str, Any]],
        **kwargs,
    ) -> "DistanceTeamMovement":
        required_keys = {"cluster", "x", "y", "z"}
        for entry in clusters:
            if not required_keys.issubset(entry.keys()):
                raise ValueError(
                    "Each dictionary must contain keys: 'cluster', 'x', 'y', 'z'."
                )

        coordinates = {
            str(entry["cluster"]): Coordinate(
                entry["x"],
                entry["y"],
                entry["z"],
            )
            for entry in clusters
        }

        return cls(cluster_coordinates=coordinates, **kwargs)

    def get_move_days(
        self,
        from_cluster: str,
        to_cluster: str,
    ) -> float:
        if from_cluster == to_cluster:
            return self.same_cluster_move_days

        try:
            from_coor = self.cluster_coordinates[from_cluster]
            to_coor = self.cluster_coordinates[to_cluster]
        except KeyError:
            return self.min_days_between_clusters

        distance_meters = from_coor.distance_to(to_coor)
        travel_days = (distance_meters / (self.team_speed_kmh * 1000)) / 24

        return self.min_days_between_clusters + travel_days


@dataclass
class BusyInterval:
    start: datetime
    end: datetime
    interval_type: Literal["work", "travel"]
    # Cluster of a work interval (empty for travel intervals). Used to find
    # where the crew actually has to come from before a neighbouring job.
    cluster: str = ""


@dataclass
class TeamState:
    busy_intervals: list[BusyInterval] = field(default_factory=list)
    current_cluster: Optional[str] = None


TaskLimits: TypeAlias = dict[Task, int]
YearlyLimits: TypeAlias = dict[int, TaskLimits]


class BaseTeamManager(ABC):
    """Holds the calendar of every crew.

    ``get_assignments`` computes a tentative schedule for a candidate without
    changing crew state; ``assign`` commits the schedule of the chosen
    candidate.
    """

    def __init__(
        self,
        team_pool: TeamPool,
        movement: BaseMovement = SimpleTeamMovement(),
        enable_team_count: bool = True,
        limits: Optional[YearlyLimits] = None,
        min_gap_after_predecessor: timedelta = timedelta(days=1),
    ):
        self.team_pool = team_pool
        self.movement = movement
        # Minimum gap between the well barrier (readiness for the first task,
        # end of the previous task for the following ones) and the start of
        # work. One day by default: fracturing may start the day after drilling.
        self.min_gap_after_predecessor = min_gap_after_predecessor
        self._states: dict[Team, TeamState] = {
            team: TeamState() for team in self.team_pool.teams
        }
        self.enable_team_count = enable_team_count
        # Number of jobs committed to each crew. Grows only in assign(), i.e.
        # reflects accepted decisions, not speculative candidate evaluation.
        # Used as a tie-breaker when several crews finish at the same time.
        self._assignment_counts: dict[Team, int] = defaultdict(int)
        self._usage_counts: dict[int, dict[Task, set[Team]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.limits = limits or {}

    def _check_limit(self, task: Task, year: int, team: Team) -> bool:
        """Whether ``team`` may work on ``task`` in ``year`` under the yearly crew limits."""
        if not self.limits:
            return True

        year_limits = self.limits.get(year)
        if not year_limits:
            return True

        max_count = year_limits.get(task)
        if max_count is None:
            return True

        if team in self._usage_counts[year][task]:
            return True

        return len(self._usage_counts[year][task]) < max_count

    def _record_usage(self, task: Task, assignment_year: int, team: Team) -> None:
        relevant_years = sorted([y for y in self.limits.keys() if y >= assignment_year])

        for year_to_record in relevant_years:
            year_limits = self.limits.get(year_to_record, {})
            if year_limits.get(task) is None:
                continue

            if self._check_limit(task, year_to_record, team):
                self._usage_counts[year_to_record][task].add(team)

    @abstractmethod
    def get_assignments(self, context: WellPlanContext) -> WellPlanContext:
        pass

    @abstractmethod
    def assign(self, context: WellPlanContext) -> None:
        pass

    def _count_teams_on_cluster(
        self,
        context: WellPlanContext,
        task: Optional[Task] = None,
        team: Optional[Team] = None,
    ) -> WellPlanContext:
        metadata_key = f"team_count_{task.name.lower()}" if task else "team_count"
        count = self._count_teams_on_cluster_by_task(context.well.cluster, task, team)
        context.metadata[metadata_key] = count
        return context

    def _count_teams_on_cluster_by_task(
        self,
        cluster: str,
        task: Optional[Task] = None,
        team: Optional[Team] = None,
    ) -> int:
        return sum(
            1
            for _team, team_state in self._states.items()
            if (
                team_state.current_cluster == cluster
                and (task is None or task in _team.supported_tasks)
                and _team != team
            )
        )


class TeamManager(BaseTeamManager):
    def get_assignments(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        """Schedule every task of the well on the crew that finishes it earliest."""
        tasks = context.well.tasks

        for task in tasks:
            if task not in self.team_pool.supported_tasks:
                raise ValueError(f"Task '{task.name}' is not supported by any team")

            candidates = []

            for team in self.team_pool.get_teams_for_task(task):
                state = self._states[team]

                travel_time = self._get_travel_time(state, context.well)
                start_time = self._find_available_start_time(
                    task=task, team=team, travel_time=travel_time, context=context
                )

                if start_time is None:
                    continue
                start_time = start_time + travel_time
                end_time = start_time + task.duration
                candidates.append((start_time, end_time, team, travel_time))

            if not candidates:
                # The task could not be placed. The following tasks of the
                # chain (e.g. fracturing after drilling) depend on it and
                # cannot be scheduled either.
                break

            # Primary key: the earliest finish. On ties (moves run in
            # parallel, so all crews may reach the same barrier) take the
            # least loaded crew; otherwise jobs would pile up on the first
            # crew of the pool purely because of iteration order.
            best_start, best_end, best_team, travel_time = min(
                candidates,
                key=lambda candidate: (
                    candidate[1],
                    self._assignment_counts[candidate[2]],
                ),
            )

            # The travel time above was measured from the last ASSIGNED job
            # (state.current_cluster), not from the job adjacent IN TIME: the
            # slot may lie between two accepted jobs. The move cost is derived
            # from this field, so record the move that actually happens.
            travel_time = self._actual_travel(
                best_team, str(context.well.cluster), best_start
            )

            context.entries.append(
                ScheduleEntry(
                    team=best_team,
                    task=task,
                    start=best_start,
                    end=best_end,
                    travel_time=travel_time,
                )
            )

            if self.enable_team_count:
                self._count_teams_on_cluster(context, task, best_team)

        return context

    def _work_intervals(self, team: Team) -> list[BusyInterval]:
        state = self._states[team]
        return sorted(
            (iv for iv in state.busy_intervals if iv.interval_type == "work"),
            key=lambda iv: iv.start,
        )

    def _fits_between_neighbours(
        self, team: Team, cluster: str, start_dt: datetime, end_dt: datetime,
    ) -> bool:
        """Whether the crew can reach the slot and then reach its next job in time."""
        previous: Optional[Tuple[datetime, str]] = None
        following: Optional[Tuple[datetime, str]] = None
        for interval in self._work_intervals(team):
            if interval.end <= start_dt:
                if previous is None or interval.end > previous[0]:
                    previous = (interval.end, interval.cluster)
            elif interval.start >= end_dt:
                if following is None or interval.start < following[0]:
                    following = (interval.start, interval.cluster)

        if previous is not None and previous[1] != cluster:
            need = timedelta(days=self.movement.get_move_days(
                from_cluster=previous[1], to_cluster=cluster))
            if start_dt - previous[0] < need:
                return False
        if following is not None and following[1] != cluster:
            need = timedelta(days=self.movement.get_move_days(
                from_cluster=cluster, to_cluster=following[1]))
            if following[0] - end_dt < need:
                return False
        return True

    def _earliest_feasible(
        self, team: Team, cluster: str, task: Task,
        travel_time: timedelta, not_before: datetime,
    ) -> datetime:
        """The earliest slot that conflicts neither with busy time nor with crew moves.

        Returning None is not allowed here: it would stop crew selection and
        the well would enter the plan with drilling but without fracturing.
        """
        state = self._states[team]
        entries = self._work_intervals(team)
        candidate = not_before
        for _ in range(len(entries) + len(state.busy_intervals) + 2):
            start_dt = candidate + travel_time
            end_dt = start_dt + task.duration
            conflict = next(
                (iv for iv in sorted(state.busy_intervals, key=lambda iv: iv.start)
                 if not (end_dt <= iv.start or start_dt >= iv.end)),
                None,
            )
            if conflict is not None:
                candidate = conflict.end
                continue
            # Move from the previous job in time: if it cannot be made, shift
            # the start by exactly the missing time.
            previous = max(
                (iv for iv in entries if iv.end <= start_dt),
                key=lambda iv: iv.end, default=None,
            )
            if previous is not None and previous.cluster != cluster:
                need = timedelta(days=self.movement.get_move_days(
                    from_cluster=previous.cluster, to_cluster=cluster))
                if start_dt - previous.end < need:
                    candidate = previous.end + need - travel_time
                    continue
            # The crew must also be able to leave the slot for its next job in
            # time; otherwise inserting into a free window breaks an accepted move.
            following = min(
                (iv for iv in entries if iv.start >= end_dt),
                key=lambda iv: iv.start, default=None,
            )
            if following is not None and following.cluster != cluster:
                need = timedelta(days=self.movement.get_move_days(
                    from_cluster=cluster, to_cluster=following.cluster))
                if following.start - end_dt < need:
                    candidate = following.start
                    continue
            return candidate
        return candidate

    def _actual_travel(self, team: Team, cluster: str, start: datetime) -> timedelta:
        """The move the crew actually makes before working in this slot."""
        previous = max(
            (iv for iv in self._work_intervals(team) if iv.end <= start),
            key=lambda iv: iv.end, default=None,
        )
        if previous is None:
            # First job of the crew: it is already on site.
            return timedelta(days=self.movement.get_move_days(
                from_cluster=cluster, to_cluster=cluster))
        return timedelta(days=self.movement.get_move_days(
            from_cluster=previous.cluster, to_cluster=cluster))

    def assign(self, context: WellPlanContext) -> None:
        """Commit the schedule of ``context`` to the crew calendars.

        For every entry a travel interval and a work interval are added to the
        crew calendar, the crew's current cluster is updated and the yearly
        crew limits are recorded.
        """
        for entry in context.entries:
            state = self._states.get(entry.team, TeamState())
            travel_time = self._get_travel_time(state, context.well)

            state.busy_intervals.append(
                BusyInterval(
                    start=entry.start - travel_time,
                    end=entry.start,
                    interval_type="travel",
                )
            )
            state.busy_intervals.append(
                BusyInterval(
                    start=entry.start, end=entry.end, interval_type="work",
                    cluster=str(context.well.cluster),
                )
            )
            state.busy_intervals.sort(key=lambda iv: iv.start)

            state.current_cluster = context.well.cluster
            self._assignment_counts[entry.team] += 1
            self._record_usage(entry.task, entry.end.year, entry.team)

    def _get_travel_time(
        self,
        state: TeamState,
        well: Well,
    ) -> timedelta:
        move_days = self.movement.get_move_days(
            from_cluster=state.current_cluster or well.cluster,
            to_cluster=well.cluster,
        )
        return timedelta(days=move_days)

    def _find_available_start_time(
        self, task: Task, team: Team, travel_time: timedelta, context: WellPlanContext
    ) -> Optional[datetime]:
        """A slot from ``_raw_available_start_time`` checked against moves to time neighbours.

        The raw search inserts the job into any gap that fits ``travel_time``
        before and after it, but ``travel_time`` is measured from the cluster of
        the last ASSIGNED job, not from the job that ends up adjacent IN TIME.
        If the slot is physically infeasible it is shifted by
        ``_earliest_feasible``.
        """
        cluster = str(context.well.cluster)
        candidate = self._raw_available_start_time(task, team, travel_time, context)
        if candidate is None:
            return None

        start_dt = candidate + travel_time
        end_dt = start_dt + task.duration
        if self._fits_between_neighbours(team, cluster, start_dt, end_dt):
            return candidate
        return self._earliest_feasible(team, cluster, task, travel_time, candidate)

    def _raw_available_start_time(
        self, task: Task, team: Team, travel_time: timedelta, context: WellPlanContext
    ) -> Optional[datetime]:
        """The first feasible start of the crew move towards the task.

        1. The occupied span is ``travel_time + task.duration``.
        2. The well barrier (``context.get_next_available_date()``) is the moment
           before which work may not start: readiness of the well for the first
           task, the end of the previous task of the same well for the following
           ones (fracturing cannot start before drilling ends).
        3. A crew move may run in parallel with work on the well: a GTM crew
           can leave another cluster while drilling is in progress and be on
           site when it ends. The earliest move start (``travel_floor``) is
           therefore shifted back by the travel time, and the barrier applies to
           the start of work.
        4. Free gaps are checked in order: before the first busy interval,
           between consecutive intervals, and finally after the last one.

        Every candidate goes through ``slot_ok``, which raises it to
        ``travel_floor``, applies the yearly crew limits and rejects overlaps
        with the crew's own busy time.
        """
        state = self._states[team]
        required_interval = travel_time + task.duration

        barrier = context.get_next_available_date()
        if context.entries:
            # A preceding task exists on this well. Work starts no earlier than
            # min_gap_after_predecessor after it; the move runs in parallel.
            travel_floor = barrier + self.min_gap_after_predecessor - travel_time
        else:
            # First task of the well. The barrier is infrastructure readiness;
            # there is nowhere to travel before it.
            travel_floor = barrier

        last_end = max((iv.end for iv in state.busy_intervals), default=travel_floor)
        base = max(last_end, travel_floor)

        intervals = sorted(state.busy_intervals, key=lambda iv: iv.start)

        def slot_ok(start_dt: datetime) -> Optional[datetime]:
            start_dt = max(start_dt, travel_floor)
            if not self._check_limit(task, (start_dt + travel_time).year, team):
                return None
            end_dt = start_dt + required_interval
            for iv in intervals:
                if not (end_dt <= iv.start or start_dt >= iv.end):
                    return None
            return start_dt

        if intervals:
            first = intervals[0]
            if travel_floor + required_interval <= first.start:
                candidate = slot_ok(travel_floor)
                if candidate is not None:
                    return candidate
            for prev_iv, next_iv in zip(intervals, intervals[1:]):
                candidate_start = max(prev_iv.end, travel_floor)
                if candidate_start + required_interval <= next_iv.start:
                    candidate = slot_ok(candidate_start)
                    if candidate is not None:
                        return candidate

        return slot_ok(base)
