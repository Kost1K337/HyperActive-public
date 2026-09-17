"""Schedule entries, per-well plan contexts, the plan and constraint primitives."""

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Optional, TypeVar
from uuid import UUID, uuid4

from pydantic.dataclasses import Field, dataclass

from .task import Task
from .team import Team
from .well import Well


@dataclass(slots=True, frozen=True)
class ScheduleEntry:
    """One task of one well assigned to one crew."""

    task: Task
    team: Team
    start: datetime
    end: datetime
    travel_time: timedelta

    def __str__(self):
        return (
            f"  Task: {self.task.name}\n"
            f"  Team: {self.team.id}\n"
            f"  Timeframe: {self.start.strftime('%Y-%m-%d %H:%M')} - {self.end.strftime('%Y-%m-%d %H:%M')}\n"
            f"  Duration: {(self.end - self.start).days} days\n"
            f"  Travel time: {self.travel_time} days"
        )


@dataclass(slots=True)
class WellPlanContext:
    """A well together with its tentative schedule, production and economics.

    Candidates of a planning step are contexts; the chosen one is appended to
    the plan as is.
    """

    well: Well
    start: datetime
    end: datetime
    entries: list[ScheduleEntry] = Field(default_factory=list)
    cost: Optional[float] = Field(default=None)
    npv_value: Optional[float] = None
    oil_prod_profile: list[float] = Field(
        default_factory=list,
        description="Monthly oil production, t",
    )
    liq_prod_profile: list[float] = Field(
        default_factory=list,
        description="Monthly liquid production, t",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Storage for extra information",
    )

    def get_next_available_date(self) -> datetime:
        """End of the last scheduled task, or ``start`` if nothing is scheduled."""
        return max(entry.end for entry in self.entries) if self.entries else self.start

    def get_entry_by_task(self, task: Task) -> Optional[ScheduleEntry]:
        for entry in self.entries:
            if task == entry.task:
                return entry
        return None

    @property
    def launch_date(self):
        """Commissioning date: the end of the last task."""
        if self.entries:
            return max(entry.end for entry in self.entries)
        raise ValueError("Well has not been planned yet")


KeyType = TypeVar("KeyType")


@dataclass(slots=True)
class Plan:
    """An ordered list of planned wells."""

    id: UUID = Field(default_factory=uuid4)
    well_plans: list[WellPlanContext] = Field(default_factory=list)

    @property
    def start_date(self):
        return min(wp.start for wp in self.well_plans)

    @property
    def end_date(self):
        return max(wp.end for wp in self.well_plans)

    def add_context(
        self,
        context: WellPlanContext,
    ) -> None:
        self.well_plans.append(context)

    def total_profit(self) -> float:
        """Plan NPV: the sum of the per-well NPVs."""
        return sum(wp.cost for wp in self.well_plans if wp.cost is not None)

    def total_oil(self) -> float:
        return sum(self.get_oil_production_per_year().values())

    def get_all_entries(self) -> list[ScheduleEntry]:
        entries = []
        for context in self.well_plans:
            entries.extend(context.entries)
        return entries

    def __str__(self):
        well_plan_strs = []
        for wp in self.well_plans:
            parts = [
                f"Well: {wp.well.name}",
                f"Cluster: {wp.well.cluster}",
                f"Purpose: {wp.well.purpose}",
                f"Well type: {wp.well.well_type}",
                f"Start: {wp.start.strftime('%Y-%m-%d %H:%M')}",
                f"End: {wp.end.strftime('%Y-%m-%d %H:%M')}",
                f"Metadata: {wp.metadata}",
                "Entries:",
            ]
            parts.extend(str(entry) for entry in wp.entries)
            if wp.cost is not None:
                parts.append(f"Cost: {wp.cost}")
            well_plan_strs.append("\n".join(parts))
        return f"\n{'=' * 30}\n".join(well_plan_strs)

    def _aggregate_production(
        self, extractor: Callable[[WellPlanContext], Iterable[tuple[KeyType, float]]]
    ) -> dict[KeyType, float]:
        aggregated: defaultdict[KeyType, float] = defaultdict(float)
        for wp in self.well_plans:
            try:
                _ = wp.launch_date
                for key, value in extractor(wp):
                    aggregated[key] += value
            except ValueError:
                continue
        return dict(sorted(aggregated.items()))

    def get_oil_production_per_year(self) -> dict[int, float]:
        def extractor(wp: WellPlanContext) -> list[tuple[int, float]]:
            return self._monthly_to_yearly(wp.launch_date, wp.oil_prod_profile)

        return self._aggregate_production(extractor)

    def get_well_start_per_year(self) -> dict[int, int]:
        def extractor(wp: WellPlanContext) -> list[tuple[int, float]]:
            return [(wp.launch_date.year, 1.0)]

        aggregated_float = self._aggregate_production(extractor)
        return {k: int(v) for k, v in aggregated_float.items()}

    def get_capex_per_year(self) -> dict[int, float]:
        def extractor(wp: WellPlanContext) -> list[tuple[int, float]]:
            return [(wp.launch_date.year, wp.metadata.get("capex", 0.0))]

        return self._aggregate_production(extractor)

    def _monthly_to_yearly(
        self, launch_date: datetime, production: list[float]
    ) -> list[tuple[int, float]]:
        """Map the i-th month of a profile started at ``launch_date`` to its calendar year."""
        launch_year = launch_date.year
        launch_month = launch_date.month
        return [
            (launch_year + (launch_month + idx - 1) // 12, prod)
            for idx, prod in enumerate(production)
        ]


@dataclass
class ConstraintBound:
    """A bound value, either for one calendar year (``date`` set) or for every year."""

    value: float
    date: Optional[datetime] = Field(default=None)

    @property
    def year(self) -> Optional[int]:
        return self.date.year if self.date is not None else None


@dataclass
class Constraint(ABC):
    """A feasibility rule a candidate must satisfy to be added to the plan."""

    bounds: list[ConstraintBound]

    def __post_init__(self):
        self.bounds = [
            ConstraintBound(**bound) if isinstance(bound, dict) else bound
            for bound in self.bounds
        ]

    @abstractmethod
    def is_violated(self, plan: Plan, context: WellPlanContext) -> bool:
        pass

    def get_applicable_bound(self, year: int) -> Optional[ConstraintBound]:
        """The tightest bound that applies to ``year``."""
        specific_bounds = [b for b in self.bounds if b.year and b.year == year]
        general_bounds = [b for b in self.bounds if b.year is None]

        min_specific = min(specific_bounds, key=lambda b: b.value, default=None)
        min_general = min(general_bounds, key=lambda b: b.value, default=None)

        if min_specific and min_general:
            return (
                min_specific if min_specific.value <= min_general.value else min_general
            )
        return min_specific or min_general
