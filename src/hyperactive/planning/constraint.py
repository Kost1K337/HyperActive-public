"""Plan-level feasibility constraints."""

from datetime import datetime
from functools import cached_property
from typing import Optional

from pydantic.dataclasses import Field, dataclass

from hyperactive.core import Constraint, Plan, WellPlanContext


@dataclass
class ConstraintManager:
    constraints: list[Constraint] = Field(default_factory=list)

    @cached_property
    def time_bounds(self) -> list[datetime]:
        bounds: set[datetime] = set()
        for constraint in self.constraints:
            bounds.update(
                bound.date for bound in constraint.bounds if bound.date is not None
            )
        return sorted(bounds)

    def get_period_end(
        self,
        current_date: datetime,
    ) -> Optional[datetime]:
        """The first dated bound after ``current_date``."""
        for bound in self.time_bounds:
            if bound > current_date:
                return bound
        return None

    def is_violated(
        self,
        plan: Plan,
        context: WellPlanContext,
    ) -> bool:
        return any(
            constraint.is_violated(plan, context) for constraint in self.constraints
        )


class CapexConstraint(Constraint):
    """Cumulative discounted CAPEX of the plan must not exceed the bound."""

    def is_violated(
        self,
        plan: Plan,
        context: WellPlanContext,
    ) -> bool:
        current_plan_capex = sum(wp.metadata.get("capex", 0.0) for wp in plan.well_plans)
        total_capex = current_plan_capex + context.metadata.get("capex", 0.0)

        for bound in self.bounds:
            if bound.date is None or bound.date > context.launch_date:
                if bound.value < total_capex:
                    return True
        return False


class OilConstraint(Constraint):
    """Annual oil production of the plan must not exceed the bound in any year."""

    def is_violated(
        self,
        plan: Plan,
        context: WellPlanContext,
    ) -> bool:
        oil_tuples = plan._monthly_to_yearly(
            context.launch_date, context.oil_prod_profile
        )
        context_oil_per_year: dict[int, float] = {}
        for year, oil in oil_tuples:
            context_oil_per_year[year] = context_oil_per_year.get(year, 0.0) + oil

        plan_oil_per_year = plan.get_oil_production_per_year()

        for target_year, context_oil in context_oil_per_year.items():
            bound = self.get_applicable_bound(target_year)
            if bound is None:
                continue

            planned_oil = plan_oil_per_year.get(target_year, 0.0)
            if planned_oil + context_oil > bound.value:
                return True
        return False
