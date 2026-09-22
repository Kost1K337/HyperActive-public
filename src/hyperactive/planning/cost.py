"""Well economics: capital and operating costs and the discounted NPV of a candidate."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import numpy as np

from hyperactive.core import Task, Well, WellPlanContext


class CapitalCost(Protocol):
    def compute(
        self,
        well: Well,
    ) -> float:
        pass


class OperationalCost(Protocol):
    def compute(
        self,
        monthly_oil_prod: list[float],
        monthly_water_prod: list[float],
    ) -> list[float]:
        pass


class CostFunction(Protocol):
    """Fills ``context.cost`` (the value the planner maximises) and returns the context."""

    def compute(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        pass


class BaseCapex:
    """CAPEX = cost per metre (by well type) * well length + equipment cost."""

    def __init__(
        self,
        build_cost_per_metr: dict[str, float],
        equipment_cost: float,
    ):
        self.build_cost_per_metr = build_cost_per_metr
        self.equipment = equipment_cost

    def compute(
        self,
        well: Well,
    ) -> float:
        return self.build_cost_per_metr[well.well_type] * well.length + self.equipment


class BaseOpex:
    """Monthly OPEX: variable cost per tonne of oil and water plus fixed monthly costs."""

    def __init__(
        self,
        oil_cost_per_tone: float,
        water_cost_per_tone: float,
        repair_per_year: float,
        maintain_per_year: float,
    ):
        self.oil_cost = oil_cost_per_tone
        self.water_cost = water_cost_per_tone
        self.repair_monthly = repair_per_year / 12
        self.maintain_monthly = maintain_per_year / 12

    def compute(
        self,
        monthly_oil_prod: list[float],
        monthly_water_prod: list[float],
    ) -> list[float]:
        return [
            (
                0
                if (oil == 0 and water == 0)
                else (
                    np.array(oil) * self.oil_cost
                    + np.array(water) * self.water_cost
                    + self.repair_monthly
                    + self.maintain_monthly
                )
            )
            for oil, water in zip(monthly_oil_prod, monthly_water_prod)
        ]


class NPV:
    """Net present value of a single well given its schedule and production profile.

    With launch date ``t0`` (end of the last task), project start ``T0``,
    ``s = (t0 - T0).days / 365`` and discount rate ``r``::

        NPV = sum_m (P * oil_m - opex_m) / (1 + r) ** (s + m / 12)
              - capex / (1 + r) ** s
              - travel_days * travel_cost_per_day

    where ``travel_days`` is the crew move preceding the drilling task. The
    move cost is not discounted.
    """

    def __init__(
        self,
        oil_price_per_tone: float,
        project_start_date: datetime,
        capex_cost: CapitalCost,
        opex_cost: OperationalCost,
        discount_rate: float = 0.125,
        travel_cost_per_day: float = 1500000,
    ):
        self.capex_cost = capex_cost
        self.opex_cost = opex_cost
        self.discount_rate = discount_rate
        self.oil_price = oil_price_per_tone
        self.start = project_start_date
        self.travel_cost_per_day = travel_cost_per_day

    def _discount(
        self,
        cash_flow: float,
        years: int | float,
    ) -> float:
        return cash_flow / (1 + self.discount_rate) ** years

    def compute(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        shift_years = (context.get_next_available_date() - self.start).days / 365

        safe_length = np.nan_to_num(context.well.length, nan=0.0)
        safe_oil_profile = [np.nan_to_num(p, nan=0.0) for p in context.oil_prod_profile]
        safe_liq_profile = [np.nan_to_num(p, nan=0.0) for p in context.liq_prod_profile]

        cost_per_metre = self.capex_cost.build_cost_per_metr.get(context.well.well_type)
        if cost_per_metre is None:
            # A missing price would otherwise make the well free to drill and
            # push it to the top of every plan.
            raise KeyError(
                f"No drilling cost per metre for well type {context.well.well_type!r} "
                f"(well {context.well.name!r}); priced types: "
                f"{sorted(self.capex_cost.build_cost_per_metr)}"
            )
        capex = cost_per_metre * safe_length + self.capex_cost.equipment
        monthly_opex = self.opex_cost.compute(
            monthly_oil_prod=safe_oil_profile,
            monthly_water_prod=[
                np.array(liq) - np.array(oil)
                for liq, oil in zip(safe_liq_profile, safe_oil_profile)
            ],
        )

        monthly_cash_flows = [
            (np.array(oil) * self.oil_price) - np.array(opex)
            for oil, opex in zip(safe_oil_profile, monthly_opex)
        ]
        discounted_cash_flows = sum(
            np.sum(self._discount(cf, shift_years + (month / 12)))
            for month, cf in enumerate(monthly_cash_flows)
        )
        discounted_capex = self._discount(capex, shift_years)

        entry = context.get_entry_by_task(Task.DRILLING)
        travel_cost = (
            entry.travel_time.days * self.travel_cost_per_day if entry else 0.0
        )

        # Move cost scaled by the number of drilling crews already on the
        # target cluster. It is not part of the NPV; the greedy baseline
        # subtracts it when ranking candidates.
        drill_team_penalty = (
            context.metadata.get(f"team_count_{Task.DRILLING.name.lower()}", 0)
            * travel_cost
        )

        context.cost = discounted_cash_flows - discounted_capex - travel_cost

        context.metadata["travel_cost"] = travel_cost
        context.metadata["cash_flow"] = discounted_cash_flows
        context.metadata["capex"] = discounted_capex
        context.metadata["drill_team_penalty"] = drill_team_penalty

        return context
