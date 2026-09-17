"""Monthly production profiles of a newly commissioned well."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from hyperactive.core import WellPlanContext


class ProductionProfile(Protocol):
    """Fills ``oil_prod_profile`` and ``liq_prod_profile`` (tonnes per calendar month).

    The profile starts at the launch date (end of the last task) and is
    truncated at ``context.end``, the end of the economic horizon.
    """

    def compute(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        pass


def _next_month(month_start: datetime) -> datetime:
    if month_start.month == 12:
        return month_start.replace(year=month_start.year + 1, month=1, day=1)
    return month_start.replace(month=month_start.month + 1, day=1)


class LinearProductionProfile:
    """Constant rate: monthly volume = rate * producing days in the month."""

    def compute(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        oil_rates = []
        liq_rates = []
        start = context.get_next_available_date()
        end = context.end
        well = context.well

        current_month = context.get_next_available_date().replace(day=1)
        while current_month <= context.end:
            next_month = _next_month(current_month)
            month_end = next_month - timedelta(days=1)

            period_start = max(start, current_month)
            period_end = min(end, month_end)

            days = (period_end - period_start).days + 1

            if days > 0:
                oil_rates.append(well.oil_rate * days)
                liq_rates.append(well.liq_rate * days)

            current_month = next_month

        context.oil_prod_profile = oil_rates
        context.liq_prod_profile = liq_rates
        return context


class ArpsDeclineProductionProfile:
    """Hyperbolic Arps decline.

    ``q(t) = q0 / (1 + b * D * t) ** (1 / b)`` with ``t`` in years since
    launch, evaluated at the first day of each month and multiplied by the
    producing days of that month. The same decline is applied to oil and
    liquid rates.
    """

    def __init__(
        self,
        D: float = 0.175,
        b: float = 1.548,
    ):
        self.D = D
        self.b = b

    def compute(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        oil_rates = []
        liq_rates = []
        start = context.get_next_available_date()
        end = context.end
        well = context.well

        current_month = context.get_next_available_date().replace(day=1)
        while current_month <= context.end:
            next_month = _next_month(current_month)
            month_end = next_month - timedelta(days=1)

            period_start = max(start, current_month)
            period_end = min(end, month_end)

            days = (period_end - period_start).days + 1

            if days > 0:
                t_years = (current_month - start).days / 365.0

                oil_rate = well.oil_rate / ((1 + self.b * self.D * t_years) ** (1 / self.b))
                liq_rate = well.liq_rate / ((1 + self.b * self.D * t_years) ** (1 / self.b))

                oil_rates.append(oil_rate * days)
                liq_rates.append(liq_rate * days)

            current_month = next_month

        context.oil_prod_profile = oil_rates
        context.liq_prod_profile = liq_rates
        return context
