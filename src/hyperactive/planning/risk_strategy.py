"""Optional production risk applied per cluster."""

import random
from collections import defaultdict
from typing import Protocol

from hyperactive.core import WellPlanContext


class RiskStrategy(Protocol):
    def apply_risk(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        pass

    def define_risk(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        pass


class ClusterRandomRiskStrategy:
    """With probability ``trigger_chance`` a planned well reduces the oil output of its cluster.

    ``apply_risk`` scales candidate profiles by the accumulated reduction of
    their cluster; ``define_risk`` rolls the event for the well that was just
    chosen. With ``trigger_chance=0`` (the setting used for training and
    evaluation) profiles are left unchanged, but the global ``random`` state
    is still advanced once per planned well.
    """

    def __init__(
        self,
        trigger_chance: float,
        impact: float = 0.2,
    ):
        self.trigger = trigger_chance
        self.impact = impact
        self._affected_clusters: defaultdict[str, float] = defaultdict(float)

    def apply_risk(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        if context.well.cluster in self._affected_clusters:
            reduction = self._affected_clusters[context.well.cluster]
            context.oil_prod_profile = [
                oil * (1 - reduction) for oil in context.oil_prod_profile
            ]
            context.metadata["applied_risk"] = reduction
        return context

    def define_risk(
        self,
        context: WellPlanContext,
    ) -> WellPlanContext:
        if random.random() < self.trigger:
            cluster = context.well.cluster
            current_reduction = self._affected_clusters.get(cluster, 0.0)
            max_impact_remaining = 1.0 - current_reduction

            if max_impact_remaining > 0:
                additional_impact = min(self.impact, max_impact_remaining)
                self._affected_clusters[cluster] += additional_impact
                self._affected_clusters[cluster] = min(
                    self._affected_clusters[cluster], 1.0
                )

            self.apply_risk(context)
        return context
