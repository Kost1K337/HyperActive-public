"""When a well becomes available for crew work."""

from datetime import datetime
from typing import Protocol

from hyperactive.core import Well


class Infrastructure(Protocol):
    def get_ready_date(
        self,
        well: Well,
    ) -> datetime:
        pass


class SimpleInfrastructure:
    """A well is ready at its ``readiness_date``; without one it is ready immediately."""

    def get_ready_date(
        self,
        well: Well,
    ) -> datetime:
        if well.readiness_date:
            return well.readiness_date
        return datetime.min
