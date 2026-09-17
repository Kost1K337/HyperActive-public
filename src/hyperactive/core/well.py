"""A candidate well of the development plan."""

from datetime import datetime
from typing import List, Optional, Union

from pydantic import ConfigDict
from pydantic.dataclasses import Field, dataclass

from .task import Task


@dataclass(
    slots=True,
    config=ConfigDict(coerce_numbers_to_str=True),
)
class Well:
    name: str
    cluster: str
    field: str
    layer: str
    well_type: str
    oil_rate: Union[List[float], float] = Field(..., description="Initial oil rate, t/day")
    liq_rate: Union[List[float], float] = Field(..., description="Initial liquid rate, t/day")
    length: float = Field(..., description="Total well length, m")
    purpose: Optional[str] = None
    init_entry_date: Optional[datetime] = Field(
        default=None, description="Planned commissioning date from the input data"
    )
    readiness_date: Optional[datetime] = Field(
        default=None, description="Date the pad infrastructure is ready for work"
    )
    depend_from_cluster: Optional[str] = Field(
        default=None, description="The well may only be scheduled after this cluster is finished"
    )

    @property
    def tasks(self) -> tuple[Task, ...]:
        """The task chain encoded by ``well_type`` (e.g. ``ГС+ГРП``)."""
        return tuple(Task.from_code(code.strip()) for code in self.well_type.split("+"))
