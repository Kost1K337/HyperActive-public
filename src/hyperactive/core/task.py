"""Crew tasks and the well-type codes that map onto them."""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum


@dataclass(frozen=True)
class TaskMixin:
    duration: timedelta
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


class Task(TaskMixin, Enum):
    """A unit of crew work with a fixed duration.

    A well type is a ``+``-separated chain of codes, e.g. ``ГС+ГРП`` is a
    horizontal well (drilling) followed by hydraulic fracturing.

    Codes handled by a drilling crew: ``ГС`` horizontal well, ``ННС``
    directional well, ``МЗС`` multilateral well, ``БУРЕНИЕ`` drilling.
    Codes handled by a GTM (well intervention) crew: ``ГРП`` hydraulic
    fracturing.
    """

    DRILLING = timedelta(days=30), "DRILLING", ("ГС", "ННС", "МЗС", "БУРЕНИЕ")
    GTM = timedelta(days=20), "GTM", ("ГРП",)

    @classmethod
    def from_code(cls, code: str) -> "Task":
        normalized_code = code.strip().upper()
        for task in cls:
            accepted_codes = (task.name, task.description, *task.aliases)
            if normalized_code in {
                accepted_code.strip().upper()
                for accepted_code in accepted_codes
                if accepted_code
            }:
                return task
        raise ValueError(f"Invalid task code: {code}")
