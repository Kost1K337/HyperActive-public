"""Crews and the pool of crews available to a plan."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable
from uuid import UUID, uuid4

from pydantic.dataclasses import Field, dataclass

from .task import Task


@dataclass(slots=True, frozen=True)
class Team:
    """A crew that can perform a fixed set of tasks."""

    id: UUID = Field(default_factory=uuid4)
    supported_tasks: frozenset[Task] = Field(default_factory=frozenset)


class TeamPool:
    """Crews indexed by the task they can perform."""

    def __init__(self):
        self._task_teams_map: dict[Task, list[Team]] = defaultdict(list)

    def add_team(
        self,
        supported_tasks: Iterable[Task | str],
    ) -> None:
        tasks = frozenset(
            Task.from_code(task) if isinstance(task, str) else task
            for task in supported_tasks
        )
        team = Team(supported_tasks=tasks)
        for task in tasks:
            self._task_teams_map[task].append(team)

    def add_teams(
        self,
        supported_tasks: Iterable[Task | str],
        num_teams: int,
    ) -> None:
        for _ in range(num_teams):
            self.add_team(supported_tasks)

    def get_teams_for_task(self, task: Task) -> list[Team]:
        return self._task_teams_map.get(task, [])

    @property
    def supported_tasks(self) -> set[Task]:
        return set(self._task_teams_map.keys())

    @property
    def teams(self) -> tuple[Team, ...]:
        return tuple(
            {team for team_list in self._task_teams_map.values() for team in team_list}
        )
