from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Generator, Optional

from .task_utils import run_subtask_util


class TaskStatus(Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3


@dataclass
class TaskDTO:
    status: TaskStatus
    error: str
    update: Dict[str, Any]
    on_result_retrieval_hook: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda result: result

    @classmethod
    def pending(cls):
        return TaskDTO(status=TaskStatus.PENDING, error="", update={})

    @classmethod
    def running(cls):
        return TaskDTO(status=TaskStatus.RUNNING, error="", update={})

    @classmethod
    def finished(cls, result: Dict[str, Any], on_result_retrieval_hook: Optional[Callable] = None):
        return TaskDTO(status=TaskStatus.FINISHED, error="",
                       update=result, on_result_retrieval_hook=on_result_retrieval_hook)

    @classmethod
    def failed(cls, error: str):
        return TaskDTO(status=TaskStatus.FAILED, error=error, update={})

    def add_update(self, update: Dict[str, Any]) -> TaskDTO:
        return TaskDTO(status=self.status, error=self.error, update=update)

    def dict(self):
        update = self.update
        if self.status == TaskStatus.FINISHED:
            update = self.on_result_retrieval_hook(update)
        return {"status": self.status.name, "error": self.error, **update}


class TaskInterface(ABC):
    @abstractmethod
    def run_task(self, update_dto_callback: Callable) -> TaskDTO:
        pass

    @staticmethod
    def run_subtask(subtask: TaskInterface) -> Generator[TaskDTO, None, None]:
        yield from run_subtask_util(subtask=subtask)