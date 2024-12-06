from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict

from .task_utils import run_subtask_util


class TaskStatus(Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3


class TaskInterface(ABC):
    def __init__(self):
        self._status = TaskStatus.PENDING
        self._result = None
        self._error = None

    @abstractmethod
    def run(self) -> Any:
        pass

    def get_task_status(self) -> TaskStatus:
        return self._status

    def set_result(self, result: Any):
        self._result = result
        self._status = TaskStatus.FINISHED

    def set_failed(self, error: str):
        self._error = error
        self._status = TaskStatus.FAILED

    def get_result(self) -> Any:
        return self._result

    @abstractmethod
    def update(self) -> Dict[str, Any]:
        pass

    @staticmethod
    def run_subtask(task) -> Any:
        return run_subtask_util(task)
