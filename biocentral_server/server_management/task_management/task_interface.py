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
    @abstractmethod
    def run(self) -> Any:
        pass

    @abstractmethod
    def get_status_update(self) -> Dict[str, Any]:
        pass