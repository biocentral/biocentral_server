from __future__ import annotations

import asyncio
import threading

from enum import Enum
from typing import Dict, Any
from abc import ABC, abstractmethod

from .task_utils import run_subtask_util

try:
    import torch.multiprocessing as mp
except ImportError:
    import multiprocessing as mp


class TaskStatus(Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3


class TaskInterface(ABC):

    @abstractmethod
    async def run(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def _run_task(self):
        # Implement the actual task logic here
        pass

    @abstractmethod
    def get_task_status(self) -> TaskStatus:
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    async def run_subtask(subtask: TaskInterface) -> Any:
        await run_subtask_util(subtask=subtask)


class ThreadedTask(TaskInterface, ABC):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.status = TaskStatus.PENDING
        self._result = None

    async def run(self) -> Any:
        self.thread = threading.Thread(target=self._run_in_thread)
        self.thread.start()

    def _run_in_thread(self):
        asyncio.run(self._run_task())

    def get_result(self) -> Any:
        return self._result

    def get_task_status(self) -> TaskStatus:
        if not self.thread:
            return TaskStatus.PENDING
        if self.thread.is_alive():
            return TaskStatus.RUNNING
        return TaskStatus.FINISHED


class MultiprocessingTask(TaskInterface, ABC):

    def __init__(self):
        super().__init__()
        self.process = None
        self.status = TaskStatus.PENDING
        self.result_queue = mp.Queue()

    async def run(self) -> Any:
        self.process = mp.Process(target=self._run_in_process)
        self.process.start()

    def _run_in_process(self):
        result = asyncio.run(self._run_task())
        self.result_queue.put(result)

    def get_result(self) -> Any:
        return self.result_queue.get()

    def get_task_status(self) -> TaskStatus:
        if not self.process:
            return TaskStatus.PENDING
        if self.process.is_alive():
            return TaskStatus.RUNNING
        return TaskStatus.FINISHED
