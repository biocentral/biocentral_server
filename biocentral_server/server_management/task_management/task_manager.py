import asyncio
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, Any

from .task_interface import TaskInterface, TaskStatus

logger = logging.getLogger(__name__)


class TaskManager:
    _instance = None
    _max_concurrent_tasks = 5

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._task_dict = {}
        self._task_result_dict = {}
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent_tasks)
        self._lock = Lock()

    def set_max_concurrent_tasks(self, max_tasks: int):
        self._max_concurrent_tasks = max_tasks
        self._executor._max_workers = max_tasks

    def add_task(self, task: TaskInterface) -> str:
        with self._lock:
            task_id = self._generate_task_id()
            self._task_dict[task_id] = task
            future = self._executor.submit(self._execute_task, task_id)
            future.add_done_callback(lambda f: self._task_completed(task_id, f.result()))
        return task_id

    @staticmethod
    def _generate_task_id():
        return f"biocentral-server-task-{str(uuid.uuid4())}"

    def _execute_task(self, task_id: str):
        task = self._task_dict[task_id]
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(task.run())
            loop.close()
        except Exception as e:
            logger.error(f"Task {task_id} failed with error: {str(e)}")
            # TODO: Implement error handling for the task

    def _task_completed(self, task_id: str, result: Any):
        with self._lock:
            if task_id in self._task_dict:
                del self._task_dict[task_id]

    def get_task_status(self, task_id: str) -> TaskStatus:
        return self._task_dict[task_id].get_task_status() if task_id in self._task_dict else TaskStatus.FINISHED

    def is_task_finished(self, task_id: str) -> bool:
        return self.get_task_status(task_id) in [TaskStatus.FINISHED, TaskStatus.FAILED]

    def get_task_result(self, task_id: str) -> Any:
        if task_id in self._task_dict:
            result = self._task_dict[task_id].get_result()
            if result is not None and self.is_task_finished(task_id):
                del self._task_dict[task_id]
                return result
        return None

    def get_task_update(self, task_id: str) -> Dict:
        return self._task_dict[task_id].update() if task_id in self._task_dict else {}

    def get_current_number_of_running_tasks(self) -> int:
        return len(self._task_dict)

    def __del__(self):
        self._executor.shutdown(wait=True)