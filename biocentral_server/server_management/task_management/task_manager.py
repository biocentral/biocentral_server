import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
from typing import Dict, Any

from .task_interface import TaskStatus, TaskInterface

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
        self._manager = Manager()
        self._task_status = self._manager.dict()
        self._task_results = self._manager.dict()
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent_tasks)

    def set_max_concurrent_tasks(self, max_tasks: int):
        self._max_concurrent_tasks = max_tasks
        self._executor._max_workers = max_tasks

    def add_task(self, task: TaskInterface) -> str:
        task_id = self._generate_task_id()
        self._task_status[task_id] = TaskStatus.PENDING.name
        future = self._executor.submit(self._execute_task, task_id, task)
        future.add_done_callback(lambda f: self._task_completed(task_id, f))
        return task_id

    def _execute_task(self, task_id: str, task: TaskInterface):
        self._task_status[task_id] = TaskStatus.RUNNING.name
        # TODO Add task callback
        return task.run()

    @staticmethod
    def _generate_task_id():
        return f"biocentral-server-task-{str(uuid.uuid4())}"

    def _task_completed(self, task_id: str, future):
        try:
            result = future.result()
            self._task_results[task_id] = result
            self._task_status[task_id] = TaskStatus.FINISHED.name
        except Exception as e:
            logger.error(f"Task {task_id} failed with error: {str(e)}")
            self._task_status[task_id] = TaskStatus.FAILED.name
            self._task_results[task_id] = str(e)

    def get_task_status(self, task_id: str) -> TaskStatus:
        return TaskStatus[self._task_status.get(task_id, TaskStatus.FINISHED.name)]

    def is_task_finished(self, task_id: str) -> bool:
        return self.get_task_status(task_id) in [TaskStatus.FINISHED, TaskStatus.FAILED]

    def get_task_result(self, task_id: str) -> Any:
        return self._task_results.get(task_id)

    def get_task_update(self, task_id: str) -> Dict:
        status = self._task_status.get(task_id, TaskStatus.FINISHED.name)
        return {"status": status}

    def get_current_number_of_running_tasks(self) -> int:
        return sum(1 for status in self._task_status.values() if status == TaskStatus.RUNNING.name)

    def __del__(self):
        self._executor.shutdown(wait=True)