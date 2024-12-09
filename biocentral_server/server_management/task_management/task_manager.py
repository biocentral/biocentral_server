import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
from typing import Dict, Any, Type

from .task_interface import TaskStatus, TaskInterface, TaskDTO

logger = logging.getLogger(__name__)


class TaskManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._manager = Manager()
        self._task_dtos = self._manager.dict()
        _max_concurrent_tasks = 5  # TODO [CONFIG] Make this configurable from config and UI
        self._executor = ThreadPoolExecutor(max_workers=_max_concurrent_tasks)
        self._lock = threading.Lock()  # TODO [OPTIMIZATION] Maybe use a log-free data structure

    def add_task(self, task: TaskInterface, task_id: str = "") -> str:
        if task_id == "" or "biocentral" not in task_id:  # biocentral: Sanity check
            task_id = self._generate_task_id(task=task.__class__)
        self._task_dtos[task_id] = TaskDTO.pending()
        future = self._executor.submit(self._execute_task, task_id, task)
        future.add_done_callback(lambda f: self._task_completed_callback(task_id, f))
        return task_id

    def get_unique_task_id(self, task: Type) -> str:
        return self._generate_task_id(task=task)

    @staticmethod
    def _generate_task_id(task):
        return f"biocentral-{task.__name__}-{str(uuid.uuid4())}"

    def _execute_task(self, task_id: str, task: TaskInterface):
        self._task_dtos[task_id] = TaskDTO.running()

        def dto_callback(dto: TaskDTO):
            self._update_task_dto_callback(task_id=task_id, task_dto=dto)

        return task.run_task(update_dto_callback=dto_callback)

    def _task_completed_callback(self, task_id: str, future):
        try:
            result = future.result()
            with self._lock:
                self._task_dtos[task_id] = TaskDTO.finished(result)
        except Exception as e:
            logger.error(f"Task {task_id} failed with error: {e}")
            self._task_dtos[task_id] = TaskDTO.failed(str(e))

    def _update_task_dto_callback(self, task_id: str, task_dto: TaskDTO):
        with self._lock:
            self._task_dtos[task_id] = task_dto

    def get_task_status(self, task_id: str) -> TaskStatus:
        return self._task_dtos[task_id].status

    def is_task_finished(self, task_id: str) -> bool:
        return self.get_task_status(task_id) in [TaskStatus.FINISHED, TaskStatus.FAILED]

    def get_task_dto(self, task_id: str) -> Any:
        if task_id in self._task_dtos:
            return self._task_dtos[task_id]
        return TaskDTO.failed(error=f"task {task_id} not found on server!")

    def get_current_number_of_running_tasks(self) -> int:
        return sum(1 for dto in self._task_dtos.values() if dto.status == TaskStatus.RUNNING)

    def __del__(self):
        self._executor.shutdown(wait=True)
