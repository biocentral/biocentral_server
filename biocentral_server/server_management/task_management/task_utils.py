import time
from typing import Any, Generator


def run_subtask_util(subtask) -> Generator:
    # Separated to task_utils because of circular import
    from .task_manager import TaskManager
    task_manager = TaskManager()
    subtask_id = task_manager.add_task(subtask)
    while not task_manager.is_task_finished(subtask_id):
        time.sleep(0.1)
        task_dto = task_manager.get_task_dto(subtask_id)
        yield task_dto
    task_dto = task_manager.get_task_dto(subtask_id)
    yield task_dto