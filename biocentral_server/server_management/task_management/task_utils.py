import asyncio
from typing import Any


async def run_subtask_util(subtask) -> Any:
    # Separated to task_utils because of circular import
    from .task_manager import TaskManager
    task_manager = TaskManager()
    subtask_id = task_manager.add_task(subtask)
    while not task_manager.is_task_finished(subtask_id):
        await asyncio.sleep(0.1)
    task_result = task_manager.get_task_result(subtask_id)
    return task_result
