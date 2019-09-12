import dataclasses
from typing import List


@dataclasses.dataclass
class Task(object):
    id: int
    running_time: int
    current_finish_time: int = -1


@dataclasses.dataclass
class Container(object):

    start_time: int
    is_active: bool
    tasks: List[Task] = dataclasses.field(default_factory=list)

    def pop_waiting_tasks(self, current_time: int):
        waiting_tasks = []
        length = len(self.tasks)
        for i in range(length - 1, -1, -1):
            task = self.tasks[i]
            task_start_time = task.current_finish_time - task.running_time
            if task_start_time > current_time:
                waiting_tasks.append(task)
                self.tasks.pop(i)

        return waiting_tasks

    def add_task(self, task: Task):
        task.current_finish_time = self.finish_time + task.running_time
        self.tasks.append(task)

    def add_tasks(self, tasks: List[Task]):
        for t in tasks:
            self.add_task(t)

    @property
    def finish_time(self):
        return self.tasks[-1].current_finish_time if len(self.tasks) else self.start_time
