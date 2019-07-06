from typing import List, Optional


class Task(object):
    def __init__(self, task_id, running_time, current_finish_time):
        self.id = task_id
        self.running_time = running_time
        self.current_finish_time = current_finish_time


class Container(object):
    def __init__(self, start_time: int):
        self.start_time = start_time
        self.tasks: List[Task] = []

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


class JobFinishTimePredictor(object):

    def predict(self, current_time: int, containers: List[Container]) -> int:
        """Predicts the finish time of job"""
        waiting_tasks = []
        for c in containers:
            waiting_tasks.extend(c.pop_waiting_tasks(current_time))
        waiting_tasks.sort(key=lambda item: item.id)

        for task in waiting_tasks:
            earliest_finish_container = self.get_earliest_finish_container(containers)
            earliest_finish_container.add_task(task)

        return max([container.finish_time for container in containers])

    @staticmethod
    def get_earliest_finish_container(containers: List[Container]) -> Container:
        ret: Optional[Container] = None
        for cur in containers:
            if ret is None or ret.finish_time > cur.finish_time:
                ret = cur
        return ret
