from typing import List

from optimizer.environment.spark import calculationmodel


class JobFinishTimePredictor(object):

    def __init__(self):
        self.containers: List[calculationmodel.Container] = []

    def simulate(self, container_additions: List[calculationmodel.Container], tasks: List[calculationmodel.Task]):
        container_additions.sort(key=lambda item: item.start_time)
        container_additions = [c for c in container_additions if c.is_active]
        if len(container_additions) == 0:
            return 0

        tasks.sort(key=lambda item: item.id)
        container_additions[0].add_tasks(tasks)
        for addition in container_additions:
            self.containers.append(addition)
            self.simulate_step(addition.start_time)

        return self.current_time_delay

    def simulate_step(self, current_time: int):
        """Predicts the finish time of job"""
        waiting_tasks = []
        for c in self.containers:
            waiting_tasks.extend(c.pop_waiting_tasks(current_time))
        waiting_tasks.sort(key=lambda item: item.id)

        for task in waiting_tasks:
            earliest_finish_container = self._get_earliest_finish_container()
            earliest_finish_container.add_task(task)

    @property
    def current_time_delay(self):
        return max([container.finish_time for container in self.containers])

    def _get_earliest_finish_container(self):
        ret = None
        for cur in self.containers:
            if ret is None or ret.finish_time > cur.finish_time:
                ret = cur
        return ret
