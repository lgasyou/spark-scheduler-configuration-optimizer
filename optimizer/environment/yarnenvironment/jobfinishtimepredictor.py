from .yarnmodel import *


# TODO: Implement this class.
class JobFinishTimePredictor(object):

    def __init__(self, job):
        self.job = job

    def real_predict(self, last_time: int, container_additions: List[ContainerAddition], tasks: List[Task]):
        pass

    @staticmethod
    def simulate(last_time: int, container_additions: List[ContainerAddition], current_containers: List[Container]):
        for addition in container_additions:
            current_containers.extend(addition.containers)
            JobFinishTimePredictor.predict(addition.time, current_containers)
        return JobFinishTimePredictor.predict(last_time, current_containers)

    @staticmethod
    def predict(current_time: int, containers: List[Container]) -> int:
        """Predicts the finish time of job"""
        waiting_tasks = []
        for c in containers:
            waiting_tasks.extend(c.pop_waiting_tasks(current_time))
        waiting_tasks.sort(key=lambda item: item.id)

        for task in waiting_tasks:
            earliest_finish_container = JobFinishTimePredictor._get_earliest_finish_container(containers)
            earliest_finish_container.add_task(task)

        return max([container.finish_time for container in containers])

    @staticmethod
    def _get_earliest_finish_container(containers: List[Container]) -> Container:
        ret: Optional[Container] = None
        for cur in containers:
            if ret is None or ret.finish_time > cur.finish_time:
                ret = cur
        return ret
