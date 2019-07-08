from typing import List


from optimizer.hyperparameters import QUEUES


class Task(object):
    def __init__(self, task_id, running_time, current_finish_time):
        self.id = task_id
        self.running_time = running_time
        self.current_finish_time = current_finish_time


class Container(object):
    def __init__(self, start_time: int, state: str):
        self.start_time = start_time
        self.state = state
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


class JobRequestResource(object):
    def __init__(self, priority: int, memory: int, cpu: int):
        self.priority: int = priority
        self.memory: int = memory
        self.cpu: int = cpu


# TODO: We may need more information to calculate its predicted time delay.
class WaitingJob(object):
    def __init__(self, elapsed_time: int, priority: int,
                 location: str, request_resources: List[JobRequestResource]):
        self.elapsed_time = elapsed_time
        self.priority = priority
        self.location = location
        self.request_resources = request_resources

    @property
    def converted_location(self):
        return queue_name_to_index(self.location)

    # TODO: Implement this function
    @property
    def predicted_time_delay(self):
        """If this job doesn't start in this episode, its predicted time delay will be 0."""
        return 1


class RunningJob(object):
    def __init__(self, elapsed_time: int, priority: int, location: str, progress: float, queue_usage_percentage: float,
                 request_resources: List[JobRequestResource], containers: List[Container]):
        self.elapsed_time = elapsed_time
        self.priority = priority
        self.location = location
        self.progress = progress
        self.queue_usage_percentage = queue_usage_percentage
        self.request_resources = request_resources
        self.containers = containers

    @property
    def converted_location(self):
        return queue_name_to_index(self.location)

    # TODO: Implement this function
    @property
    def predicted_time_delay(self):
        return 1


class FinishedJob(object):
    def __init__(self, elapsed_time: int):
        self.elapsed_time = elapsed_time


class Resource(object):
    def __init__(self, vcore_num: int, mem: int):
        self.vcore_num: int = vcore_num         # vCore
        self.mem: int = mem                     # Memory(GB)


class QueueConstraint(object):
    def __init__(self, name: str, used_capacity: float, capacity: float, max_capacity: float):
        self.name = name
        self.used_capacity = used_capacity
        self.capacity = capacity
        self.max_capacity = max_capacity

    @property
    def converted_name(self):
        return queue_name_to_index(self.name)


class State(object):
    def __init__(self, waiting_jobs: List[WaitingJob], running_jobs: List[RunningJob],
                 resources: List[Resource], constraints: List[QueueConstraint]):
        self.waiting_jobs = waiting_jobs
        self.running_jobs = running_jobs
        self.resources = resources
        self.constraints = constraints


class ContainerAddition(object):
    def __init__(self, add_time: int, num_containers: int):
        self.time = add_time
        self.containers = [Container(add_time) for _ in range(num_containers)]


def queue_name_to_index(queue_name: str) -> int:
    return QUEUES['names'].index(queue_name)
