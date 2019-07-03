from typing import List

from optimizer.hyperparameters import QUEUES


class JobRequestResource(object):
    def __init__(self, priority: int, memory: int, cpu: int):
        self.priority: int = priority
        self.memory: int = memory
        self.cpu: int = cpu


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


class RunningJob(object):
    def __init__(self, elapsed_time: int, priority: int, location: str, progress: float, queue_usage_percentage: float,
                 memory_seconds: int, vcore_seconds: int, request_resources: List[JobRequestResource]):
        self.elapsed_time = elapsed_time
        self.priority = priority
        self.location = location
        self.progress = progress
        self.queue_usage_percentage = queue_usage_percentage
        self.memory_seconds = memory_seconds
        self.vcore_seconds = vcore_seconds
        self.request_resources = request_resources

    @property
    def converted_location(self):
        return queue_name_to_index(self.location)


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


def queue_name_to_index(queue_name: str) -> int:
    return QUEUES['names'].index(queue_name)
