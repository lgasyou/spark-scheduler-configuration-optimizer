import dataclasses
from typing import List

from optimizer.hyperparameters import QUEUES


@dataclasses.dataclass
class ApplicationRequestResource(object):
    priority: int
    memory: int
    cpu: int


@dataclasses.dataclass
class WaitingApplication(object):

    application_id: str
    elapsed_time: int
    priority: int
    location: str
    predicted_time_delay: int
    request_resources: List[ApplicationRequestResource] = dataclasses.field(default_factory=list)

    @property
    def converted_location(self):
        return queue_name_to_index(self.location)


@dataclasses.dataclass
class RunningApplication(object):

    application_id: str
    elapsed_time: int
    priority: int
    location: str
    progress: float
    queue_usage_percentage: float
    predicted_time_delay: int
    request_resources: List[ApplicationRequestResource] = dataclasses.field(default_factory=list)

    @property
    def converted_location(self):
        return queue_name_to_index(self.location)


@dataclasses.dataclass
class FinishedApplication(object):
    started_time: int
    finished_time: int
    elapsed_time: int


@dataclasses.dataclass
class Resource(object):
    vcore_num: int      # vCore
    mem: int            # Memory(GB)


@dataclasses.dataclass
class QueueConstraint(object):

    name: str
    used_capacity: float
    capacity: float
    max_capacity: float

    @property
    def converted_name(self):
        return queue_name_to_index(self.name)


@dataclasses.dataclass
class State(object):
    waiting_apps: List[WaitingApplication] = dataclasses.field(default_factory=list)
    running_apps: List[RunningApplication] = dataclasses.field(default_factory=list)
    resources: List[Resource] = dataclasses.field(default_factory=list)
    constraints: List[QueueConstraint] = dataclasses.field(default_factory=list)


def queue_name_to_index(queue_name: str) -> int:
    return QUEUES['names'].index(queue_name)
