import dataclasses
from typing import List

from optimizer.hyperparameters import QUEUES


@dataclasses.dataclass
class YarnApplication(object):

    location: str
    request_container_count: int
    application_id: str = None
    predicted_delay: int = -1

    @property
    def converted_location(self):
        return queue_name_to_index(self.location) + 1


@dataclasses.dataclass
class FinishedApplication(object):
    started_time: int
    finished_time: int
    elapsed_time: int


@dataclasses.dataclass
class QueueConstraint(object):

    capacity: float
    max_capacity: float


@dataclasses.dataclass
class State(object):
    waiting_apps: List[YarnApplication] = dataclasses.field(default_factory=list)
    running_apps: List[YarnApplication] = dataclasses.field(default_factory=list)
    constraints: List[QueueConstraint] = dataclasses.field(default_factory=list)


def queue_name_to_index(queue_name: str) -> int:
    return QUEUES['names'].index(queue_name)
