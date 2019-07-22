import dataclasses
from typing import List


@dataclasses.dataclass
class Stage(object):
    name: str
    input_ratio: float
    block_size: int


@dataclasses.dataclass
class Application(object):
    name: str
    average_action_process_rates: dict = dataclasses.field(default_factory=dict)
    stages: List[Stage] = dataclasses.field(default_factory=list)
