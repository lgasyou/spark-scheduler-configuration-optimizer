import dataclasses
from typing import List


@dataclasses.dataclass
class ModelStage(object):
    name: str
    input_size_rate: float
    bytes_each_task: int


@dataclasses.dataclass
class CompletedSparkApplication(object):
    name: str
    computed_stage: dict = dataclasses.field(default_factory=dict)
    stages: List[ModelStage] = dataclasses.field(default_factory=list)
